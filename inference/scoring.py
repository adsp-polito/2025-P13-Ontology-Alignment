"""
Cross-encoder scoring module for ontology-attribute alignment inference.

This file contains ONLY Stage-B scoring (high precision), i.e. given:
  - an attribute text
  - a shortlist of candidate ontology classes (IRIs)
  - a mapping iri -> class_text (RICH_TEXT)

it scores (attribute_text, class_text) pairs using a HuggingFace
sequence classification cross-encoder.

Production constraints addressed:
  - The cross-encoder tokenizer/model are loaded ONCE and cached.
  - Scoring is done in batches.

Typical usage:
  scorer = CrossEncoderScorer(model_id="...", device="cuda", max_length=256)
  ranked = scorer.score_candidates(attr_text, candidates, iri2text, top_k=20)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate (IRI) with an associated cross-encoder score."""
    iri: str
    score: float


def scores_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert model logits to a single probability-like score per example.

    Handles common cases:
      - logits shape [B, 1]: sigmoid
      - logits shape [B, 2]: softmax -> positive class prob
      - logits shape [B, K>2]: softmax -> max prob (fallback)

    Returns:
        torch.Tensor shape [B], float32 on same device.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits [B, C], got shape={tuple(logits.shape)}")

    if logits.size(1) == 1:
        return torch.sigmoid(logits[:, 0])
    if logits.size(1) == 2:
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1]
    probs = torch.softmax(logits, dim=1)
    return probs.max(dim=1).values


class CrossEncoderScorer:
    """
    Cached HuggingFace cross-encoder wrapper for scoring (attribute, class_text) pairs.

    Key design goal: load tokenizer/model ONCE, then reuse score_* methods.
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: Optional[str] = None,
        max_length: int = 256,
    ) -> None:
        self.model_id = str(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = int(max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_pairs(
        self,
        left_texts: Sequence[str],
        right_texts: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Score lists of (left_text, right_text) pairs.

        Args:
            left_texts: list of strings (e.g., attribute texts)
            right_texts: list of strings (e.g., class texts)
            batch_size: scoring batch size

        Returns:
            np.ndarray float32 with shape [len(left_texts)].
        """
        if len(left_texts) != len(right_texts):
            raise ValueError(
                f"left_texts and right_texts must have same length: "
                f"{len(left_texts)} != {len(right_texts)}"
            )

        n = len(left_texts)
        if n == 0:
            return np.zeros((0,), dtype="float32")

        scores_all: List[torch.Tensor] = []

        for start in range(0, n, int(batch_size)):
            a = ["" if t is None else str(t) for t in left_texts[start : start + int(batch_size)]]
            b = ["" if t is None else str(t) for t in right_texts[start : start + int(batch_size)]]

            enc = self.tokenizer(
                a,
                b,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            out = self.model(**enc)
            batch_scores = scores_from_logits(out.logits)
            scores_all.append(batch_scores.detach().cpu())

        scores = torch.cat(scores_all, dim=0).numpy().astype("float32", copy=False)
        return scores

    @torch.no_grad()
    def score_candidates(
        self,
        attr_text: str,
        candidates: Sequence,  # expects objects with attribute .iri (e.g., retrieval.Candidate)
        *,
        iri2text: Dict[str, str],
        top_k: Optional[int] = None,
        batch_size: int = 32,
    ) -> List[ScoredCandidate]:
        """
        Score a shortlist of candidate IRIs for ONE attribute.

        Args:
            attr_text: the attribute string
            candidates: sequence of candidates with .iri
            iri2text: mapping iri -> RICH_TEXT
            top_k: optional cap on how many candidates to score (takes the first top_k)
            batch_size: scoring batch size

        Returns:
            List[ScoredCandidate] sorted by descending score.
        """
        a = ("" if attr_text is None else str(attr_text)).strip()
        if not a:
            return []

        cand_list = list(candidates)
        if top_k is not None:
            cand_list = cand_list[: int(top_k)]

        left: List[str] = []
        right: List[str] = []
        kept_iris: List[str] = []

        for c in cand_list:
            iri = str(getattr(c, "iri"))
            txt = iri2text.get(iri)
            if txt is None:
                continue
            left.append(a)
            right.append(str(txt))
            kept_iris.append(iri)

        if not kept_iris:
            return []

        scores = self.score_pairs(left, right, batch_size=int(batch_size))
        ranked_idx = np.argsort(-scores)

        return [ScoredCandidate(iri=kept_iris[i], score=float(scores[i])) for i in ranked_idx]

    def best_prediction(
        self,
        attr_text: str,
        candidates: Sequence,
        *,
        iri2text: Dict[str, str],
        top_k: Optional[int] = None,
        batch_size: int = 32,
    ) -> Optional[ScoredCandidate]:
        """
        Convenience helper: return the best scored candidate (or None).
        """
        ranked = self.score_candidates(
            attr_text,
            candidates,
            iri2text=iri2text,
            top_k=top_k,
            batch_size=batch_size,
        )
        return ranked[0] if ranked else None


def attach_scores_to_candidates(
    scored: Sequence[ScoredCandidate],
) -> List[Tuple[str, float]]:
    """
    Utility to convert scored candidates to a simple (iri, score) list.

    Returns:
        List of tuples (iri, score), already ranked.
    """
    return [(s.iri, float(s.score)) for s in scored]