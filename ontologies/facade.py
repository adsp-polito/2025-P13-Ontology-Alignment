# ontologies/facade.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any, Mapping

import pandas as pd

from .raw_loader import ontology_classes_to_dataframe
from .unified_view import build_unified_view
from data.text_encoding import build_source_text, build_target_text


@dataclass
class SourceTextConfig:
    """
    Configuration for building the source SHORT_TEXT.

    The flags determine which semantic attributes are included
    in the textual representation of the source ontology.
    """
    use_label: bool = True
    use_description: bool = True
    use_synonyms: bool = False
    use_parents: bool = False
    use_equivalent: bool = False
    use_disjoint: bool = False


def ontology_loader(
    src_path_or_iri: str,
    tgt_path_or_iri: str,
    *,
    src_class_prefix: Optional[str] = None,
    tgt_class_prefix: Optional[str] = None,
    src_text_config: Optional[SourceTextConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end ontology loading pipeline for two ontologies.

    1. Loads both ontologies (OWL/RDF) using the raw loader.
    2. Builds the unified view (label, description, synonyms, parents_label,
       equivalent_to, disjoint_with).
    3. Constructs:
       - SHORT_TEXT for the source (configurable via SourceTextConfig)
       - RICH_TEXT for the target (using build_target_text)
    4. Returns two minimal DataFrames, ready to be joined with the reference
       alignment via the IRI.
    """
    if src_text_config is None:
        src_text_config = SourceTextConfig()

    # 1) Raw loader for the two ontologies
    df_src_raw = ontology_classes_to_dataframe(
        path=src_path_or_iri,
        class_iri_prefix=src_class_prefix,
    )

    df_tgt_raw = ontology_classes_to_dataframe(
        path=tgt_path_or_iri,
        class_iri_prefix=tgt_class_prefix,
    )

    # 2) Unified view
    df_src_uni = build_unified_view(df_src_raw)
    df_tgt_uni = build_unified_view(df_tgt_raw)

    # 3) String construction

    # Source: SHORT_TEXT configurable
    def _build_src_text(row: Mapping[str, Any]) -> str:
        return build_source_text(
            row,
            use_label=src_text_config.use_label,
            use_description=src_text_config.use_description,
            use_synonyms=src_text_config.use_synonyms,
            use_parents=src_text_config.use_parents,
            use_equivalent=src_text_config.use_equivalent,
            use_disjoint=src_text_config.use_disjoint,
        )

    df_src_uni["text"] = df_src_uni.apply(_build_src_text, axis=1)

    # Target: fixed RICH_TEXT
    df_tgt_uni["text"] = df_tgt_uni.apply(build_target_text, axis=1)

    # 4) Riduci ai campi minimi per il join col reference alignment
    df_src_text = df_src_uni[["iri", "local_name", "label", "text", "synonyms"]].copy()
    df_tgt_text = df_tgt_uni[["iri", "local_name", "label", "text", "synonyms"]].copy()
    
    df_src_renamed = df_src_text.rename(columns={"iri": "source_iri", "label": "source_label", "text": "source_text", "synonyms": "source_synonyms"})
    df_tgt_renamed = df_tgt_text.rename(columns={"iri": "target_iri", "label": "target_label", "text": "target_text", "synonyms": "target_synonyms"})

    return df_src_renamed, df_tgt_renamed