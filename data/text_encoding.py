from __future__ import annotations
from typing import Mapping, Any


def build_source_text(
    row: Mapping[str, Any],
    *,
    use_label: bool = True,
    use_description: bool = False,
    use_synonyms: bool = False,
    use_parents: bool = False,
    use_equivalent: bool = False,
    use_disjoint: bool = False,
) -> str:
    """
    Build a configurable SHORT_TEXT for the source_attribute side.

    The goal is to simulate different levels of information that may be
    available from study attributes, and to support ablation experiments
    (e.g., label-only vs. label+description, etc.).

    Parameters
    ----------
    use_label : bool
        Include the concept label (main field). Default True.
    use_description : bool
        Include the textual definition / description. Default False.
    use_synonyms : bool
        Include synonyms, if available. Default False.
    use_parents : bool
        Include parent class labels. Default False.
    use_equivalent : bool
        Include textual equivalents (equivalent_to). Default False.
    use_disjoint : bool
        Include textual disjointness axioms (disjoint_with). Default False.

    Returns
    -------
    str
        A short textual representation assembled according to the flags.
    """

    parts: list[str] = []

    # LABEL
    if use_label:
        label = str(row.get("label", "")).strip()
        if label:
            parts.append(f'label: {label}')
        else:
            # reasonable fallback
            local_name = str(row.get("local_name", "")).strip()
            if local_name:
                parts.append(f'local name: {local_name}')

    # DESCRIPTION
    if use_description:
        desc = str(row.get("description", "")).strip()
        if desc:
            parts.append(f'description: {desc}')

    # SYNONYMS
    if use_synonyms:
        syn = str(row.get("synonyms", "")).strip()
        if syn:
            parts.append(f"synonyms: {syn}")

    # PARENTS
    if use_parents:
        parents = str(row.get("parents_label", "")).strip()
        if parents:
            parts.append(f"parents: {parents}")

    # EQUIVALENT CLASSES
    if use_equivalent:
        equivalent_to = str(row.get("equivalent_to", "")).strip()
        if equivalent_to:
            parts.append(f"equivalent to: {equivalent_to}")

    # DISJOINT CLASSES
    if use_disjoint:
        disjoint_with = str(row.get("disjoint_with", "")).strip()
        if disjoint_with:
            parts.append(f"disjoint with: {disjoint_with}")

    short_text = "; ".join(parts).strip()
    return short_text


def build_target_text(row: Mapping[str, Any]) -> str:
    """
    Build the RICH_TEXT representation for the target_concept side,
    starting from a row of the unified view.

    Fields used (if present):
      - label: main concept name
      - description: textual definition / comment
      - synonyms: synonyms string (pipe-separated)
      - parents_label: parent class labels (pipe-separated)
      - equivalent_to: textual equivalent classes
      - disjoint_with: textual disjoint classes

    The target side is designed to be semantically rich, so the model
    sees the name, definition, synonyms and structural context.
    """
    parts: list[str] = []

    # LABEL
    label = str(row.get("label", "")).strip()
    if label:
            parts.append(f'label: {label}')
    else:
        # reasonable fallback
        local_name = str(row.get("local_name", "")).strip()
        if local_name:
            parts.append(f'local name: {local_name}')

    # DESCRIPTION
    description = str(row.get("description", "")).strip()
    if description:
        parts.append(f'description: {description}')

    # SYNONYMS
    synonyms = str(row.get("synonyms", "")).strip()
    if synonyms:
        parts.append(f"Synonyms: {synonyms}")

    # PARENTS
    parents_label = str(row.get("parents_label", "")).strip()
    if parents_label:
        parts.append(f"Parents: {parents_label}")

    # EQUIVALENT CLASSES
    equivalent_to = str(row.get("equivalent_to", "")).strip()
    if equivalent_to:
        parts.append(f"Equivalent to: {equivalent_to}")

    # DISJOINT CLASSES
    disjoint_with = str(row.get("disjoint_with", "")).strip()
    if disjoint_with:
        parts.append(f"Disjoint with: {disjoint_with}")

    rich_text = "; ".join(parts).strip()
    return rich_text