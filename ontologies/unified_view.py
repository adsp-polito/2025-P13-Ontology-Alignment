from dataclasses import dataclass
import pandas as pd


@dataclass
class UnifiedConcept:
    iri: str
    local_name: str
    label: str
    description: str
    synonyms: str
    parents_label: str
    equivalent_to: str
    disjoint_with: str


def build_unified_view(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build a unified, ontology-agnostic view of ontology classes.

    This function takes the raw ontology DataFrame produced by the loader
    (which may contain many ontology-specific annotation columns) and maps
    each class to a compact, standardized schema. For every row (class)
    it extracts:

      - iri:          global identifier of the class
      - local_name:   OWL local name (or fallback name)
      - label:        preferred human-readable label (or local_name if missing)
      - description:  textual definition, chosen from a prioritized set of
                      common definition/description properties
      - synonyms:     concatenation of all columns whose name contains
                      the substring "synonym"
      - parents_label: textual representation of parent classes
      - equivalent_to: natural-language encoding of class equivalence axioms
      - disjoint_with: natural-language encoding of class disjointness axioms

    The output is a DataFrame where each class is represented with the same
    fields, regardless of how the original ontology is modeled. This unified
    view is used as the main semantic interface for downstream components
    (dataset builder and text encoders).
    """

    rows = []

    # Pre-compute synonym columns once
    syn_cols = [c for c in df_raw.columns if "synonym" in c.lower()]

    for row in df_raw.itertuples(index=False):
        iri = row.iri
        local_name = getattr(row, "local_name", getattr(row, "name", "")).strip()
        
        # Best label
        label = (getattr(row, "label", "") or local_name).strip()

        # Description (priority order)
        description = ""
        for cand in ["IAO_0000115", "definition", "description", "hasDefinition", "comment", "note", "IAO_0000116"]:
            val = getattr(row, cand, "")
            if isinstance(val, str) and val.strip():
                description = val.strip()
                break

        # Synonyms: all columns containing "Synonym"
        syn_values = []
        for c in syn_cols:
            # try direct attribute; if not present, try a mangled name (spaces -> underscore)
            val = getattr(row, c, "")
            if isinstance(val, str) and val.strip():
                syn_values.append(val.strip())
        synonyms = " | ".join(syn_values)

        # Parents
        parents_label = getattr(row, "parents", "").strip()

        # Equivalent classes
        equivalent_to = getattr(row, "equivalent_to", "").strip()

        # Disjoint classes
        disjoint_with = getattr(row, "disjoint_with", "").strip()

        rows.append({
            "iri": iri,
            "local_name": local_name,
            "label": label,
            "description": description,
            "synonyms": synonyms,
            "parents_label": parents_label,
            "equivalent_to": equivalent_to,
            "disjoint_with": disjoint_with,
        })

    return pd.DataFrame(rows)