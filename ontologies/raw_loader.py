from typing import List, Dict, Any, Optional
from owlready2 import World, ThingClass
from owlready2.class_construct import ClassConstruct
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse

import re

def humanize_property_name(name: str) -> str:
    """
    Converte nomi CamelCase, snake_case o nomi tecnici in testo naturale.
    Esempi:
      - "hasPartOf" → "has part of"
      - "located_in" → "located in"
      - "is-part-of" → "is part of"
    """
    # Normalizza underscore e trattini
    s = name.replace("_", " ").replace("-", " ")

    # Inserisce spazio prima delle maiuscole (CamelCase → Camel Case)
    s = re.sub(r"(?<!^)(?=[A-Z])", " ", s)

    # Abbassa tutto
    s = s.lower().strip()

    return s

def ontology_classes_to_dataframe(
    path: str,
    class_iri_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads an OWL/RDF ontology and returns a DataFrame containing one row per class.
    For each class, the following information is extracted:
        - iri, local_name, label, parents, restrictions, equivalent_to, disjoint_with
        - all available annotation properties (IAO_0000115, hasExactSynonym, comment, ...).

    Parameters
    ----------
    path : str
        Local path to the .owl / .rdf file, or a URL.
    ontology_iri : str, optional
        Explicit ontology IRI for get_ontology(). If None, `path` is used.
    class_iri_prefix : str, optional
        If provided, only classes whose IRI starts with this prefix
        (e.g., "http://purl.obolibrary.org/obo/ENVO_") are kept.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with the following columns:
          - iri, local_name, label, parents, restrictions, equivalent_to, disjoint_with
          - one column per annotation property found in the ontology 
            (e.g., label, IAO_0000115, hasExactSynonym, comment,
             created_by, creation_date, hasDbXref, ...).
"""
    world = World()

    parsed = urlparse(path)

    # Case 1 → local file
    if parsed.scheme == "" and Path(path).exists():
        # Use a native filesystem path when loading local files. Using a file:// URI
        # can cause an extra leading slash on Windows (e.g. '/C:/...') which breaks
        # owlready2's file handling. Passing the resolved local path avoids this.
        onto_source = str(Path(path).resolve()) # instead of Path(path).resolve().as_uri()
    # Case 2 → URL http/https/file
    elif parsed.scheme in ("http", "https", "file"):
        onto_source = path
    else:
        raise ValueError(
            f"Cannot determine whether '{path}' is a local file or a valid URL. "
            "Please specify ontology_iri explicitly."
        )

    # Ontology loading
    onto = world.get_ontology(onto_source).load()

    annotation_props = list(world.annotation_properties())
    anno_name_by_iri: Dict[str, str] = {ap.iri: ap.name for ap in annotation_props}

    records: List[Dict[str, Any]] = []

    for cls in onto.classes():
        # if we only want to filter a certain namespace
        if class_iri_prefix is not None and not cls.iri.startswith(class_iri_prefix):
            continue

        row: Dict[str, Any] = {}

        row["iri"] = cls.iri
        row["local_name"] = cls.name  

        # Retrieves the "best label" for the concept
        labels = getattr(cls, "label", [])
        if labels:
            clean_labels = list({lbl.strip() for lbl in labels if lbl.strip()})
            if clean_labels:
                row["label"] = clean_labels[0]
            else:
                row["label"] = cls.name
        else:
            # No available labels: fallback to OWL name
            row["label"] = cls.name
        
        parents = []
        restrictions = []

        for sup in cls.is_a:
            if isinstance(sup, ThingClass):
                # Retrieve the label of the parent class
                parent_label_list = getattr(sup, "label", [])
                if parent_label_list:
                    parent_label = parent_label_list[0].strip()
                else:
                    parent_label = humanize_property_name(sup.name)

                parent_label = " ".join(parent_label.lower().split())
                parents.append(f"sub class of {parent_label}")
            elif isinstance(sup, ClassConstruct):
                try:
                    prop = sup.property
                    if getattr(prop, "label", []):
                        prop_label = prop.label[0].strip()
                    else:
                        prop_label = humanize_property_name(prop.name)
                    prop_label = " ".join(prop_label.lower().split())

                    target = sup.value
                    if getattr(target, "label", []):
                        target_label = target.label[0].strip()
                    else:
                        target_label = humanize_property_name(target.name)
                    target_label = " ".join(target_label.lower().split())

                    restrictions.append(f"{prop_label}: {target_label}")
                except Exception:
                    restrictions.append(str(sup))
        
        # Axioms of equivalence and disjointness
        equiv_str: list[str] = []

        for e in cls.equivalent_to:
            if isinstance(e, ThingClass):
                if getattr(e, "label", []):
                    eq_label = e.label[0].strip()
                else:
                    eq_label = humanize_property_name(e.name)
                eq_label = " ".join(eq_label.lower().split())
                equiv_str.append(f"equivalent to {eq_label}")

            elif isinstance(e, ClassConstruct):
                try:
                    prop = e.property
                    if getattr(prop, "label", []):
                        prop_label = prop.label[0].strip()
                    else:
                        prop_label = humanize_property_name(prop.name)
                    prop_label = " ".join(prop_label.lower().split())

                    target = e.value
                    if getattr(target, "label", []):
                        target_label = target.label[0].strip()
                    else:
                        target_label = humanize_property_name(target.name)
                    target_label = " ".join(target_label.lower().split())

                    equiv_str.append(f"equivalent to ({prop_label}: {target_label})")
                except Exception:
                    equiv_str.append(f"equivalent to {str(e)}")

        disjoint_str: list[str] = []

        for c in cls.disjoints():
            if isinstance(c, ThingClass):
                if getattr(c, "label", []):
                    dj_label = c.label[0].strip()
                else:
                    dj_label = humanize_property_name(c.name)
                dj_label = " ".join(dj_label.lower().split())
                disjoint_str.append(f"disjoint with {dj_label}")
            else:
                disjoint_str.append(f"disjoint with {str(c)}")

        row["parents"] = " | ".join(parents)
        row["restrictions"] = " | ".join(restrictions)
        row["equivalent_to"] = " | ".join(equiv_str)
        row["disjoint_with"] = " | ".join(disjoint_str)

        # Annotation properties: read ALL available ones
        for ap in annotation_props:
            values = ap[cls]  # list of values for this annotation property
            if not values:
                continue
            
            col_name = anno_name_by_iri[ap.iri]
            if col_name in {"label", "equivalent_to", "disjoint_with"}:
                continue
            
            str_values = []
            for v in values:
                # if v is a OWL entity, try to get its label
                if hasattr(v, "label") and getattr(v, "label", []):
                    str_values.append(getattr(v, "label")[0])
                else:
                    str_values.append(str(v))
            
            row[col_name] = " | ".join(str_values)
        
        # Be sure to always have label/comment even if not declared
        if "label" not in row:
            labels = getattr(cls, "label", [])
            if labels:
                row["label"] = " | ".join(labels)
        
        if "comment" not in row:
            comments = getattr(cls, "comment", [])
            if comments:
                row["comment"] = " | ".join(comments)

        records.append(row)

    df = pd.DataFrame(records).fillna("")
    return df