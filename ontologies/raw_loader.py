from typing import List, Dict, Any, Optional
from owlready2 import World, ThingClass
from owlready2.class_construct import ClassConstruct
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
from rdflib import Graph, RDF
from rdflib.namespace import SKOS, RDFS
import logging

import re


def _suppress_rdflib_datetime_noise() -> None:
    """Suppress rdflib term casting noise (empty xsd:dateTime, etc.).

    rdflib logs messages like:
    "Failed to convert Literal lexical form to value ... Invalid isoformat string: ''"
    repeatedly during parsing or Literal conversion. They are harmless for our
    usage and just pollute stdout/stderr. This filter turns them off locally.
    """
    logger = logging.getLogger("rdflib.term")
    # Ensure only errors or above are emitted
    logger.setLevel(logging.ERROR)

    class _DropCastNoise(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            msg = record.getMessage()
            if "Failed to convert Literal lexical form to value" in msg:
                return False
            if "Invalid isoformat string" in msg:
                return False
            return True

    # Prevent propagation to root handlers and add our filter defensively
    logger.propagate = False
    logger.addFilter(_DropCastNoise())

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

    classes = list(onto.classes())

    # If OWL classes exist, keep current behaviour
    if classes:
        for cls in classes:
            if class_iri_prefix is not None and not cls.iri.startswith(class_iri_prefix):
                continue

            row: Dict[str, Any] = {}
            row["iri"] = cls.iri
            row["local_name"] = cls.name

            labels = getattr(cls, "label", [])
            if labels:
                clean_labels = list({lbl.strip() for lbl in labels if lbl.strip()})
                row["label"] = clean_labels[0] if clean_labels else cls.name
            else:
                row["label"] = cls.name

            parents = []
            restrictions = []

            for sup in cls.is_a:
                if isinstance(sup, ThingClass):
                    parent_label_list = getattr(sup, "label", [])
                    parent_label = parent_label_list[0].strip() if parent_label_list else humanize_property_name(sup.name)
                    parent_label = " ".join(parent_label.lower().split())
                    parents.append(f"sub class of {parent_label}")
                elif isinstance(sup, ClassConstruct):
                    try:
                        prop = sup.property
                        prop_label = prop.label[0].strip() if getattr(prop, "label", []) else humanize_property_name(prop.name)
                        prop_label = " ".join(prop_label.lower().split())

                        target = sup.value
                        target_label = target.label[0].strip() if getattr(target, "label", []) else humanize_property_name(target.name)
                        target_label = " ".join(target_label.lower().split())

                        restrictions.append(f"{prop_label}: {target_label}")
                    except Exception:
                        restrictions.append(str(sup))

            equiv_str: list[str] = []
            for e in cls.equivalent_to:
                if isinstance(e, ThingClass):
                    eq_label = e.label[0].strip() if getattr(e, "label", []) else humanize_property_name(e.name)
                    eq_label = " ".join(eq_label.lower().split())
                    equiv_str.append(f"equivalent to {eq_label}")
                elif isinstance(e, ClassConstruct):
                    try:
                        prop = e.property
                        prop_label = prop.label[0].strip() if getattr(prop, "label", []) else humanize_property_name(prop.name)
                        prop_label = " ".join(prop_label.lower().split())

                        target = e.value
                        target_label = target.label[0].strip() if getattr(target, "label", []) else humanize_property_name(target.name)
                        target_label = " ".join(target_label.lower().split())

                        equiv_str.append(f"equivalent to ({prop_label}: {target_label})")
                    except Exception:
                        equiv_str.append(f"equivalent to {str(e)}")

            disjoint_str: list[str] = []
            for c in cls.disjoints():
                if isinstance(c, ThingClass):
                    dj_label = c.label[0].strip() if getattr(c, "label", []) else humanize_property_name(c.name)
                    dj_label = " ".join(dj_label.lower().split())
                    disjoint_str.append(f"disjoint with {dj_label}")
                else:
                    disjoint_str.append(f"disjoint with {str(c)}")

            row["parents"] = " | ".join(parents)
            row["restrictions"] = " | ".join(restrictions)
            row["equivalent_to"] = " | ".join(equiv_str)
            row["disjoint_with"] = " | ".join(disjoint_str)

            for ap in annotation_props:
                values = ap[cls]
                if not values:
                    continue

                col_name = anno_name_by_iri[ap.iri]
                if col_name in {"label", "equivalent_to", "disjoint_with"}:
                    continue

                str_values = []
                for v in values:
                    if hasattr(v, "label") and getattr(v, "label", []):
                        str_values.append(getattr(v, "label")[0])
                    else:
                        str_values.append(str(v))

                row[col_name] = " | ".join(str_values)

            if "label" not in row:
                labels = getattr(cls, "label", [])
                if labels:
                    row["label"] = " | ".join(labels)

            if "comment" not in row:
                comments = getattr(cls, "comment", [])
                if comments:
                    row["comment"] = " | ".join(comments)

            records.append(row)

        return pd.DataFrame(records).fillna("")

    # No OWL classes: parse as SKOS with rdflib
    _suppress_rdflib_datetime_noise()
    g = Graph()
    g.parse(onto_source)

    for iri in g.subjects(RDF.type, SKOS.Concept):
        iri_str = str(iri)
        if class_iri_prefix is not None and not iri_str.startswith(class_iri_prefix):
            continue

        def _first(preds: List[Any]) -> str:
            for p in preds:
                vals = list(g.objects(iri, p))
                if vals:
                    return str(vals[0])
            return ""

        def _all(pred) -> List[str]:
            return [str(v) for v in g.objects(iri, pred)]

        pref_label = _first([SKOS.prefLabel, RDFS.label]) or iri_str
        description = _first([SKOS.definition, SKOS.scopeNote, RDFS.comment])
        synonyms = " | ".join(_all(SKOS.altLabel))
        parents = " | ".join([f"broader concept {v}" for v in _all(SKOS.broader)])
        equivalents = " | ".join([f"exact match {v}" for v in _all(SKOS.exactMatch) + _all(SKOS.closeMatch)])

        records.append({
            "iri": iri_str,
            "local_name": iri_str.split("#")[-1].split("/")[-1],
            "label": pref_label,
            "description": description,
            "synonyms": synonyms,
            "parents": parents,
            "restrictions": "",
            "equivalent_to": equivalents,
            "disjoint_with": "",
        })

    return pd.DataFrame(records).fillna("")