from rdflib import RDF, Graph, Namespace
import pandas as pd
from typing import Optional

def load_alignment_file(
        path: str,
        source_iri_prefix: Optional[str] = None,
        target_iri_prefix: Optional[str] = None
) -> pd.DataFrame:
    """
    Load a reference alignment file in RDF format and return a DataFrame
    with the correspondences.

    The output DataFrame contains the following columns:
      - source_iri: IRI of the source class
      - target_iri: IRI of the target class
      - label: alignment measure / confidence score (float)
    """
    graph = Graph()
    graph.parse(path)
    ALIGN = Namespace("http://knowledgeweb.semanticweb.org/heterogeneity/")
    alignments = []
    for cell in graph.subjects(RDF.type, ALIGN.alignmentCell):
        e1 = str(graph.value(cell, ALIGN.alignmententity1)) # envo entity
        e2 = str(graph.value(cell, ALIGN.alignmententity2)) # sweet entity
        if (source_iri_prefix and not e1.startswith(source_iri_prefix)) or \
              (target_iri_prefix and not e2.startswith(target_iri_prefix)):
            continue
        # measure = graph.value(cell, ALIGN.alignmentmeasure)
        alignments.append({
            "source_iri": e2, # less informative ontology as source
            "target_iri": e1, # more informative ontology as target
            # "label": float(measure) if measure is not None else None,
            "sample_type": "positive", 
            "match": 1
        })
    return pd.DataFrame(alignments)