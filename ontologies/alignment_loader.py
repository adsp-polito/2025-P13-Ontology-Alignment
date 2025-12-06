from rdflib import RDF, Graph, Namespace
import pandas as pd

def load_alignment_file(path: str) -> pd.DataFrame:
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
        # measure = graph.value(cell, ALIGN.alignmentmeasure)
        alignments.append({
            "source_iri": e1, # swapped because envo is more informative than sweet
            "target_iri": e2,
            # "label": float(measure) if measure is not None else None,
            "match": 1
        })
    return pd.DataFrame(alignments)