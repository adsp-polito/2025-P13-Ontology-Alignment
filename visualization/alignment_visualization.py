import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pyvis.network import Network

def visualize_alignments(
    df: pd.DataFrame,
    animated : bool = False,
    source_ontology_name: str = "sweet",
    target_ontology_name: str = "envo"
) -> None:
    """Visualize the alignments extracted from the alignment file.

    Parameters:
        df (pd.DataFrame): dataframe with alignments
        animated (bool, optional): True for animated visualization, False otherwise. Defaults to False.
        source_ontology_name (str, optional): source ontology name. Defaults to "sweet".
        target_ontology_name (str, optional): target ontology name. Defaults to "envo".
    """
    
    if not animated:

        g = nx.Graph()
        for _, row in df.iterrows():
            c1 = row["source_label"]
            c2 = row["target_label"]
            g.add_node(c1, ontology="source")
            g.add_node(c2, ontology="target")
            g.add_edge(c1, c2)
    
        pos = nx.spring_layout(g, seed=42)

        node_colors = ["lightblue" if g.nodes[n]["ontology"]=="source" else "salmon" for n in g.nodes()]

        plt.figure(figsize=(10,7))

        nx.draw(
            g,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=200,
            font_size=4,
            edge_color="gray"
        )

        plt.title("Ontology Alignment Graph")
        plt.show()
    
    else:

        net = Network(height="600px",
                      width= "100%",
                      bgcolor="#ffffff",
                      font_color="black",
                      notebook=False)
        
        color_map = {
            "source": "lightblue",
            "target": "salmon"
        }

        for _, row in df.iterrows():
            c1 = row["source_label"]
            c2 = row["target_label"]
            net.add_node(c1, label=c1, color=color_map["source"], title=f"Ontology: {source_ontology_name}")
            net.add_node(c2, label=c2, color=color_map["target"], title=f"Ontology: {target_ontology_name}")
            net.add_edge(c1, c2)
            
        
        # Enable physics
        net.show_buttons(filter_=['physics'])
        net.force_atlas_2based()

        # Export and show
        net.show("ontology_alignment_graph_interactive.html", notebook=False)
        print("Interactive graph saved as ontology_alignment_graph_interactive.html")

if __name__ == "__main__":
    df = pd.read_csv("outputs/envo_sweet_training.csv")
    df = df[df["match"]==1.0]
    visualize_alignments(df, animated=True, source_ontology_name="sweet", target_ontology_name="envo")