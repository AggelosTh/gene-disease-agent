"""Plotting functions for gene ontology analysis."""

import os

import matplotlib.pyplot as plt
import pandas as pd

from config import ROOT_DIR
from network import load_graphml_graph, visualize_subgraph


def plot_top_go_terms(path):
    """Plot the top 10 most frequent GO terms."""
    df_goa = pd.read_csv(os.path.join(ROOT_DIR, path))
    top_go_terms = df_goa["go_term"].value_counts().head(10)

    top_go_terms.head(10).plot(kind="barh", figsize=(8, 6), color="skyblue")
    plt.xlabel("Number of Genes")
    plt.title("Top 10 Most Frequent GO Terms")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, "top_go_terms.png"), dpi=300)
    plt.show()


# Load the graph and visualize it
G = load_graphml_graph("gene_go_network.graphml")
visualize_subgraph(G)
# Plot the top GO terms
plot_top_go_terms("data/gene_go_links.csv")
