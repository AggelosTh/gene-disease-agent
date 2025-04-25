import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def build_graph(data_path):
    df_goa = pd.read_csv(data_path)

    # Create the graph
    G = nx.Graph()

    # Add nodes and edges from the GAF data
    for _, row in df_goa.iterrows():
        G.add_node(row["gene_symbol"], type="gene")
        G.add_node(row["go_term"], type="go_term")
        G.add_edge(row["gene_symbol"], row["go_term"], relation="go_annotation")

    nx.write_graphml(G, "gene_go_network.graphml")
    nx.write_gexf(G, "gene_go_network.gexf")

    with open("gene_go_relationships.txt", "w") as f:
        f.write("gene_symbol\tgo_term\trelation\n")
        for u, v, data in G.edges(data=True):
            gene = u if G.nodes[u]["type"] == "gene" else v
            go_term = v if G.nodes[v]["type"] == "go_term" else u
            f.write(f"{gene}\t{go_term}\t{data['relation']}\n")

    print(f"Graph Statistics:")
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")
    print(
        f"Gene nodes: {len([n for n, attr in G.nodes(data=True) if attr.get('type') == 'gene'])}"
    )
    print(
        f"GO term nodes: {len([n for n, attr in G.nodes(data=True) if attr.get('type') == 'go_term'])}"
    )


def load_graphml_graph(graph_path):
    G_graphml = nx.read_graphml("gene_go_network.graphml")
    return G_graphml


def visualize_subgraph(G, max_nodes=100):
    if G.number_of_nodes() > max_nodes:
        # Get a subgraph with some of the highest-degree gene nodes
        gene_nodes = [n for n, attr in G.nodes(data=True) if attr.get("type") == "gene"]
        gene_nodes_by_degree = sorted(
            gene_nodes, key=lambda x: G.degree(x), reverse=True
        )[:20]

        # Get connected GO terms
        connected_go_terms = set()
        for gene in gene_nodes_by_degree:
            connected_go_terms.update(
                [n for n in G.neighbors(gene) if G.nodes[n].get("type") == "go_term"]
            )

        # Limit GO terms if too many
        if len(connected_go_terms) > (max_nodes - len(gene_nodes_by_degree)):
            connected_go_terms = list(connected_go_terms)[
                : max_nodes - len(gene_nodes_by_degree)
            ]

        # Create subgraph
        nodes_to_include = gene_nodes_by_degree + list(connected_go_terms)
        subG = G.subgraph(nodes_to_include)
    else:
        subG = G

    pos = nx.spring_layout(subG, seed=42)
    plt.figure(figsize=(12, 10))

    # Draw nodes by type
    gene_nodes = [n for n, attr in subG.nodes(data=True) if attr.get("type") == "gene"]
    go_nodes = [n for n, attr in subG.nodes(data=True) if attr.get("type") == "go_term"]

    nx.draw_networkx_nodes(
        subG,
        pos,
        nodelist=gene_nodes,
        node_color="skyblue",
        node_size=500,
        label="Genes",
    )
    nx.draw_networkx_nodes(
        subG,
        pos,
        nodelist=go_nodes,
        node_color="lightgreen",
        node_size=300,
        label="GO Terms",
    )

    # Draw edges and labels
    nx.draw_networkx_edges(subG, pos, alpha=0.5)

    # Create cleaner labels
    labels = {}
    for node in subG.nodes():
        if subG.nodes[node].get("type") == "gene":
            labels[node] = node
        else:
            labels[node] = node[:15] + "..." if len(node) > 15 else node

    nx.draw_networkx_labels(subG, pos, labels=labels, font_size=8)

    plt.title("Gene-GO Term Network")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("gene_go_network.png", dpi=300)
    plt.show()


G = load_graphml_graph("gene_go_network.graphml")
visualize_subgraph(G)
