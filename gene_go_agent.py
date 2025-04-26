import networkx as nx
import pandas as pd
from fuzzywuzzy import process
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# Load graph
G = nx.read_graphml("gene_go_network.graphml")

# Load gene-pathway links
gene_pathway_df = pd.read_csv("data/gene_pathway_links.csv")


def get_genes_by_disease(disease_name: str) -> str:
    """Find all genes associated with a disease via KEGG pathways

    Args:
        disease_name (str): The name of the disease

    Returns:
        str: A string listing the genes associated with the disease
    """
    all_diseases = gene_pathway_df["pathway"].unique().tolist()

    # Find the closest match to the disease name
    best_match, score = process.extractOne(disease_name, all_diseases)
    if score < 80:
        return f"No close match found for the disease name '{disease_name}'. Please check the spelling or provide a more specific name."

    matches = gene_pathway_df[
        gene_pathway_df["pathway"].str.lower() == best_match.lower()
    ]
    gene_list = matches["gene_id"].unique().tolist()
    if not gene_list:
        return f"No genes found for the disease: {disease_name}"
    return f"Genes involved in {disease_name}: {', '.join(gene_list)}"


@tool
def get_gene_go_terms(gene_symbol: str) -> str:
    """Find all GO terms associated with a gene

    Args:
        gene_symbol (str): the gene symbol

    Returns:
        _str: A string listing the GO terms associated with the gene
    """
    if gene_symbol not in G.nodes():
        return f"Gene {gene_symbol} not found in the knowledge base."

    go_terms = []
    go_terms = [
        n for n in G.neighbors(gene_symbol) if G.nodes[n].get("type") == "go_term"
    ]
    if not go_terms:
        return f"No GO terms found for gene {gene_symbol}."

    return f"Gene {gene_symbol} is associated with GO terms: {', '.join(go_terms)}"


@tool
def get_genes_by_go_term(go_term: str) -> str:
    """Find all genes with a specific GO term

    Args:
        go_term (str): the GO term ID

    Returns:
        str: A string listing the genes associated with the GO term
    """
    if go_term not in G.nodes():
        return f"GO term {go_term} not found in the knowledge base."

    genes = []
    for neighbor in G.neighbors(go_term):
        if G.nodes[neighbor].get("type") == "gene":
            genes.append(neighbor)

    return f"GO term {go_term} is associated with genes: {', '.join(genes)}"


@tool
def get_diseases_by_gene(gene_id: str) -> str:
    """Find all diseases a gene is involved in via KEGG pathways

    Args:
        gene_id (str): The gene ID

    Returns:
        str: A string listing the diseases associated with the gene
    """
    matches = gene_pathway_df[gene_pathway_df["gene_id"].str.lower() == gene_id.lower()]
    disease_list = matches["pathway"].unique().tolist()
    if not disease_list:
        return f"No diseases found for gene {gene_id}."
    return f"Gene {gene_id} is involved in the following diseases: {', '.join(disease_list)}"


@tool
def generate_hypothesis(input_str: str) -> str:
    """enerate a hypothesis about a gene's potential role in a disease

    Args:
        input_str (str): Input string containing gene ID and disease name in the format "gene_id, disease_name"

    Returns:
        str: A hypothesis about the gene's potential role in the disease
    """

    try:
        gene_id, disease_name = [x.strip() for x in input_str.split(",", 1)]
    except ValueError:
        return "Please provide input in format: 'gene_id, disease_name'"

    # Check gene-disease association
    diseases = get_diseases_by_gene(gene_id)
    if disease_name.lower() not in diseases.lower():
        return f"Gene {gene_id} is not directly associated with {disease_name} in pathway data."

    # Get GO terms
    go_terms = get_gene_go_terms(gene_id)
    if "not found" in go_terms:
        return go_terms

    # Generate hypothesis
    return (
        f"Hypothesis for {gene_id} in {disease_name}:\n"
        f"1. Pathway data confirms association\n"
        f"2. Gene's GO terms suggest potential mechanisms: {go_terms}\n"
        f"3. Potential roles: Based on its annotations, {gene_id} might contribute to {disease_name} "
        "through these biological processes or molecular functions."
    )


@tool
def find_shared_mechanisms(genes_input: str) -> dict:
    """Find shared GO terms between two genes that might indicate common mechanisms

    Args:
        genes_input (str): Input string containing two gene symbols separated by a comma
        (e.g., "SOX9, TP53")

    Returns:
        dict: A dictionary containing the two genes, their shared GO terms, and the count of shared terms
    """
    genes = genes_input.split(",")
    if len(genes) != 2:
        return {
            "error": "Please provide exactly two gene symbols, separated by a comma."
        }
    gene1, gene2 = genes[0].strip(), genes[1].strip()
    if gene1 not in G or gene2 not in G:
        return {"error": "One or both genes not found in graph"}

    # Get GO terms for both genes
    go1 = set(n for n in G.neighbors(gene1) if G.nodes[n].get("type") == "go_term")
    go2 = set(n for n in G.neighbors(gene2) if G.nodes[n].get("type") == "go_term")

    shared = list(go1 & go2)

    return {"genes": [gene1, gene2], "shared_go_terms": shared, "count": len(shared)}


tools = [
    Tool(
        name="GetDiseasesByGene",
        func=get_diseases_by_gene,
        description="Finds diseases associated with a gene. Input: gene ID (e.g., 'hsa:1234')",
    ),
    Tool(
        name="GetGenesByDisease",
        func=get_genes_by_disease,
        description="Finds genes associated with a disease. Input: disease name (e.g., 'Alzheimer disease')",
    ),
    Tool(
        name="GetGeneGOTerms",
        func=get_gene_go_terms,
        description="Finds GO terms for a gene. Input: gene symbol (e.g., 'APOE')",
    ),
    Tool(
        name="FindSharedMechanisms",
        func=find_shared_mechanisms,
        description="Find shared GO terms between two genes that might indicate common mechanisms. Input: two gene symbols (e.g., 'APOE, TP53'). Returns shared GO terms and their count.",
    ),
    Tool(
        name="GenerateHypothesis",
        func=generate_hypothesis,
        description="Generates a hypothesis about gene-disease mechanisms. Input: gene ID and disease name",
    ),
]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = """You are a biomedical research assistant with deep knowledge of gene function analysis.
        For mechanism questions, follow this protocol:
        1. Verify gene-disease association
        2. Analyze gene's GO term profile
        3. Compare with other disease genes
        4. Generate mechanistic hypothesis
        Use the full power of the GO graph when available."""

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="generate",
    agent_kwargs={"prefix": prompt},
)

query = "What diseases might APOE be associated with?"
# query = "Which genes are involved in Alzheimer disease?"
# query = "Which genes are involved in Parkinson's disease?"
# query = "What diseases is the gene hsa:4535 (APP) involved in?"
# query = "How might SOX9 be involved in Colorectal cancer?"
query = "Which genes are related to type II diabetes mellitus?"
query = "Which genes are related to type II diabetes?"
# query = "What type II diabetes genes are associated with the GO term GO:0005975 (carbohydrate metabolic process)?"

response = agent_executor.invoke({"input": query})
print(response["output"])
