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


@tool
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


@tool
def find_downstream_genes(gene_id: str) -> dict:
    """Identify genes likely to be downstream in pathways.
    Input: gene_id (e.g., 'hsa:6662')
    Output: {'target': gene_id, 'downstream': [gene1, gene2...]}"""

    # 1. Find all pathways containing the gene
    pathways = gene_pathway_df[gene_pathway_df["gene_id"] == gene_id][
        "pathway"
    ].unique()

    # 2. Get all other genes in these pathways
    downstream = []
    for pathway in pathways:
        pathway_genes = gene_pathway_df[gene_pathway_df["pathway"] == pathway][
            "gene_id"
        ].tolist()
        # Simple heuristic: genes appearing after our target in pathway data
        if gene_id in pathway_genes:
            idx = pathway_genes.index(gene_id)
            downstream.extend(pathway_genes[idx + 1 :])  # Genes after our target

    return {
        "target": gene_id,
        "downstream": list(set(downstream)),  # Remove duplicates
        "pathways": list(pathways),
    }


@tool
def analyze_downstream_effects(gene_id: str) -> str:
    """Predict functional effects on downstream genes using GO term analysis.
    Input: gene_id (e.g., 'hsa:6662')
    Output: Analysis report"""

    # 1. Find downstream genes
    downstream_data = find_downstream_genes(gene_id)
    if not downstream_data["downstream"]:
        return f"No downstream genes found for {gene_id} in pathway data."

    # 2. Analyze pathway context
    pathway_counts = {}
    for pathway in downstream_data["pathways"]:
        pathway_counts[pathway] = len(
            gene_pathway_df[gene_pathway_df["pathway"] == pathway]
        )

    # 3. Generate report
    report = [
        f"Downstream analysis for {gene_id}:",
        f"- Found in {len(downstream_data['pathways'])} pathways",
        f"- Potentially affects {len(downstream_data['downstream'])} downstream genes",
        "\nPathway context:",
    ]

    # Sort pathways by gene count (complexity)
    sorted_pathways = sorted(pathway_counts.items(), key=lambda x: x[1], reverse=True)
    for pathway, count in sorted_pathways:
        report.append(f"- {pathway}: {count} total genes")

    # Show some downstream genes
    report.append("\nTop downstream genes:")
    for gene in downstream_data["downstream"][:5]:  # Show top 5 downstream genes
        report.append(f"- {gene}")

    report.append("\nNote: This analysis is based on pathway topology from KEGG data.")

    return "\n".join(report)


@tool
def get_pathway_info(pathway_name: str) -> str:
    """Get detailed information about a pathway.
    Input: pathway name (e.g., 'Alzheimer disease')
    Output: Details about genes in the pathway"""

    # Check if pathway exists
    if pathway_name not in gene_pathway_df["pathway"].values:
        # Try fuzzy matching
        all_pathways = gene_pathway_df["pathway"].unique()
        best_match, score = process.extractOne(pathway_name, all_pathways)

        if score < 80:
            return f"Pathway '{pathway_name}' not found. Please check the spelling."

        pathway_name = best_match

    # Get genes in pathway
    pathway_genes = gene_pathway_df[gene_pathway_df["pathway"] == pathway_name][
        "gene_id"
    ].tolist()

    report = [
        f"Pathway: {pathway_name}",
        f"Total genes: {len(set(pathway_genes))}",
        "\nGenes (in pathway order):",
    ]

    # Show first 10 genes to keep report concise
    for gene in pathway_genes[:10]:
        report.append(f"- {gene}")

    if len(pathway_genes) > 10:
        report.append(f"... and {len(pathway_genes) - 10} more genes")

    return "\n".join(report)


@tool
def find_connecting_pathways(gene_ids: str) -> str:
    """Find pathways that connect two or more genes.
    Input: comma-separated gene IDs (e.g., 'hsa:6662, hsa:7124')
    Output: List of shared pathways"""

    gene_list = [g.strip() for g in gene_ids.split(",")]

    # Validate gene IDs
    for gene in gene_list:
        if gene not in gene_pathway_df["gene_id"].values:
            return f"Gene ID '{gene}' not found in pathway data."

    # Find pathways for each gene
    gene_pathways = {}
    for gene in gene_list:
        gene_pathways[gene] = set(
            gene_pathway_df[gene_pathway_df["gene_id"] == gene]["pathway"]
        )

    # Find shared pathways
    shared_pathways = set.intersection(*gene_pathways.values())

    if not shared_pathways:
        return f"No shared pathways found connecting the genes: {', '.join(gene_list)}"

    # Generate report
    report = [
        f"Connecting pathways for genes: {', '.join(gene_list)}",
        f"\nFound {len(shared_pathways)} shared pathways:",
    ]

    for pathway in sorted(shared_pathways):
        # Get positions of each gene in the pathway
        pathway_genes = gene_pathway_df[gene_pathway_df["pathway"] == pathway][
            "gene_id"
        ].tolist()
        positions = []
        for gene in gene_list:
            if gene in pathway_genes:
                pos = pathway_genes.index(gene) + 1  # 1-based indexing
                positions.append(f"{gene} (pos {pos})")

        report.append(f"- {pathway}: {', '.join(positions)}")

    return "\n".join(report)


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
    Tool(
        name="FindDownstreamGenes",
        func=find_downstream_genes,
        description="Identifies genes likely to be downstream in pathways. Input: gene ID (e.g., 'hsa:6662'). Returns target gene and downstream genes.",
    ),
    Tool(
        name="AnalyzeDownstreamEffects",
        func=analyze_downstream_effects,
        description="Predicts functional effects on downstream genes based on pathway topology. Input: gene ID (e.g., 'hsa:6662'). Output: Analysis report",
    ),
    Tool(
        name="GetPathwayInfo",
        func=get_pathway_info,
        description="Get detailed information about a pathway. Input: pathway name (e.g., 'Alzheimer disease')",
    ),
    Tool(
        name="FindConnectingPathways",
        func=find_connecting_pathways,
        description="Find pathways that connect two or more genes. Input: comma-separated gene IDs (e.g., 'hsa:6662, hsa:7124')",
    ),
]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = """You are an advanced biomedical research assistant specialized in gene-pathway-disease relationships. Follow this analytical framework:

        1. **Gene-Disease Validation**
        - First confirm associations using pathway data (KEGG)
        - Distinguish between gene IDs (e.g., 'hsa:6662') and gene symbols (e.g., 'TP53')
        - Example: "Is gene X linked to disease Y in known pathways?"

        2. **Functional Profiling**
        - For gene symbols: Analyze GO terms from the knowledge graph
        - For gene IDs: Focus on pathway relationships and topology
        - Identify key biological processes/molecular functions

        3. **Downstream Analysis Protocol**:
        a) For gene IDs (hsa:XXXX): Identify downstream genes based on pathway order
        b) Analyze pathway context and complexity
        c) Consider relative positions within pathways
        d) Look for connecting pathways between multiple genes
        e) Examine pathway topology for potential regulatory relationships

        4. **Mechanistic Hypothesis Generation**
        - Combine pathway position + functional analysis
        - Propose testable biological mechanisms
        - Example: "Gene X likely influences disease Y through [pathway Z] by affecting [downstream genes]"

        Special Capabilities:
        - Leverage both GO annotations (for gene symbols) AND pathway topology (for gene IDs)
        - Distinguish between upstream regulators and downstream effectors in pathways
        - Identify shared pathways that connect multiple genes of interest

        Guidelines:
        - Be precise with identifiers - specify whether using gene IDs (hsa:XXXX) or gene symbols
        - For gene IDs, use pathway-based downstream analysis tools
        - For gene symbols, use GO term-based functional analysis
        - When analyzing gene relationships, consider:
        * Pathway context and gene order
        * Disease associations
        * Potential regulatory mechanisms
        - Flag uncertain predictions as "hypothetical"
        """

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
# query = "Which genes are related to type II diabetes mellitus?"
# query = "Which genes are related to type II diabetes?"
# query = "What type II diabetes genes are associated with the GO term GO:0005975 (carbohydrate metabolic process)?"
# query = "How might TP53 affect downstream genes in cancer?"
# query = "How might SOX9 influence other genes in chondrogenesis?"
query = "How might SIRT1 influence other genes in chondrogenesis?"

response = agent_executor.invoke({"input": query})
print(response["output"])
