"""Development of a LangChain agent for gene-GO analysis using KEGG pathways."""

import pickle
from argparse import ArgumentParser

import networkx as nx
import pandas as pd
from fuzzywuzzy import process
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

parser = ArgumentParser(description="Pathway and disease hypothesis generation agent")

parser.add_argument(
    "--temperature",
    type=float,
    default=0.0,
    help="Temperature for the language model",
)
parser.add_argument(
    "--verbose",
    type=bool,
    default=True,
    help="Enable verbose output",
)
parser.add_argument(
    "--max_iterations",
    type=int,
    default=5,
    help="Maximum number of iterations for the agent (default: 5)",
)
parser.add_argument(
    "--graphml_file",
    type=str,
    default="data/gene_go_network.graphml",
    help="Path to the GraphML file containing the gene-GO network",
)
parser.add_argument(
    "--gene_pathway_file",
    type=str,
    default="data/gene_pathway_links.csv",
    help="Path to the CSV file containing gene-pathway links",
)
parser.add_argument(
    "--gene_id_to_symbol_file",
    type=str,
    default="data/gene_id_to_symbol.pkl",
    help="Path to the pickle file containing gene ID to symbol mapping",
)
parser.add_argument(
    "--query",
    type=str,
    help="Query to run the agent",
)
args = parser.parse_args()

# Load graph
print("Loading graph...")
G = nx.read_graphml(args.graphml_file)

# Load gene-pathway links
print("Loading gene-pathway links...")
gene_pathway_df = pd.read_csv(args.gene_pathway_file)

# Load gene ID to symbol mapping
print("Loading gene ID to symbol mapping...")
with open(args.gene_id_to_symbol_file, "rb") as f:
    gene_id_to_symbol = pickle.load(f)

# Create reverse mapping
symbol_to_gene_id = {v: k for k, v in gene_id_to_symbol.items()}


def convert_id_to_symbol(gene_id: str) -> str:
    """Convert a gene ID to its symbol
    Args:
        gene_id (str): The gene ID (e.g., 'hsa:6662')
    Returns:
        str: The gene symbol (e.g., 'SOX9')"""
    return gene_id_to_symbol.get(gene_id, gene_id)


def convert_symbol_to_id(gene_symbol: str) -> str:
    """Convert a gene symbol to its ID
    Args:
        gene_symbol (str): The gene symbol (e.g., 'SOX9')
    Returns:
        str: The gene ID (e.g., 'hsa:6662')"""
    return symbol_to_gene_id.get(gene_symbol, gene_symbol)


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
    gene_ids = matches["gene_id"].unique().tolist()

    # Convert gene IDs to symbols
    gene_symbols = [convert_id_to_symbol(gene_id) for gene_id in gene_ids]
    gene_symbols = [symbol for symbol in gene_symbols if symbol is not None]

    if not gene_symbols:
        return f"No genes found for the disease: {disease_name}"

    return f"Genes involved in {best_match}: {', '.join(gene_symbols[:10])}{'... and ' + str(len(gene_symbols) - 10) + ' more genes' if len(gene_symbols) > 10 else ''}"


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
def get_diseases_by_gene(gene_input: str) -> str:
    """Find all diseases a gene is involved in via KEGG pathways

    Args:
        gene_input (str): The gene ID or symbol

    Returns:
        str: A string listing the diseases associated with the gene
    """
    # Check if input is a gene symbol and convert to ID if needed
    gene_id = gene_input
    if not gene_input.startswith("hsa:"):
        gene_id = convert_symbol_to_id(gene_input)
        if gene_id == gene_input and not gene_input.startswith("hsa:"):
            return f"Could not find gene ID for symbol {gene_input}."

    matches = gene_pathway_df[gene_pathway_df["gene_id"] == gene_id]
    disease_list = matches["pathway"].unique().tolist()

    if not disease_list:
        return f"No diseases found for gene {gene_input} (ID: {gene_id})."

    return f"Gene {gene_input} (ID: {gene_id}) is involved in the following diseases: {', '.join(disease_list)}"


@tool
def generate_hypothesis(input_str: str) -> str:
    """Generate a hypothesis about a gene's potential role in a disease

    Args:
        input_str (str): Input string containing gene ID or symbol and disease name in the format "gene, disease_name"

    Returns:
        str: A hypothesis about the gene's potential role in the disease
    """
    try:
        gene_input, disease_name = [x.strip() for x in input_str.split(",", 1)]
    except ValueError:
        return "Please provide input in format: 'gene, disease_name'"

    gene_id = gene_input
    gene_symbol = gene_input

    if not gene_input.startswith("hsa:"):
        gene_id = convert_symbol_to_id(gene_input)
        if gene_id == gene_input and not gene_input.startswith("hsa:"):
            return f"Could not find gene ID for symbol {gene_input}."
    else:
        gene_symbol = convert_id_to_symbol(gene_input)

    # Check gene-disease association
    diseases = get_diseases_by_gene(gene_id)
    if disease_name.lower() not in diseases.lower():
        return f"Gene {gene_symbol} (ID: {gene_id}) is not directly associated with {disease_name} in pathway data."

    go_terms = get_gene_go_terms(gene_symbol)
    if "not found" in go_terms:
        return go_terms

    # Generate hypothesis
    return (
        f"Hypothesis for {gene_symbol} (ID: {gene_id}) in {disease_name}:\n"
        f"1. Pathway data confirms association\n"
        f"2. Gene's GO terms suggest potential mechanisms: {go_terms}\n"
        f"3. Potential roles: Based on its annotations, {gene_symbol} might contribute to {disease_name} "
        "through these biological processes or molecular functions."
    )


@tool
def find_shared_mechanisms(genes_input: str) -> dict:
    """Find shared GO terms between two genes that might indicate common mechanisms

    Args:
        genes_input (str): Input string containing two gene symbols or IDs separated by a comma
        (e.g., "SOX9, TP53" or "hsa:6662, hsa:7157")

    Returns:
        dict: A dictionary containing the two genes, their shared GO terms, and the count of shared terms
    """
    genes = [g.strip() for g in genes_input.split(",")]
    if len(genes) != 2:
        return {
            "error": "Please provide exactly two gene symbols/IDs, separated by a comma."
        }

    gene_symbols = []
    for gene in genes:
        if gene.startswith("hsa:"):
            symbol = convert_id_to_symbol(gene)
            if symbol is None:
                return {"error": f"Gene ID {gene} not found in mapping."}
            gene_symbols.append(symbol)
        else:
            gene_symbols.append(gene)

    gene1, gene2 = gene_symbols[0], gene_symbols[1]

    if gene1 not in G.nodes():
        return {"error": f"Gene {gene1} not found in graph"}
    if gene2 not in G.nodes():
        return {"error": f"Gene {gene2} not found in graph"}

    # Get GO terms for both genes
    go1 = set(n for n in G.neighbors(gene1) if G.nodes[n].get("type") == "go_term")
    go2 = set(n for n in G.neighbors(gene2) if G.nodes[n].get("type") == "go_term")

    shared = list(go1 & go2)

    return {"genes": [gene1, gene2], "shared_go_terms": shared, "count": len(shared)}


@tool
def find_downstream_genes(gene_input: str) -> dict:
    """Find genes likely to be downstream in pathways based on the input gene.
    Args:
        gene_input (str): Gene ID or symbol (e.g., 'hsa:6662' or 'SOX9')
    Returns:
        dict: A dictionary containing the target gene, downstream genes, and pathways
    """

    # Convert gene symbol to ID if needed
    gene_id = gene_input
    if not gene_input.startswith("hsa:"):
        gene_id = convert_symbol_to_id(gene_input)
        if gene_id == gene_input and not gene_input.startswith("hsa:"):
            return {"error": f"Could not find gene ID for symbol {gene_input}."}

    # Find all pathways containing the gene
    pathways = gene_pathway_df[gene_pathway_df["gene_id"] == gene_id][
        "pathway"
    ].unique()

    # Get all other genes in these pathways
    downstream_ids = []
    for pathway in pathways:
        pathway_genes = gene_pathway_df[gene_pathway_df["pathway"] == pathway][
            "gene_id"
        ].tolist()
        # Simple heuristic: genes appearing after our target in pathway data
        if gene_id in pathway_genes:
            idx = pathway_genes.index(gene_id)
            downstream_ids.extend(pathway_genes[idx + 1 :])  # Genes after the target

    # Convert IDs to symbols
    downstream_symbols = [convert_id_to_symbol(g_id) for g_id in downstream_ids]
    downstream_symbols = [s for s in downstream_symbols if s is not None]

    return {
        "target": gene_input,
        "downstream": list(set(downstream_symbols)),
        "pathways": list(pathways),
    }


@tool
def analyze_downstream_effects(gene_input: str) -> str:
    """Predict functional effects on downstream genes based on pathway topology.
    Args:
        gene_input (str): Gene ID or symbol (e.g., 'hsa:6662' or 'SOX9')
    Returns:
        str: Analysis report summarizing the downstream effects
    """

    # Convert gene symbol to ID if needed
    gene_id = gene_input
    gene_symbol = gene_input

    if not gene_input.startswith("hsa:"):
        gene_id = convert_symbol_to_id(gene_input)
        if gene_id == gene_input and not gene_input.startswith("hsa:"):
            return f"Could not find gene ID for symbol {gene_input}."
    else:
        gene_symbol = convert_id_to_symbol(gene_id)

    # Find downstream genes
    downstream_data = find_downstream_genes(gene_id)
    if "error" in downstream_data:
        return downstream_data["error"]

    if not downstream_data["downstream"]:
        return f"No downstream genes found for {gene_symbol} (ID: {gene_id}) in pathway data."

    # Analyze pathway context
    pathway_counts = {}
    for pathway in downstream_data["pathways"]:
        pathway_counts[pathway] = len(
            gene_pathway_df[gene_pathway_df["pathway"] == pathway]
        )

    # Generate report
    report = [
        f"Downstream analysis for {gene_symbol} (ID: {gene_id}):",
        f"- Found in {len(downstream_data['pathways'])} pathways",
        f"- Potentially affects {len(downstream_data['downstream'])} downstream genes",
        "\nPathway context:",
    ]

    # Sort pathways by gene count
    sorted_pathways = sorted(pathway_counts.items(), key=lambda x: x[1], reverse=True)
    for pathway, count in sorted_pathways:
        report.append(f"- {pathway}: {count} total genes")

    # Show top 5 downstream genes
    report.append("\nTop downstream genes:")
    for gene in downstream_data["downstream"][:5]:
        report.append(f"- {gene}")

    report.append("\nNote: This analysis is based on pathway topology from KEGG data.")

    return "\n".join(report)


@tool
def get_pathway_info(pathway_name: str) -> str:
    """Get detailed information about a specific pathway.
    Args:
        pathway_name (str): The name of the pathway (e.g., 'Alzheimer disease')
    Returns:
        str: A report summarizing the pathway and its associated genes
    """

    if pathway_name not in gene_pathway_df["pathway"].values:
        # Try fuzzy matching
        all_pathways = gene_pathway_df["pathway"].unique()
        best_match, score = process.extractOne(pathway_name, all_pathways)

        if score < 80:
            return f"Pathway '{pathway_name}' not found. Please check the spelling."

        pathway_name = best_match

    # Get genes in pathway
    pathway_gene_ids = gene_pathway_df[gene_pathway_df["pathway"] == pathway_name][
        "gene_id"
    ].tolist()

    # Convert gene IDs to symbols
    pathway_gene_symbols = [
        convert_id_to_symbol(gene_id) for gene_id in pathway_gene_ids
    ]
    pathway_gene_symbols = [s for s in pathway_gene_symbols if s is not None]

    report = [
        f"Pathway: {pathway_name}",
        f"Total genes: {len(set(pathway_gene_ids))}",
        "\nGenes (in pathway order):",
    ]

    # Show gene IDs and symbols
    for i, (gene_id, gene_symbol) in enumerate(
        zip(pathway_gene_ids[:10], pathway_gene_symbols[:10])
    ):
        report.append(f"- {gene_symbol} ({gene_id})")

    if len(pathway_gene_ids) > 10:
        report.append(f"... and {len(pathway_gene_ids) - 10} more genes")

    return "\n".join(report)


@tool
def find_connecting_pathways(gene_inputs: str) -> str:
    """Find pathways that connect two or more genes.
    Args:
        gene_inputs (str): Comma-separated list of gene IDs or symbols (e.g., 'hsa:6662, hsa:7124' or 'SOX9, TP53')
    Returns:
        str: A report summarizing the connecting pathways and their associated genes
    """

    gene_list = [g.strip() for g in gene_inputs.split(",")]
    gene_id_list = []

    # Convert symbols to IDs if needed
    for gene in gene_list:
        if gene.startswith("hsa:"):
            gene_id_list.append(gene)
        else:
            gene_id = convert_symbol_to_id(gene)
            if gene_id == gene:
                return f"Gene symbol '{gene}' not found in mapping."
            gene_id_list.append(gene_id)

    # Validate gene IDs
    for gene_id in gene_id_list:
        if gene_id not in gene_pathway_df["gene_id"].values:
            return f"Gene ID '{gene_id}' not found in pathway data."

    # Find pathways for each gene
    gene_pathways = {}
    for gene_id in gene_id_list:
        gene_pathways[gene_id] = set(
            gene_pathway_df[gene_pathway_df["gene_id"] == gene_id]["pathway"]
        )

    # Find shared pathways
    shared_pathways = set.intersection(*gene_pathways.values())

    if not shared_pathways:
        original_genes_str = ", ".join(gene_list)
        return f"No shared pathways found connecting the genes: {original_genes_str}"

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
        for i, gene_id in enumerate(gene_id_list):
            if gene_id in pathway_genes:
                pos = pathway_genes.index(gene_id) + 1  # 1-based indexing
                gene_symbol = convert_id_to_symbol(gene_id)
                positions.append(f"{gene_symbol} ({gene_id}, pos {pos})")

        report.append(f"- {pathway}: {', '.join(positions)}")

    return "\n".join(report)


@tool
def analyze_multi_gene_impact(genes_input: str) -> str:
    """Analyze the combined biological processes and disease associations of multiple genes.

    Args:
        genes_input (str): Comma-separated list of gene IDs or symbols.

    Returns:
        str: A report summarizing shared biological processes and potential combined disease relevance.
    """
    input_genes = [g.strip() for g in genes_input.split(",")]

    # Process gene inputs - convert to symbols for GO terms, IDs for pathway analysis
    gene_symbols = []
    gene_ids = []

    for gene in input_genes:
        if gene.startswith("hsa:"):
            # It's an ID, convert to symbol for graph analysis
            symbol = convert_id_to_symbol(gene)
            if symbol is None:
                return f"Gene ID {gene} not found in mapping."
            gene_symbols.append(symbol)
            gene_ids.append(gene)
        else:
            # It's a symbol, convert to ID for pathway analysis
            gene_symbols.append(gene)
            gene_id = convert_symbol_to_id(gene)
            if gene_id == gene and not gene.startswith("hsa:"):
                return f"Gene symbol {gene} not found in mapping."
            gene_ids.append(gene_id)

    missing_genes = [g for g in gene_symbols if g not in G.nodes]
    if missing_genes:
        return f"These genes were not found in the graph: {', '.join(missing_genes)}"

    # Collect GO terms for all genes
    all_go_terms = []
    for gene in gene_symbols:
        go_terms = [n for n in G.neighbors(gene) if G.nodes[n].get("type") == "go_term"]
        all_go_terms.append(set(go_terms))

    shared_go_terms = set.intersection(*all_go_terms) if all_go_terms else set()

    # Collect disease associations
    disease_sets = []
    for gene_id in gene_ids:
        diseases = gene_pathway_df[gene_pathway_df["gene_id"] == gene_id][
            "pathway"
        ].tolist()
        disease_sets.append(set(diseases))

    shared_diseases = set.intersection(*disease_sets) if disease_sets else set()

    # Generate summary
    report = [
        f"Multi-Gene Analysis for: {', '.join(gene_symbols)} ({', '.join(gene_ids)})\n"
    ]

    if shared_go_terms:
        report.append(
            f"- Shared GO terms ({len(shared_go_terms)}): {', '.join(list(shared_go_terms)[:5])}"
            + (
                f", ... and {len(shared_go_terms) - 5} more"
                if len(shared_go_terms) > 5
                else ""
            )
        )
    else:
        report.append("- No shared GO terms found.")

    if shared_diseases:
        report.append(
            f"- Shared disease pathways ({len(shared_diseases)}): {', '.join(list(shared_diseases)[:3])}"
            + (
                f", ... and {len(shared_diseases) - 3} more"
                if len(shared_diseases) > 3
                else ""
            )
        )
    else:
        report.append("- No shared disease pathways found.")

    report.append(
        "\nInterpretation: The shared GO terms and diseases suggest potential combined roles in overlapping biological processes or disease mechanisms."
    )

    return "\n".join(report)


tools = [
    Tool(
        name="GetDiseasesByGene",
        func=get_diseases_by_gene,
        description="Finds diseases associated with a gene. Input: gene ID (e.g., 'hsa:1234') or gene symbol (e.g., 'APOE')",
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
        description="Find shared GO terms between two genes. Input: two gene symbols or IDs (e.g., 'APOE, TP53' or 'hsa:348, hsa:7157'). Returns shared GO terms and their count.",
    ),
    Tool(
        name="GenerateHypothesis",
        func=generate_hypothesis,
        description="Generates a hypothesis about gene-disease mechanisms. Input: gene ID or symbol and disease name (e.g., 'APOE, Alzheimer disease')",
    ),
    Tool(
        name="FindDownstreamGenes",
        func=find_downstream_genes,
        description="Identifies genes likely to be downstream in pathways. Input: gene ID or symbol (e.g., 'hsa:6662' or 'SOX9'). Returns target gene and downstream genes.",
    ),
    Tool(
        name="AnalyzeDownstreamEffects",
        func=analyze_downstream_effects,
        description="Predicts functional effects on downstream genes based on pathway topology. Input: gene ID or symbol (e.g., 'hsa:6662' or 'SOX9').",
    ),
    Tool(
        name="GetPathwayInfo",
        func=get_pathway_info,
        description="Get detailed information about a pathway. Input: pathway name (e.g., 'Alzheimer disease')",
    ),
    Tool(
        name="FindConnectingPathways",
        func=find_connecting_pathways,
        description="Find pathways that connect two or more genes. Input: comma-separated gene IDs or symbols (e.g., 'hsa:6662, hsa:7124' or 'SOX9, TP53')",
    ),
    Tool(
        name="AnalyzeMultiGeneImpact",
        func=analyze_multi_gene_impact,
        description="Analyzes the combined biological processes and disease associations of multiple genes. Input: comma-separated gene symbols or IDs.",
    ),
]

prompt = """You are a biomedical research assistant focused on concise functional analysis of gene-disease relationships. Follow this framework:

        1. Gene-Disease Validation
        - Confirm gene association with diseases via pathway data (KEGG).
        - Clearly distinguish between gene IDs (e.g., 'hsa:6662') and gene symbols (e.g., 'TP53').

        2. Functional Profiling
        - For gene symbols: Summarize main biological processes (GO terms) in 1-2 sentences.
        - For gene IDs: Summarize pathway involvement, focusing on key roles.

        3. Downstream Insights
        - Briefly mention if the gene has known upstream or downstream relationships in pathways.
        - If uncertain, state "No direct evidence found."

        4. Hypothesis Generation
        - Suggest a potential mechanism in 1 sentence, combining pathway position + function.

        Answering style:
        - Be brief. No need for long explanations.
        - Focus on the main biological idea.
        - Flag uncertainties clearly but without long elaborations.

        Special Skills:
        - Use both GO terms and pathway topology depending on the identifier type.
        - Prioritize clarity and brevity over exhaustive detail.
        """

if __name__ == "__main__":

    llm = ChatOpenAI(model="gpt-4o", temperature=args.temperature)

    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=args.verbose,
        handle_parsing_errors=True,
        max_iterations=args.max_iterations,
        early_stopping_method="generate",
        agent_kwargs={"prefix": prompt},
    )
    if not args.query:
        print("Please provide a query using the --query argument.")
        exit(1)

    print("Running agent...")
    response = agent_executor.invoke({"input": args.query})
    print(response["output"])
