import networkx as nx
from langchain.agents import (
    AgentExecutor,
    AgentType,
    Tool,
    create_tool_calling_agent,
    initialize_agent,
)
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# Load your graph
G = nx.read_graphml("gene_go_network.graphml")


# Define tools that operate on your graph
@tool
def get_gene_go_terms(gene_symbol):
    """Find all GO terms associated with a gene"""
    if gene_symbol not in G.nodes():
        return f"Gene {gene_symbol} not found in the knowledge base."

    go_terms = []
    for neighbor in G.neighbors(gene_symbol):
        if G.nodes[neighbor].get("type") == "go_term":
            go_terms.append(neighbor)

    return f"Gene {gene_symbol} is associated with GO terms: {', '.join(go_terms)}"


@tool
def get_genes_by_go_term(go_term):
    """Find all genes with a specific GO term"""
    if go_term not in G.nodes():
        return f"GO term {go_term} not found in the knowledge base."

    genes = []
    for neighbor in G.neighbors(go_term):
        if G.nodes[neighbor].get("type") == "gene":
            genes.append(neighbor)

    return f"GO term {go_term} is associated with genes: {', '.join(genes)}"


@tool
def get_shared_go_terms(gene1, gene2):
    """Find GO terms shared between two genes"""
    if gene1 not in G.nodes():
        return f"Gene {gene1} not found in the knowledge base."
    if gene2 not in G.nodes():
        return f"Gene {gene2} not found in the knowledge base."

    gene1_terms = set(
        n for n in G.neighbors(gene1) if G.nodes[n].get("type") == "go_term"
    )
    gene2_terms = set(
        n for n in G.neighbors(gene2) if G.nodes[n].get("type") == "go_term"
    )
    shared = gene1_terms.intersection(gene2_terms)

    return f"Genes {gene1} and {gene2} share GO terms: {', '.join(shared)}"


# Define tools list
tools = [
    Tool(
        name="GetGeneGOTerms",
        func=get_gene_go_terms,
        description="Finds all GO terms associated with a specific gene. Input should be a single gene symbol (e.g., 'SOX9'). Returns a list of GO terms annotating that gene.",
    ),
    Tool(
        name="GetGenesWithGOTerm",
        func=get_genes_by_go_term,
        description="Finds all genes annotated with a specific GO term. Input should be a GO term ID (e.g., 'GO:0060350'). Returns a list of genes associated with that GO term.",
    ),
    Tool(
        name="GetSharedGOTerms",
        func=get_shared_go_terms,
        description="Finds GO terms shared between two genes. Input should be two gene symbols separated by a comma (e.g., 'SOX9,ACAN'). Returns a list of GO terms common to both genes.",
    ),
]

# Set up the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Create the correct prompt for ReAct agent
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
            You are a bioinformatics research assistant with expertise in gene function, pathways, and diseases.
            You have access to a knowledge base of gene-GO term relationships.
            Based on this data and your knowledge, generate hypotheses about gene functions and disease associations.

            Remember that Gene Ontology (GO) terms represent biological processes, molecular functions, and cellular components.
            GO:0060350 refers to "endochondral bone morphogenesis" for example.

            You have access to the following tools:

            You can use the following tools: {tool_names}

            Detailed tool descriptions:
            {tools}
            
            Use these tools to explore the gene-GO relationships and formulate your hypotheses.
            """
        ),
        HumanMessage(content="{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

prompt = """
        You are a bioinformatics research assistant with expertise in gene function, pathways, and diseases.
        You have access to a knowledge base of gene-GO term relationships.
        Based on this data and your knowledge, generate hypotheses about gene functions and disease associations.

        Remember that Gene Ontology (GO) terms represent biological processes, molecular functions, and cellular components.
        GO:0060350 refers to "endochondral bone morphogenesis" for example.
        
        Use the tools to explore the gene-GO relationships and formulate your hypotheses.
        """

# Create the agent
# agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
# agent_executor = AgentExecutor.from_agent_and_tools(
#     agent=agent, tools=tools, verbose=True
# )
# agent_executor = AgentExecutor(agent=agent, tools=tools)


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

# Example usage
response = agent_executor.invoke(
    {
        "input": "What diseases might SOX9 be associated with based on its GO annotations?"
    }
)
print(response["output"])
