# Gene-Pathway Disease Hypothesis Generation Agent

This project implements an LLM-powered agent designed to generate hypothesis by linking genes to diseases. The system uses curated data from KEGG pathways and Gene Ontology (GO) annotations, along with a modular toolset and a ReAct-based reasoning loop to dynamically respond to natural language queries.

## Overview

The agent integrates:

- **Structured biological data** from KEGG and GO
- **Modular tools** for pathway retrieval, GO term querying, graph exploration, downstream analysis, and hypothesis generation
- **LLM reasoning** powered by GPT-4o and uses multi-step ReAct logic from LangChain

The system enables users to:

- Query disease-gene relationships
- Explore shared mechanisms between genes
- Analyze downstream effects within biological pathways
- Generate efficient hypothesis about gene-disease links

---

## Setup Instructions

Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

Add your OpenAI API key
Create a .env file in the root directory with your key:
```bash
OPENAI_API_KEY=<your-openai-api-key>
```

Move the ```goa_human.gaf``` file into the ```data``` directory.

Create a conda environment and activate it
```bash
conda env create -f environment.yml
conda activate gene_disease_env
```
Alternatively create a new conda environment and install the packages from the requirements.txt file
```bash
conda create --name <env-name> python=3.12
conda activate <env-name>
pip install -r requirements.txt
```

Parse the KGML and GAF data
```bash
python parse_data.py
```

Visualize the GO-term data
```bash
python plots.py
```

Run the agent with an input query by simply running:
```bash
python gene_go_agent.py --query '<your-input-query>'
```
