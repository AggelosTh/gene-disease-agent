"""Parse KGML and GAF files to extract gene-pathway and gene-GO term links."""

import os
import pickle
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

from config import ROOT_DIR


def get_gene_symbol(gene_id: str) -> str:
    """Fetch gene symbol from KEGG API given a gene ID.
    Args:
        gene_id (str): Gene ID to fetch the symbol for.
    Returns:
        str: Gene symbol if found, None otherwise.
    """
    url = f"http://rest.kegg.jp/get/{gene_id}"
    response = requests.get(url)
    if response.status_code == 200:
        lines = response.text.split("\n")
        for line in lines:
            if line.startswith("SYMBOL"):
                parts = line.split()
                if len(parts) > 1:
                    return parts[1].strip(",")
    return None


def fetch_gene_id_to_symbol_mapping(gene_ids: list, sleep_time: float = 0.5) -> dict:
    """Fetch gene symbols for a list of gene IDs.
    Args:
        gene_ids (list): List of gene IDs to fetch symbols for.
        sleep_time (float): Time to sleep between API calls to avoid rate limiting.
    Returns:
        dict: A dictionary mapping gene IDs to gene symbols."""
    gene_id_to_symbol = {}
    for gene_id in gene_ids:
        if gene_id not in gene_id_to_symbol:
            symbol = get_gene_symbol(gene_id)
            if symbol:
                gene_id_to_symbol[gene_id] = symbol
            time.sleep(sleep_time)  # Avoid hitting the API too hard
    return gene_id_to_symbol


def parse_kgml_file(file_path: str) -> list:
    """Parse a KGML file and extract gene-pathway links.
    Args:
        file_path (str): Path to the KGML file.
    Returns:
        list: A list of dictionaries containing gene-pathway links."""
    tree = ET.parse(file_path)
    root = tree.getroot()

    pathway_name = root.attrib.get("title", "Unknown Pathway")
    gene_pathway_links = []

    for entry in root.findall("entry"):
        if entry.attrib.get("type") == "gene":
            gene_ids = entry.attrib.get("name", "").split()
            for gene_id in gene_ids:
                gene_pathway_links.append(
                    {
                        "gene_id": gene_id,
                        "pathway": pathway_name,
                        "source": file_path.name,
                        "link": entry.attrib.get("link", ""),
                    }
                )

    return gene_pathway_links


def parse_kgml_directory(directory_path: str) -> list:
    """Parse all KGML files in a directory and extract gene-pathway links.
    Args:
        directory_path (str): Path to the directory containing KGML files.
    Returns:
        list: A list of dictionaries containing gene-pathway links."""
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory {directory_path} does not exist.")
    all_links = []
    directory = Path(directory_path)

    for kgml_file in directory.glob("*.xml"):
        gene_links = parse_kgml_file(kgml_file)
        all_links.extend(gene_links)

    return all_links


def save_to_csv(links: list, output_path: str):
    """Save the extracted links to a CSV file.
    Args:
        links (list): List of dictionaries containing gene-pathway links.
        output_path (str): Path to the output CSV file.
    """
    df = pd.DataFrame(links)
    df.to_csv(output_path, index=False)


def parse_gaf_file(file_path: str) -> pd.DataFrame:
    """Parse a GAF file and extract gene-GO term links.
    Args:
        file_path (str): Path to the GAF file.
    Returns:
        pd.DataFrame: A DataFrame containing gene-GO term links."""
    df = pd.read_csv(file_path, sep="\t", comment="!", header=None, low_memory=False)
    df = df[[2, 4]]  # Column 2: gene symbol, Column 5: GO term
    df.columns = ["gene_symbol", "go_term"]

    return df


if __name__ == "__main__":
    # Parse KGML files
    kgml_links = parse_kgml_directory(os.path.join(ROOT_DIR, "data/KGML"))

    # Collect all unique gene IDs
    unique_gene_ids = list({link["gene_id"] for link in kgml_links})

    mapping_file = os.path.join(ROOT_DIR, "data/gene_id_to_symbol.pkl")
    if os.path.exists(mapping_file):
        print("Loading cached gene_id to symbol mapping...")
        with open(mapping_file, "rb") as f:
            gene_id_to_symbol = pickle.load(f)
    else:
        print("Fetching gene_id to symbol mapping...")
        gene_id_to_symbol = fetch_gene_id_to_symbol_mapping(unique_gene_ids)
        with open(mapping_file, "wb") as f:
            pickle.dump(gene_id_to_symbol, f)

    for link in kgml_links:
        link["gene_symbol"] = gene_id_to_symbol.get(link["gene_id"], None)

    # Save updated links
    df_kgml = pd.DataFrame(kgml_links)
    df_kgml.to_csv(os.path.join(ROOT_DIR, "data/gene_pathway_links.csv"), index=False)
    print("KGML files parsed and saved.")

    # Parse GAF file
    df_goa = parse_gaf_file(os.path.join(ROOT_DIR, "data/goa_human.gaf"))
    df_goa.to_csv(os.path.join(ROOT_DIR, "data/gene_go_links.csv"), index=False)
    print("GAF file parsed and saved.")
