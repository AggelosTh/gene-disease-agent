import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from config import ROOT_DIR


def parse_kgml_file(file_path: str) -> list:
    """ "Parse a KGML file and extract gene-pathway links.
    Args:
        file_path (str): Path to the KGML file.
    Returns:
        list: A list of dictionaries containing gene IDs and their corresponding pathways.
    """

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
                    }
                )

    return gene_pathway_links


def parse_kgml_directory(directory_path: str) -> list:
    """Parse all KGML files in a directory and extract gene-pathway links.
    Args:
        directory_path (str): Path to the directory containing KGML files.
    Returns:
        list: A list of dictionaries containing gene IDs and their corresponding pathways.
    """
    all_links = []
    directory = Path(directory_path)

    for kgml_file in directory.glob("*.xml"):
        gene_links = parse_kgml_file(kgml_file)
        all_links.extend(gene_links)

    return all_links


def save_to_csv(links: str, output_path: str):
    """Save the extracted links to a CSV file.
    Args:
        links (list): A list of dictionaries containing gene IDs and their corresponding pathways.
        output_path (str): Path to the output CSV file.
    """
    df = pd.DataFrame(links)
    df.to_csv(output_path, index=False)


def parse_gaf_file(file_path: str) -> pd.DataFrame:
    """Parse a GAF file and extract gene-GO term links.
    Args:
        file_path (str): Path to the GAF file.
    Returns:
        pd.DataFrame: A DataFrame containing gene symbols and their corresponding GO terms.
    """
    # Read the GAF file, skipping comments and using tab as the separator
    df = pd.read_csv(file_path, sep="\t", comment="!", header=None, low_memory=False)
    df = df[[2, 4]]  # Column 2: gene symbol, Column 5: GO term
    df.columns = ["gene_symbol", "go_term"]

    return df


kgml_links = parse_kgml_directory(os.path.join(ROOT_DIR, "data/KGML"))

df_kgml = pd.DataFrame(kgml_links)
df_kgml.to_csv(os.path.join(ROOT_DIR, "data/gene_pathway_links.csv"))

df_goa = parse_gaf_file(os.path.join(ROOT_DIR, "data/goa_human.gaf"))
df_goa.to_csv(os.path.join(ROOT_DIR, "data/gene_go_links.csv"), index=False)
