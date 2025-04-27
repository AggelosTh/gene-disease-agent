import os

import matplotlib.pyplot as plt
import pandas as pd

from config import ROOT_DIR

df_goa = pd.read_csv(os.path.join(ROOT_DIR, "data/gene_go_links.csv"))
top_go_terms = df_goa["go_term"].value_counts().head(10)

top_go_terms.head(10).plot(kind="barh", figsize=(8, 6), color="skyblue")
plt.xlabel("Number of Genes")
plt.title("Top 10 Most Frequent GO Terms")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, "top_go_terms.png"), dpi=300)
plt.show()
