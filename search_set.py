import argparse
import os
from datetime import datetime
import pandas as pd

def generate_search_set(gene):
    # Transform DGIdb Dataset to match input gene
    data_path = os.path.join("data", "dgidb", "data-1727966158790.csv")
    df = pd.read_csv(data_path)

    tdf = df[df["interaction_score"].notnull()]
    tdf = tdf[tdf["gene_symbol"] == gene].sort_values(by="interaction_score", ascending=False).reset_index(drop=True)
    tdf.rename(columns={"gene_symbol": "Gene", "concept_name": "Drug"}, inplace=True)

    # Save Search set
    os.makedirs("search", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    out_filename = f"{timestamp}_{gene}_clin_score.csv"
    out_path = os.path.join("search", out_filename)
    tdf.to_csv(out_path, index=False)
    print(f'Search set saved to {out_path}')
    