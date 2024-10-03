import numpy as np
import pandas as pd
import scvelo as scv
import scanpy as sc
from scvelo_adapted_utils import load_files
from scvelo_adapted_metrics import compute_scvelo_metrics, deg_genes
from scvelo_adapted_plots import plot_important_genes
from benchmark_preprocessing import preprocess
import os 


n_highly_var_genes = 2000
smooth_k = 30
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters","celltype", "clusters"]


datasets = ["forebrain"]
cell_type_keys = ["Clusters"]

for dataset, cell_type_key in zip(datasets, cell_type_keys):
    adata = preprocess(dataset, cell_type_key, n_highly_var_genes, smooth_k)
    scv.tl.velocity(adata, mode="stochastic")
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding_stream(adata, color=cell_type_key)
    compute_scvelo_metrics(adata, dataset, False, cell_type_key)
    #deg_genes(adata, dataset, cell_type_key=cell_type_key, n_deg_rows=5)
    adata.write_h5ad(f"{dataset}/stochastic_{dataset}.h5ad")
