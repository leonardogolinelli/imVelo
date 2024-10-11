import numpy as np
import pandas as pd
import scvelo as scv
import scanpy as sc
from scvelo_adapted_utils import load_files
from scvelo_adapted_metrics import compute_scvelo_metrics, deg_genes
from scvelo_adapted_plots import plot_important_genes
import os 
from benchmark_preprocessing import preprocess

n_highly_var_genes = 2000
smooth_k = 30
#datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
#cell_type_keys = ["Clusters", "clusters","celltype", "clusters"]

datasets = ["forebrain"]
cell_type_keys = ["Clusters"]

for dataset, cell_type_key in zip(datasets, cell_type_keys):
    adata = preprocess(dataset, cell_type_key, n_highly_var_genes, smooth_k)
    print("recovering dynamics")
    scv.tl.recover_dynamics(adata, n_jobs=4)
    scv.tl.velocity(adata, mode="dynamical")
    scv.tl.velocity_graph(adata)
    adata.X = adata.X.astype(float)
    # Filter out genes with NaN values
    adata = adata[:, ~np.isnan(adata.layers["velocity"]).any(axis=0)]
    scv.pl.velocity_embedding_stream(adata, color=cell_type_key)
    compute_scvelo_metrics(adata, dataset, False, cell_type_key)
    print("writing h5ad")
    adata.write_h5ad(f"{dataset}/scvelo_{dataset}.h5ad")
    plot_important_genes(adata, dataset, cell_type_key)
    #deg_genes(adata, dataset, cell_type_key=cell_type_key)
    print("analysis finished")

