import numpy as np
import pandas as pd
import scvelo as scv
import scanpy as sc
from scvelo_adapted_utils import load_files
from scvelo_adapted_metrics import compute_scvelo_metrics, deg_wilcoxon
from scvelo_adapted_plots import plot_important_genes
import os 


datasets = ["forebrain", "dentategyrus_lamanno"]
cell_type_keys = ["Clusters", "clusters"]

for dataset, cell_type_key in zip(datasets, cell_type_keys):
    adata_path = os.path.expanduser(f"/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_pancreas_K11_knn_rep_ve_best_key_None_0_kl_weight_1e7_1e5/pancreas/K11/adata/adata_K11_dt_ve.h5ad")
    adata = sc.read_h5ad(adata_path)
    if dataset == "forebrain":
        adata.obs[cell_type_key] = [str(name) for name in adata.obs[cell_type_key]]
        adata.obs[cell_type_key] = pd.Series(adata.obs[cell_type_key], dtype="category")
    del adata.layers["velocity"]
    del adata.layers["velocity_u"]
    sc.pp.neighbors(adata)
    scv.tl.velocity(adata, mode="stochastic")
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding_stream(adata, color=cell_type_key)
    compute_scvelo_metrics(adata, dataset, False, cell_type_key)
    deg_wilcoxon(adata, dataset, cell_type_key=cell_type_key)
    adata.write_h5ad(os.path.expanduser(f"~/top_adatas/stochastic_{dataset}.h5ad"))
    adata.write_h5ad(os.path.expanduser(f"{dataset}/stochastic_{dataset}.h5ad"))


