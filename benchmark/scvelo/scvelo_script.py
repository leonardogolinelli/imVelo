import numpy as np
import pandas as pd
import scvelo as scv
import scanpy as sc
from scvelo_adapted_utils import load_files
from scvelo_adapted_metrics import compute_scvelo_metrics, deg_genes
from scvelo_adapted_plots import plot_important_genes
import os 

#datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "forebrain", "dentategyrus_lamanno_P5"]
#cell_type_keys = ["Clusters", "clusters", "celltype", "Clusters", "clusters"]

datasets = ["dentategyrus_lamanno_P5"]
cell_type_keys = ["clusters"]

for dataset, cell_type_key in zip(datasets, cell_type_keys):
    if dataset == "forebrain":
        adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_forebrain_K11_knn_rep_ve_best_key_None_0_kl_weight_1e-9_1e-08_20k/forebrain/K11/adata/adata_K11_dt_ve.h5ad"
    elif dataset == "pancreas":
        adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_pancreas_K11_knn_rep_ve_best_key_None_0_kl_weight_1e-9_1e-8/pancreas/K11/adata/adata_K11_dt_ve.h5ad"
    elif dataset == "gastrulation_erythroid":
        adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_gastrulation_erythroid_K11_knn_rep_ve_best_key_None_0_kl_weight_1e-9_1e-08_20k_256/gastrulation_erythroid/K11/adata/adata_K11_dt_ve.h5ad"
    elif dataset == "dentategyrus_lamanno_P5":
        adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_dentategyrus_lamanno_P5_K31_knn_rep_pca_best_key_pca_unique_0_kl_weight_1e-9_1e-08_20k_12_july/dentategyrus_lamanno_P5/K31/adata/adata_K31_dt_pca.h5ad"

    adata = sc.read_h5ad(adata_path)
    del adata.layers["velocity"]
    del adata.layers["velocity_u"]
    sc.pp.neighbors(adata)
    scv.tl.recover_dynamics(adata, n_jobs=4)
    scv.tl.velocity(adata, mode="dynamical")
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding_stream(adata, color=cell_type_key)
    compute_scvelo_metrics(adata, dataset, False, cell_type_key)
    adata.write_h5ad(f"{dataset}/scvelo_{dataset}.h5ad")
    plot_important_genes(adata, dataset, cell_type_key)
    #deg_genes(adata, dataset, cell_type_key=cell_type_key)

