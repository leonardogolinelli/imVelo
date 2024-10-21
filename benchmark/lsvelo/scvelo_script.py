import numpy as np
import pandas as pd
import scvelo as scv
import scanpy as sc
from scvelo_adapted_metrics import compute_scvelo_metrics, deg_genes
from scvelo_adapted_plots import plot_important_genes
import os
from compute_velocity import learn_and_store_velocity

datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters","celltype", "clusters"]

datasets=["dentategyrus_lamanno_P5"]
cell_type_keys = ["clusters"]
for dataset, cell_type_key in zip(datasets, cell_type_keys):
    print(f"processing dataset: {dataset}")

    """if os.path.isdir(f"{dataset}"):
        print("skipping")
        continue"""
    
    print(f"processing dataset: {dataset}")
    adata_path = f"../imVelo/{dataset}/imVelo_{dataset}.h5ad"
    adata = sc.read_h5ad(adata_path)
    if dataset == "forebrain":
        adata.obs["pseudotime"] = 1/(1e-1 + np.load(f"pseudotimes/{dataset}_pseudotime.npy"))
    elif dataset == "dentategyrus_lamanno_P5":
        adata.obs["pseudotime"] =  adata.obs["isomap_1_velocity"].copy()
    else:
        adata.obs["pseudotime"] =  adata.obs["isomap_1"].copy()

    del adata.layers["velocity"]
    del adata.layers["velocity_u"]
    del adata.obsm["z"]

    learn_and_store_velocity(adata, cell_type_key)

    if not dataset == "pancreas":
        adata.layers["velocity"] *=-1
        adata.layers["velocity_u"] *=-1

    if dataset == "dentategyrus_lamanno_P5":
        idx_immuno = adata.obs["clusters"] == "ImmAstro"
        idx_opc = adata.obs["clusters"] == "OPC"
        adata.layers["velocity"][idx_immuno, :] *= -1
        adata.layers["velocity"][idx_opc, :] *= -1

    # Save the adata object with velocities
    os.makedirs(dataset, exist_ok=True)
    adata.write_h5ad(f"{dataset}/lsvelo_{dataset}.h5ad")

    #adata = sc.read_h5ad(f"{dataset}/lsvelo_{dataset}.h5ad")
    # this will train the model and predict the velocity vectore. The result is stored in adata.layers['velocity']. You can use trainer.model to access the model.
    compute_scvelo_metrics(adata, dataset, False, cell_type_key)
    adata.write_h5ad(f"{dataset}/lsvelo_{dataset}.h5ad")
    plot_important_genes(adata, dataset, cell_type_key)
    deg_genes(adata, dataset, cell_type_key=cell_type_key)
    print("analysis finished")
    os.rename(dataset, f"{dataset}")

#else:
    #print(f"Dataset {dataset} already processed: skipping..")
    print("-------------------")

