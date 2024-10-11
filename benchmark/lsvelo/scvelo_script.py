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
time_keys = ["pca_1", "isomap_1"]

datasets = ["gastrulation_erythroid"]
cell_type_keys = ["celltype"]

for time_key in time_keys:
    print(f"processing time key: {time_key}")
    for dataset, cell_type_key in zip(datasets, cell_type_keys):
        """if os.path.isdir(f"{time_key}/{dataset}"):
            print("skipping")
            continue"""

        print(f"processing dataset: {dataset}")
        adata_path = f"../imVelo/{dataset}/imVelo_{dataset}.h5ad"
        adata = sc.read_h5ad(adata_path)
        del adata.layers["velocity"]
        del adata.layers["velocity_u"]
        del adata.obsm["z"]

        learn_and_store_velocity(adata, time_key, cell_type_key)

        adata.layers["velocity"] *=-1
        adata.layers["velocity_u"] *=-1

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
        os.makedirs(time_key, exist_ok=True)
        os.rename(dataset, f"{time_key}/{dataset}")

    #else:
        #print(f"Dataset {dataset} already processed: skipping..")
        print("-------------------")

