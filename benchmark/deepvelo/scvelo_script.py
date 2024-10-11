import numpy as np
import pandas as pd
import scvelo as scv
import scanpy as sc
from scvelo_adapted_utils import load_files
from scvelo_adapted_metrics import compute_scvelo_metrics, deg_genes
from scvelo_adapted_plots import plot_important_genes
import os
import torch
#import deepvelo as dv

datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters","celltype", "clusters"]

for dataset, cell_type_key in zip(datasets, cell_type_keys):
    """adata_path = f"../imVelo/{dataset}/imVelo_{dataset}.h5ad"
    adata = sc.read_h5ad(adata_path)
    del adata.layers["velocity"]
    del adata.layers["velocity_u"]
    del adata.obsm["z"]

    # Modify default configurations
    dv.Constants._default_configs['loss']['args']['inner_batch_size'] = 1024
    dv.Constants._default_configs['arch']['args']['pred_unspliced'] = True  # Enable unspliced velocity prediction

    # Train the model and get the trainer object
    trainer = dv.train(adata, dv.Constants._default_configs)

    # Evaluate the model to extract velocities
    eval_loader = trainer.data_loader  # Use the same data loader
    velo_mat_s, velo_mat_u, _ = trainer.eval(eval_loader)

    # Add velocities to adata
    adata.layers['velocity'] = velo_mat_s
    adata.layers['velocity_u'] = velo_mat_u

    # Save the adata object with velocities
    os.makedirs(dataset, exist_ok=True)
    adata.write_h5ad(f"{dataset}/deepvelo_{dataset}.h5ad")

    # Save the model's state_dict at the end of training
    torch.save(trainer.model.state_dict(), f"{dataset}/trainer.pth")"""
    print("--------------------------------")

    adata = sc.read_h5ad(f"{dataset}/deepvelo_{dataset}.h5ad")
    # this will train the model and predict the velocity vectore. The result is stored in adata.layers['velocity']. You can use trainer.model to access the model.
    compute_scvelo_metrics(adata, dataset, False, cell_type_key)
    adata.write_h5ad(f"{dataset}/deepvelo_{dataset}.h5ad")
    plot_important_genes(adata, dataset, cell_type_key)
    deg_genes(adata, dataset, cell_type_key=cell_type_key)
    print("analysis finished")

#else:
    print(f"Dataset {dataset} already processed: skipping..")

