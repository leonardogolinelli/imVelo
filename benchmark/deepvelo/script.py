import numpy as np
import pandas as pd
import scvelo as scv
import scanpy as sc
from scvelo_adapted_utils import load_files
from scvelo_adapted_metrics import compute_scvelo_metrics, deg_genes
from scvelo_adapted_plots import plot_important_genes
import os
import torch
import deepvelo as dv  # Make sure to import deepvelo

datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters", "celltype", "clusters"]

for dataset, cell_type_key in zip(datasets, cell_type_keys):
    adata_path = f"../imVelo/{dataset}/imVelo_{dataset}.h5ad"
    adata = sc.read_h5ad(adata_path)
    
    # Remove existing velocities and embeddings if present
    if 'velocity' in adata.layers:
        del adata.layers["velocity"]
    if 'velocity_u' in adata.layers:
        del adata.layers["velocity_u"]
    if 'z' in adata.obsm:
        del adata.obsm["z"]
    
    # Modify default configurations
    dv.Constants._default_configs['loss']['args']['inner_batch_size'] = 1024
    dv.Constants._default_configs['arch']['args']['pred_unspliced'] = True  # Enable unspliced velocity prediction

    # Train the model and get the trainer object
    trainer = dv.train(adata, dv.Constants._default_configs)

    # Evaluate the model to extract velocities and latent embeddings
    eval_loader = trainer.data_loader  # Use the same data loader
    velo_mat_s, velo_mat_u, latent_z = trainer.eval(eval_loader)

    # Add velocities to adata
    adata.layers['velocity'] = velo_mat_s
    adata.layers['velocity_u'] = velo_mat_u

    print(latent_z)

    # Check if latent_z is a dictionary and extract the relevant part
    if isinstance(latent_z, dict):
        latent_z = np.array(list(latent_z.values()))  # Convert dictionary values to a numpy array



    # Add latent embeddings to adata
    adata.obsm['z'] = latent_z

    # Save the adata object with velocities and embeddings
    os.makedirs(dataset, exist_ok=True)
    adata.write_h5ad(f"{dataset}/deepvelo_{dataset}.h5ad")

    # Save the model's state_dict at the end of training
    torch.save(trainer.model.state_dict(), f"{dataset}/trainer.pth")

    print("--------------------------------")
    print(f"Dataset {dataset} processed. Proceeding with analysis...")

    # Continue with your analysis
    # Compute metrics and perform downstream analyses
    """compute_scvelo_metrics(adata, dataset, False, cell_type_key)
    adata.write_h5ad(f"{dataset}/deepvelo_{dataset}.h5ad")
    plot_important_genes(adata, dataset, cell_type_key)
    deg_genes(adata, dataset, cell_type_key=cell_type_key)
    print(f"Analysis for dataset {dataset} finished.")"""

print("All datasets processed.")
