import scanpy as sc
import numpy as np
import os
from velocity_coherence_v2 import compute_velocity_coherence

# Define models, datasets, and cell type keys
models = ["imVelo", "lsvelo", "celldancer", "deepvelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "scvelo", "stochastic"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters", "celltype", "clusters"]

datasets = ["dentategyrus_lamanno_P5"]
models = ["imVelo"]

# Path to the matrix folder where the coherence matrices will be saved
output_dir = "../matrices/matrix_folder"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop over each dataset and model
for dataset, cell_type_key in zip(datasets, cell_type_keys):
    for model in models:
        print(f"Processing dataset: {dataset}, model: {model}")

        # Define paths for each matrix
        coherence_hadamard_path = os.path.join(output_dir, f"coherence_hadamard_{model}_{dataset}.npy")
        coherence_hadamard_normalized_path = os.path.join(output_dir, f"coherence_hadamard_normalized_{model}_{dataset}.npy")
        coherence_cosine_path = os.path.join(output_dir, f"coherence_cosine_{model}_{dataset}.npy")
        absolute_coherence_cosine_path = os.path.join(output_dir, f"absolute_coherence_cosine_{model}_{dataset}.npy")
        coherence_fraction_path = os.path.join(output_dir, f"coherence_fraction_{model}_{dataset}.npy")

        # Check if all matrices already exist to avoid unnecessary data loading
        if all(os.path.exists(path) for path in [
            coherence_hadamard_path, 
            coherence_hadamard_normalized_path, 
            coherence_cosine_path, 
            absolute_coherence_cosine_path,
            coherence_fraction_path
        ]):
            print(f"All coherence matrices already exist for {model} - {dataset}. Skipping...")
            continue

        # Load the AnnData object if any of the files are missing
        print(f"Loading data for {model} - {dataset}")
        adata_path = f"../{model}/{dataset}/{model}_{dataset}.h5ad"
        adata = sc.read_h5ad(adata_path)

        # Compute velocity coherence
        compute_velocity_coherence(adata, dataset, cell_type_key)

        # Save each matrix only if missing
        if not os.path.exists(coherence_hadamard_path):
            coherence_hadamard = adata.obs["coherence_hadamard"]
            np.save(coherence_hadamard_path, coherence_hadamard)
            print(f"Saved: {coherence_hadamard_path}")

        if not os.path.exists(coherence_hadamard_normalized_path):
            coherence_hadamard_normalized = adata.obs["coherence_hadamard_normalized"]
            np.save(coherence_hadamard_normalized_path, coherence_hadamard_normalized)
            print(f"Saved: {coherence_hadamard_normalized_path}")

        if not os.path.exists(coherence_cosine_path):
            coherence_cosine = adata.obs["coherence_cosine"]
            np.save(coherence_cosine_path, coherence_cosine)
            print(f"Saved: {coherence_cosine_path}")

        if not os.path.exists(absolute_coherence_cosine_path):
            absolute_coherence_cosine = adata.obs["absolute_coherence_cosine"]
            np.save(absolute_coherence_cosine_path, absolute_coherence_cosine)
            print(f"Saved: {absolute_coherence_cosine_path}")
        
        if not os.path.exists(coherence_fraction_path):
            coherence_fraction = adata.obs["coherence_fraction"]
            np.save(coherence_fraction_path, coherence_fraction)
            print(f"Saved: {coherence_fraction_path}")
