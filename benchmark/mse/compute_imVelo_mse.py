import scanpy as sc
import numpy as np

datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
model = "imVelo"

for dataset in datasets:
    print(f"processing dataset: {dataset}")
    adata = sc.read_h5ad(f"../{model}_evaluation_set/{dataset}/adata.h5ad")
    
    # Get the test indices from adata.uns
    test_indices = adata.uns["test_indices"]
    
    # Compute the MSE only on the test set
    mse = np.mean((adata.obsm["recons"][test_indices] - adata.obsm["MuMs"][test_indices])**2, axis=1)
    
    # Save the MSE for the test set
    np.save(f"../matrices/matrix_folder/test_mse_{model}_{dataset}", mse)
