#THIS CODE MUST BE RUN WITHIN THE IVELO REPO

"""
import scanpy as sc
import numpy as np
from _model import IVELO

# Specify dataset(s) and model name
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
model_names = ["ivelo", "ivelo_filtered"]

for model_name in model_names:
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        adata = sc.read_h5ad(f"../../imVelo/benchmark/{model_name}/{dataset}/{model_name}_{dataset}.h5ad")

        # Initialize the VELOVI model with the AnnData object
        IVELO.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
        model = IVELO(adata)

        # Load the pretrained model
        model_path = f"../../imVelo/benchmark/{model_name}/{dataset}/{model_name}_{dataset}"
        model = IVELO.load(model_path, adata=adata)

        # Get the training indices
        train_indices = model.train_indices

        # Infer the test indices
        all_indices = np.arange(adata.n_obs)
        test_indices = np.setdiff1d(all_indices, train_indices)

        # Get reconstructed spliced and unspliced RNA counts
        reconstructed_spliced, reconstructed_unspliced = model.get_expression_fit(
            adata, return_mean=True, return_numpy=True
        )

        # Concatenate reconstructed spliced and unspliced
        reconstructed = np.concatenate([reconstructed_unspliced, reconstructed_spliced], axis=1)

        # Get original input Mu and Ms
        input_spliced = adata.layers["Ms"]
        input_unspliced = adata.layers["Mu"]

        # Concatenate input Mu and Ms
        original = np.concatenate([input_unspliced, input_spliced], axis=1)

        # Compute MSE for the test cells only
        original_test = original[test_indices, :]
        reconstructed_test = reconstructed[test_indices, :]

        mse_per_cell_test = np.mean((original_test - reconstructed_test) ** 2, axis=1)
        
        np.save(f"../../imVelo/benchmark/matrices/matrix_folder/test_mse_{model_name}_{dataset}", mse_per_cell_test)
        
        print(f"Mean squared error for dataset {dataset} has been stored in adata.obs['mse_reconstruction'].")

"""