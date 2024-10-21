import pickle
import os
import gc
import numpy as np
import scvelo as scv

# Updated models list to include ivelo_filtered and velovi_filtered
models = ["imVelo", "lsvelo", "celldancer", "deepvelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "scvelo", "stochastic"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]

# Define the output directory where the saved files are located
output_dir = "../matrices/matrix_folder"

# Load existing gene-wise correlations if they exist
if os.path.exists('gene_wise_correlations.pkl'):
    with open('gene_wise_correlations.pkl', 'rb') as f:
        gene_wise_correlations = pickle.load(f)
else:
    gene_wise_correlations = {}

# Load existing cell-wise correlations if they exist
if os.path.exists('cell_wise_correlations.pkl'):
    with open('cell_wise_correlations.pkl', 'rb') as f:
        cell_wise_correlations = pickle.load(f)
else:
    cell_wise_correlations = {}

# Iterate through datasets and models
for dataset in datasets:
    print(f"Processing dataset: {dataset}")

    # Check if the dataset has already been fully processed (skip if done)
    if dataset in gene_wise_correlations and len(gene_wise_correlations[dataset]) == len(models) * (len(models) - 1):
        print(f"Dataset {dataset} already fully processed. Skipping...")
        continue

    for m1, m2 in [(m1, m2) for i, m1 in enumerate(models) for m2 in models[i+1:]]:
        model_pair = tuple(sorted([m1, m2]))  # Store model pairs in consistent order

        # Check if correlations are already computed
        if model_pair in gene_wise_correlations.get(dataset, {}) and model_pair in cell_wise_correlations.get(dataset, {}):
            print(f"Correlation between models {m1} and {m2} for dataset {dataset} already processed. Skipping...")
            continue

        print(f"Computing correlations between models {m1} and {m2} for dataset {dataset}")

        # Load the velocity matrices from the output directory
        velocity_path_m1 = os.path.join(output_dir, f"velocity_{m1}_{dataset}.npy")
        velocity_path_m2 = os.path.join(output_dir, f"velocity_{m2}_{dataset}.npy")

        if not (os.path.exists(velocity_path_m1) and os.path.exists(velocity_path_m2)):
            print(f"Velocity files for model pair {m1} and {m2} in dataset {dataset} not found. Skipping...")
            continue

        try:
            v_m1 = np.load(velocity_path_m1)
            v_m2 = np.load(velocity_path_m2)
        except Exception as e:
            print(f"Failed to load velocities for model pair {m1} and {m2} in dataset {dataset}: {e}")
            continue

        # Load var_names and obs_names
        var_names_m1 = np.load(os.path.join(output_dir, f"var_names_{m1}_{dataset}.npy"), allow_pickle=True)
        var_names_m2 = np.load(os.path.join(output_dir, f"var_names_{m2}_{dataset}.npy"), allow_pickle=True)
        obs_names_m1 = np.load(os.path.join(output_dir, f"obs_names_{m1}_{dataset}.npy"), allow_pickle=True)
        obs_names_m2 = np.load(os.path.join(output_dir, f"obs_names_{m2}_{dataset}.npy"), allow_pickle=True)

        # Use set operations for intersection specific to this model pair
        shared_genes = set(var_names_m1).intersection(var_names_m2)
        shared_cells = set(obs_names_m1).intersection(obs_names_m2)

        if not shared_genes or not shared_cells:
            print(f"No shared genes or cells between models {m1} and {m2} for dataset {dataset}. Skipping...")
            continue

        # Create boolean masks for subsetting the velocity matrices based on shared genes and cells
        gene_mask_m1 = np.isin(var_names_m1, list(shared_genes))
        gene_mask_m2 = np.isin(var_names_m2, list(shared_genes))

        cell_mask_m1 = np.isin(obs_names_m1, list(shared_cells))
        cell_mask_m2 = np.isin(obs_names_m2, list(shared_cells))

        # Subset velocity matrices to shared genes and cells for this model pair
        v_m1 = v_m1[cell_mask_m1, :][:, gene_mask_m1]
        v_m2 = v_m2[cell_mask_m2, :][:, gene_mask_m2]

        # Skip if either matrix is already empty after intersection
        if v_m1.size == 0 or v_m2.size == 0:
            print(f"Skipping correlation for models {m1} and {m2} in dataset {dataset} due to empty matrix after intersection.")
            continue

        # Filter out genes with NaNs, do not filter out cells
        nan_mask_m1_genes = ~np.isnan(v_m1).any(axis=0)  # Remove genes (columns) with NaNs
        nan_mask_m2_genes = ~np.isnan(v_m2).any(axis=0)

        # Subset the matrices again to exclude genes with NaNs
        v_m1 = v_m1[:, nan_mask_m1_genes]
        v_m2 = v_m2[:, nan_mask_m2_genes]

        # Ensure both matrices have the same number of genes after filtering
        min_genes = min(v_m1.shape[1], v_m2.shape[1])
        v_m1 = v_m1[:, :min_genes]
        v_m2 = v_m2[:, :min_genes]

        # Skip correlation computation if either matrix becomes empty after NaN removal
        if v_m1.shape[1] == 0 or v_m2.shape[1] == 0:
            print(f"Skipping correlation for models {m1} and {m2} in dataset {dataset} due to empty matrix after NaN removal.")
            continue

        # Gene-wise correlations (columns = genes, across shared cells)
        gene_corr = scv.utils.vcorrcoef(v_m1, v_m2, axis=1)  # Correlation per gene across cells
        gene_wise_correlations.setdefault(dataset, {})[model_pair] = gene_corr

        # Cell-wise correlations (rows = cells, across shared genes)
        cell_corr = scv.utils.vcorrcoef(v_m1, v_m2, axis=0)  # Correlation per cell across genes
        cell_wise_correlations.setdefault(dataset, {})[model_pair] = cell_corr

        # **Copy and store the reverse pair**
        gene_wise_correlations[dataset][(m2, m1)] = gene_corr
        cell_wise_correlations[dataset][(m2, m1)] = cell_corr

        # Save the correlations after each model pair (to reduce memory usage)
        with open('gene_wise_correlations.pkl', 'wb') as f:
            pickle.dump(gene_wise_correlations, f)
            print(f"Saved gene-wise correlations for models {m1} and {m2} (and {m2}, {m1}) in dataset {dataset}.")

        with open('cell_wise_correlations.pkl', 'wb') as f:
            pickle.dump(cell_wise_correlations, f)
            print(f"Saved cell-wise correlations for models {m1} and {m2} (and {m2}, {m1}) in dataset {dataset}.")

        # Free up memory explicitly
        del v_m1, v_m2, gene_corr, cell_corr
        gc.collect()  # Invoke garbage collection

print("All datasets processed and saved.")
