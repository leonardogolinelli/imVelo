import scanpy as sc
import pickle
import os
import scvelo as scv
import gc
import numpy as np

# Updated models list to include ivelo_filtered and velovi_filtered
models = ["imVelo", "lsvelo", "celldancer", "deepvelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "scvelo", "stochastic"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
#datasets = ["dentategyrus_lamanno_P5"]

# Load existing velocities dictionary if it exists
if os.path.exists('velocities.pkl'):
    with open('velocities.pkl', 'rb') as f:
        velocities = pickle.load(f)
        print("Loaded existing velocities dictionary.")
else:
    velocities = {}
    print("No existing velocities dictionary found, starting fresh.")

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

    # If the dataset is not in velocities, initialize it
    if dataset not in velocities:
        velocities[dataset] = {}

    for model in models:
        # Check if the velocities for this model have already been loaded and saved
        if model in velocities[dataset]:
            print(f"Model {model} in dataset {dataset} already processed in velocities. Skipping loading.")
            continue
        
        print(f"Processing model: {model}")
        path = f"../{model}/{dataset}/{model}_{dataset}.h5ad"
        
        try:
            adata = sc.read_h5ad(path, backed='r')  # Lazy load AnnData objects in 'backed' mode
            velocities[dataset][model] = adata.layers["velocity"]  # Extract the velocity layer
        except Exception as e:
            print(f"Failed to load velocity for model {model} in dataset {dataset}: {e}")
            continue

        # Save the updated velocities dictionary after processing each model
        with open('velocities.pkl', 'wb') as f:
            pickle.dump(velocities, f)
            print(f"Saved velocities for model {model} in dataset {dataset}")

    # At this point, we have stored the velocities for all models in the dataset
    # Now we can compute the correlations (gene-wise and cell-wise)

    if len(velocities[dataset]) > 1:
        model_pairs = [(m1, m2) for i, m1 in enumerate(models) for m2 in models[i+1:] if m1 in velocities[dataset] and m2 in velocities[dataset]]
        
        for m1, m2 in model_pairs:
            # Ensure model pairs are always stored in a consistent order: (m1, m2) is the same as (m2, m1)
            model_pair = tuple(sorted([m1, m2]))

            # Check if correlations are already computed
            if model_pair in gene_wise_correlations.get(dataset, {}) and model_pair in cell_wise_correlations.get(dataset, {}):
                print(f"Correlation between models {m1} and {m2} for dataset {dataset} already processed. Skipping...")
                continue

            print(f"Computing correlations between models {m1} and {m2} for dataset {dataset}")

            # Get shared genes (var_names) and shared cells (obs_names) for this pair
            adata_m1 = sc.read_h5ad(f"../{m1}/{dataset}/{m1}_{dataset}.h5ad", backed='r')  # Lazy load
            adata_m2 = sc.read_h5ad(f"../{m2}/{dataset}/{m2}_{dataset}.h5ad", backed='r')

            # Use set operations for intersection specific to this model pair
            shared_genes = set(adata_m1.var_names).intersection(adata_m2.var_names)
            shared_cells = set(adata_m1.obs_names).intersection(adata_m2.obs_names)

            # Create boolean masks for subsetting the velocity matrices based on shared genes and cells
            gene_mask_m1 = adata_m1.var_names.isin(shared_genes)
            gene_mask_m2 = adata_m2.var_names.isin(shared_genes)

            cell_mask_m1 = adata_m1.obs_names.isin(shared_cells)
            cell_mask_m2 = adata_m2.obs_names.isin(shared_cells)

            # Subset velocity matrices to shared genes and cells for this model pair
            v_m1 = velocities[dataset][m1][cell_mask_m1, :][:, gene_mask_m1]
            v_m2 = velocities[dataset][m2][cell_mask_m2, :][:, gene_mask_m2]

            # Skip if either matrix is already empty after intersection
            if v_m1.size == 0 or v_m2.size == 0:
                print(f"Skipping correlation for models {m1} and {m2} in dataset {dataset} due to empty matrix after intersection.")
                continue

            # **Filter out genes with NaNs, do not filter out cells**
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
            del v_m1, v_m2, gene_corr, cell_corr, adata_m1, adata_m2
            gc.collect()  # Invoke garbage collection

    # Save the velocities dictionary after processing each dataset
    with open('velocities.pkl', 'wb') as f:
        pickle.dump(velocities, f)
        print(f"Saved velocities for dataset {dataset}")

print("All datasets processed and saved.")
