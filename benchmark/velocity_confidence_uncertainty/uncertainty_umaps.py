import os
import matplotlib.pyplot as plt
import scanpy as sc

# Define models and datasets
models = ["imVelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]

# Define metrics and output directories
metrics = ["velocity_length", "velocity_confidence", "intrinsic_uncertainty", "extrinsic_uncertainty"]
output_dirs = {metric: f"./plots/{metric}_umap_plots" for metric in metrics}
for output_dir in output_dirs.values():
    os.makedirs(output_dir, exist_ok=True)

# Function to check if all plots for a dataset/model combination already exist
def all_plots_exist(dataset, model):
    missing_files = []
    for metric in metrics:
        output_file = os.path.join(output_dirs[metric], f"{dataset}_{model}_{metric}_umap.png")
        if not os.path.exists(output_file):
            missing_files.append(output_file)  # Collect missing files
    if missing_files:
        print(f"Missing plots for {dataset} - {model}: {missing_files}")
        return False
    return True  # All plots exist

# Function to generate and save UMAP scatter plots for each uncertainty metric
def plot_uncertainty_on_umap(adata, dataset, model, metric):
    if "X_umap" in adata.obsm and metric in adata.obs:
        umap_coords = adata.obsm["X_umap"]
        adata.obs["intrinsic_uncertainty"] = adata.obs["directional_cosine_sim_variance"].copy()
        adata.obs["extrinsic_uncertainty"] = adata.obs["directional_cosine_sim_variance_extrinisic"].copy()

        uncertainty_values = adata.obs[metric]

        # Create scatter plot on UMAP
        plt.figure(figsize=(10, 8))
        plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c=uncertainty_values, cmap="viridis", s=20)
        plt.colorbar(label=metric.replace("_", " ").capitalize())
        plt.title(f"{metric.replace('_', ' ').capitalize()} UMAP for {dataset.capitalize()} - {model}", fontsize=25)
        plt.xlabel("UMAP 1", fontsize=18)
        plt.ylabel("UMAP 2", fontsize=18)

        # Save plot
        output_file = os.path.join(output_dirs[metric], f"{dataset}_{model}_{metric}_umap.png")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    else:
        print(f"Warning: {metric} or UMAP coordinates missing in {dataset} - {model}")

# Generate UMAP plots for each dataset, model, and metric
for dataset in datasets:
    print(f"Processing dataset: {dataset}")
    for model in models:
        print(f"Processing model: {model}")
        # Check if all required plots exist for this dataset/model combination
        if all_plots_exist(dataset, model):
            print(f"All plots for {dataset} - {model} already exist. Skipping...")
            continue  # Skip loading and plotting

        # Load the AnnData object for each dataset and model
        adata_path = f"../{model}/{dataset}/{model}_{dataset}.h5ad"
        if os.path.exists(adata_path):
            adata = sc.read_h5ad(adata_path)

            for metric in metrics:
                plot_uncertainty_on_umap(adata, dataset, model, metric)
        else:
            print(f"Warning: AnnData file not found for {model} - {dataset}")
