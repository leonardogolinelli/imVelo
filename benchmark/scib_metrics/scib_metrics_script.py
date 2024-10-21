import os
import numpy as np
import scanpy as sc
import scib_metrics.benchmark
from scib_metrics.benchmark import Benchmarker
from functions import get_results, plot_results_table

# Define models, datasets, and cell type keys
models = ["expimap", "velovi", "ivelo", "imVelo"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters", "celltype", "clusters"]
matrix_folder = "../matrices/matrix_folder"

# Output folder for results
output_dir = "./bio_conservation_plots_unfiltered"
os.makedirs(output_dir, exist_ok=True)

# Load the cell types for each dataset
def load_cell_types(dataset):
    celltype_file = f"celltypes_{dataset}.npy"
    celltype_path = os.path.join(matrix_folder, celltype_file)
    
    if os.path.exists(celltype_path):
        return np.load(celltype_path, allow_pickle=True).tolist()
    else:
        raise FileNotFoundError(f"Cell type file for {dataset} not found.")
# Load the latent spaces for each model and dataset directly from the matrix folder
def load_latent_spaces(dataset):
    latent_spaces = {}
    for model in models:
        z_file = f"{matrix_folder}/z_{model}_{dataset}.npy"
        if os.path.exists(z_file):
            z_values = np.load(z_file, allow_pickle=True)
            if z_values.size > 0:  # Check if the latent space is non-empty
                latent_spaces[model] = z_values
            else:
                print(f"Warning: Latent space for {model} in {dataset} is empty.")
        else:
            print(f"Warning: Latent space for {model} in {dataset} not found.")
    return latent_spaces

# Perform benchmarking and plot results directly from latent spaces
for dataset, cell_type_key in zip(datasets, cell_type_keys):
    print(f"Processing dataset: {dataset}")
    
    # Load the AnnData object for the current dataset
    adata_path = f"../imVelo/{dataset}/imVelo_{dataset}.h5ad"
    if not os.path.exists(adata_path):
        print(f"Error: AnnData file for {dataset} not found at {adata_path}. Skipping dataset.")
        continue

    adata = sc.read_h5ad(adata_path)

    # Load the latent spaces for the current dataset
    latent_spaces = load_latent_spaces(dataset)
    
    if not latent_spaces:
        print(f"No latent spaces were found for dataset: {dataset}")
        continue  # Skip to the next dataset if no latent spaces are found

    # Add the latent spaces as obsm in the loaded AnnData object
    num_cells = adata.shape[0]
    for model, z_values in latent_spaces.items():
        if z_values.shape[0] != num_cells:
            print(f"Warning: Latent space for {model} in {dataset} does not match the number of cells.")
            continue
        adata.obsm[f"{dataset}-{model}"] = z_values

    # List of embedding keys for the latent spaces
    z_names = [f"{dataset}-{model}" for model in latent_spaces.keys()]
    save_path = os.path.join(output_dir, f"{dataset}_bio_conservation.png")

    # Initialize bio-conservation and batch-correction metrics
    bio_metrics = scib_metrics.benchmark.BioConservation()
    batch_metrics = scib_metrics.benchmark.BatchCorrection(
        False, False, False, False, False
    )

    adata.obs["batch"] = 0
    # Create the benchmarker
    bm = Benchmarker(
        adata,
        batch_key="batch",  # Ensure 'batch' column exists in the AnnData
        label_key=cell_type_key,  # Ensure 'cell_type' exists in the AnnData
        embedding_obsm_keys=z_names,
        bio_conservation_metrics=bio_metrics,
        batch_correction_metrics=batch_metrics,
        n_jobs=1  # Adjust the number of jobs if parallel processing is needed
    )
    
    sc.pp.neighbors(adata)
    # Run the benchmark
    try:
        bm.benchmark()
    except ValueError as e:
        print(f"Error during benchmarking for dataset {dataset}: {e}")
        continue  # Skip this dataset if there's an error

    # Get the results and plot them
    df = get_results(bm)
    plot_results_table(df, bm, save_path)
