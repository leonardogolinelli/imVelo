import os
import numpy as np
import pandas as pd
import gseapy as gp  # Ensure you have gseapy installed

# Define datasets and models
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
models = ["imVelo", "lsvelo", "celldancer", "deepvelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "scvelo", "stochastic"]

# Path to the matrices folder containing the velocities and cell types
matrices_dir = "../matrices/matrix_folder"

def compute_average_velocity_and_gsea(velocity_matrix, var_names, cell_types, gene_sets, output_dir):
    """
    Compute the average velocity of genes across cells in a particular cell type
    and perform gene set enrichment analysis (GSEA) for each cell type using the ranked list of gene weights.
    
    Parameters:
    - velocity_matrix: Normalized velocity matrix (cells x genes).
    - var_names: List of gene names corresponding to the columns of the velocity matrix.
    - cell_types: Array of cell type annotations for each cell.
    - gene_sets: Gene set database for GSEA (e.g., MSigDB or Reactome).
    - output_dir: Directory to save the GSEA results.
    
    Returns:
    - gsea_results: Dictionary of GSEA results for each cell type.
    """
    unique_cell_types = np.unique(cell_types)
    gsea_results = {}

    for cell_type in unique_cell_types:
        print(f"Processing GSEA for cell type: {cell_type}")
        
        # Subset the velocity matrix for the current cell type
        cell_type_mask = (cell_types == cell_type)
        velocity_subset = velocity_matrix[cell_type_mask, :]
        
        # Compute the average velocity for each gene across the cells of this cell type
        avg_velocity = np.mean(velocity_subset, axis=0)
        
        # Create a DataFrame with gene names and velocities, convert gene names to uppercase
        gene_velocity_df = pd.DataFrame({
            'gene': var_names.str.upper(),  # Ensure gene symbols are in uppercase
            'avg_velocity': avg_velocity
        }).set_index('gene')

        # Rank the genes based on velocity
        gene_velocity_df = gene_velocity_df.sort_values(by='avg_velocity', ascending=False)

        # Perform GSEA using the ranked gene list
        gsea_res = gp.prerank(rnk=gene_velocity_df, gene_sets=gene_sets, min_size=5, max_size=1000, outdir=None)
        
        # Save the GSEA results directly into the output_dir
        gsea_res.res2d.to_csv(os.path.join(output_dir, f"{cell_type}_gsea_results.csv"))
        
        # Store the GSEA result in memory as well
        gsea_results[cell_type] = gsea_res

    return gsea_results

# Loop over each dataset and model
for dataset in datasets:
    for model in models:
        print(f"Processing dataset: {dataset}, model: {model}")
        
        # Define paths for the velocity matrix and cell type annotations
        velocity_path = os.path.join(matrices_dir, f"normalized_velocity_{model}_{dataset}.npy")
        celltypes_path = os.path.join(matrices_dir, f"celltypes_{dataset}.npy")
        var_names_path = os.path.join(matrices_dir, f"var_names_{model}_{dataset}.npy")
        
        # Check if the required files exist
        if not os.path.exists(velocity_path) or not os.path.exists(celltypes_path) or not os.path.exists(var_names_path):
            print(f"Files for model {model} in dataset {dataset} not found, skipping...")
            continue

        # Load the velocity matrix, cell type annotations, and gene names
        velocity_matrix = np.load(velocity_path)
        cell_types = np.load(celltypes_path, allow_pickle=True)
        var_names = np.load(var_names_path, allow_pickle=True)

        # Convert var_names to a pandas Series if it is not already, and ensure all gene names are uppercase
        var_names = pd.Series(var_names).str.upper()

        # Load the gene sets for GSEA (you can define these or use an external resource like MSigDB)
        gene_sets = 'Reactome_2022'  # Example, adjust as needed or provide custom sets

        # Define the output directory for GSEA results
        output_dir = os.path.join("gsea_results", model, dataset)
        os.makedirs(output_dir, exist_ok=True)
        
        # Run GSEA for each cell type and save the results
        gsea_results = compute_average_velocity_and_gsea(velocity_matrix, var_names, cell_types, gene_sets, output_dir)
        print(f"GSEA completed for dataset {dataset}, model {model}. Results saved.")
