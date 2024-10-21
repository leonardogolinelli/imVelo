import os
import numpy as np
import pandas as pd
import gseapy as gp  # Ensure you have gseapy installed

# Define datasets
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters", "celltype", "clusters"]

# Path to the matrices folder containing Ms and Mu layers
matrices_dir = "../matrices/matrix_folder"

# Function to compute mean expression for each cell type and perform GSEA
def compute_mean_expression_and_gsea(layer, var_names, cell_types, gene_sets, output_dir):
    """
    Compute the mean expression (Ms or Mu) of genes across cells in each cell type,
    rank the genes, and perform gene set enrichment analysis (GSEA).
    
    Parameters:
    - layer: The expression matrix (cells x genes).
    - var_names: List of gene names corresponding to the columns of the matrix.
    - cell_types: Array of cell type annotations for each cell.
    - gene_sets: Gene set database for GSEA (e.g., Reactome or MSigDB).
    - output_dir: Directory to save the GSEA results.
    
    Returns:
    - gsea_results: Dictionary of GSEA results for each cell type.
    """
    unique_cell_types = np.unique(cell_types)
    gsea_results = {}

    for cell_type in unique_cell_types:
        print(f"Processing GSEA for cell type: {cell_type}")

        # Subset the matrix for the current cell type
        cell_type_mask = (cell_types == cell_type)
        layer_subset = layer[cell_type_mask, :]

        # Compute the mean expression for each gene across cells of this cell type
        avg_expression = np.mean(layer_subset, axis=0)

        # Create a DataFrame with gene names and mean expression, convert gene names to uppercase
        gene_expression_df = pd.DataFrame({
            'gene': var_names.str.upper(),  # Ensure gene symbols are in uppercase
            'avg_expression': avg_expression
        }).set_index('gene')

        # Rank the genes based on mean expression
        gene_expression_df = gene_expression_df.sort_values(by='avg_expression', ascending=False)

        # Perform GSEA using the ranked gene list
        gsea_res = gp.prerank(rnk=gene_expression_df, gene_sets=gene_sets, min_size=5, max_size=1000, outdir=None)

        # Save the GSEA results directly into the output_dir
        gsea_res.res2d.to_csv(os.path.join(output_dir, f"{cell_type}_gsea_results.csv"))

        # Store the GSEA result in memory as well
        gsea_results[cell_type] = gsea_res

    return gsea_results

# Loop over each dataset and process Ms and Mu layers
for dataset, cell_type_key in zip(datasets, cell_type_keys):
    print(f"Processing dataset: {dataset}")

    # Define paths for the Ms and Mu layers and the cell types
    Ms_path = os.path.join(matrices_dir, f"Ms_{dataset}.npy")
    Mu_path = os.path.join(matrices_dir, f"Mu_{dataset}.npy")
    celltypes_path = os.path.join(matrices_dir, f"celltypes_{dataset}.npy")
    var_names_path = os.path.join(matrices_dir, f"var_names_imVelo_{dataset}.npy")  # Using var_names from imVelo, adjust as necessary

    # Check if the required files exist
    if not os.path.exists(Ms_path) or not os.path.exists(Mu_path) or not os.path.exists(celltypes_path) or not os.path.exists(var_names_path):
        print(f"Files for dataset {dataset} not found, skipping...")
        continue

    # Load the Ms and Mu layers, cell types, and gene names
    Ms_layer = np.load(Ms_path)
    Mu_layer = np.load(Mu_path)
    cell_types = np.load(celltypes_path, allow_pickle=True)
    var_names = pd.Series(np.load(var_names_path, allow_pickle=True)).str.upper()  # Ensure gene names are uppercase

    # Define the gene sets for GSEA (e.g., from MSigDB or Reactome)
    gene_sets = 'Reactome_2022'  # Example, adjust as needed or provide custom sets

    # Process Ms layer and perform GSEA
    Ms_output_dir = os.path.join("gsea_expression_results", dataset, "Ms")
    os.makedirs(Ms_output_dir, exist_ok=True)
    print(f"Performing GSEA for Ms layer in dataset {dataset}")
    gsea_results_Ms = compute_mean_expression_and_gsea(Ms_layer, var_names, cell_types, gene_sets, Ms_output_dir)

    # Process Mu layer and perform GSEA
    Mu_output_dir = os.path.join("gsea_expression_results", dataset, "Mu")
    os.makedirs(Mu_output_dir, exist_ok=True)
    print(f"Performing GSEA for Mu layer in dataset {dataset}")
    gsea_results_Mu = compute_mean_expression_and_gsea(Mu_layer, var_names, cell_types, gene_sets, Mu_output_dir)

    print(f"GSEA completed for dataset {dataset}. Results saved for Ms and Mu layers.")
