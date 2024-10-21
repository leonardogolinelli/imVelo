import os
import numpy as np
import pandas as pd
import gseapy as gp  # gseapy provides an interface to Enrichr

# Define datasets
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters", "celltype", "clusters"]

# Path to the matrices folder containing Ms and Mu layers
matrices_dir = "../matrices/matrix_folder"

# Function to perform enrichment analysis using Enrichr on the top 500 expressed genes
def perform_enrichr_analysis(layer, var_names, cell_types, library, output_dir):
    """
    Perform enrichment analysis using Enrichr on the top 500 expressed genes for each cell type.
    
    Parameters:
    - layer: The expression matrix (cells x genes).
    - var_names: List of gene names corresponding to the columns of the matrix.
    - cell_types: Array of cell type annotations for each cell.
    - library: Enrichr gene set library to use (e.g., 'Reactome_2022').
    - output_dir: Directory to save the Enrichr results.
    
    Returns:
    - enrichr_results: Dictionary of Enrichr results for each cell type.
    """
    unique_cell_types = np.unique(cell_types)
    enrichr_results = {}

    for cell_type in unique_cell_types:
        print(f"Processing Enrichr analysis for cell type: {cell_type}")

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

        # Select the top 500 expressed genes
        top_500_genes = gene_expression_df.sort_values(by='avg_expression', ascending=False).head(500).index.tolist()

        # Perform Enrichr enrichment analysis using the top 500 genes
        enrichr_res = gp.enrichr(gene_list=top_500_genes, gene_sets=library, outdir=None)

        # Save the Enrichr results directly into the output_dir
        enrichr_res.results.to_csv(os.path.join(output_dir, f"{cell_type}_enrichr_results.csv"))

        # Store the Enrichr result in memory as well
        enrichr_results[cell_type] = enrichr_res

    return enrichr_results

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

    # Define the Enrichr library to use for enrichment analysis
    enrichr_library = 'Reactome_2022'  # Example, adjust as needed or provide custom sets

    # Process Ms layer and perform Enrichr analysis
    Ms_output_dir = os.path.join("enrichr_results", dataset, "Ms")
    os.makedirs(Ms_output_dir, exist_ok=True)
    print(f"Performing Enrichr analysis for Ms layer in dataset {dataset}")
    enrichr_results_Ms = perform_enrichr_analysis(Ms_layer, var_names, cell_types, enrichr_library, Ms_output_dir)

    # Process Mu layer and perform Enrichr analysis
    Mu_output_dir = os.path.join("enrichr_results", dataset, "Mu")
    os.makedirs(Mu_output_dir, exist_ok=True)
    print(f"Performing Enrichr analysis for Mu layer in dataset {dataset}")
    enrichr_results_Mu = perform_enrichr_analysis(Mu_layer, var_names, cell_types, enrichr_library, Mu_output_dir)

    print(f"Enrichr analysis completed for dataset {dataset}. Results saved for Ms and Mu layers.")
