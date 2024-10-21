import numpy as np
import pandas as pd
import scanpy as sc
import os
import gseapy as gp  # Ensure you have gseapy installed

datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters", "celltype", "clusters"]

def compute_average_velocity_and_gsea(adata, cell_type_key, gene_sets, output_dir):
    """
    Compute the average velocity of genes across cells in a particular cell type
    and perform gene set enrichment analysis for each cell type using the ranked list of gene weights.
    
    Parameters:
    - adata: AnnData object containing velocity data.
    - cell_type_key: Column name in adata.obs that contains cell type annotations.
    - gene_sets: Dictionary of gene sets for GSEA (e.g., from MSigDB).
    - output_dir: Directory to save the GSEA results.
    
    Returns:
    - gsea_results: Dictionary of GSEA results for each cell type.
    """
    cell_types = adata.obs[cell_type_key].unique()
    gsea_results = {}

    for cell_type in cell_types:
        # Subset adata for the current cell type
        cell_type_mask = adata.obs[cell_type_key] == cell_type
        adata_subset = adata[cell_type_mask]
        
        # Compute average velocity for each gene across cells of the current cell type


        # UNNECESSARY ------------
        avg_velocity_u = np.mean(adata_subset.layers["velocity_u"], axis=0) #
        avg_velocity_s = np.mean(adata_subset.layers["velocity"], axis=0)
        
        # Combine unspliced and spliced velocities
        avg_velocity = avg_velocity_u + avg_velocity_s
        
        # Create a DataFrame with gene names and velocities
        gene_velocity_df = pd.DataFrame({
            'gene': adata.var_names.str.upper(),  # Convert gene symbols to uppercase
            'avg_velocity': avg_velocity
        }).set_index('gene')

        # Rank the genes based on velocity
        gene_velocity_df = gene_velocity_df.sort_values(by='avg_velocity', ascending=False)

        # Perform GSEA using the ranked gene list
        gene_list = gene_velocity_df['avg_velocity'].rank(method='first', ascending=False).to_list()

        # Using gseapy for GSEA with adjusted min_size and max_size
        gsea_res = gp.prerank(rnk=gene_velocity_df, gene_sets=gene_sets, min_size=5, max_size=1000, outdir=None)
        
        # Save the GSEA results directly into the output_dir
        gsea_res.res2d.to_csv(os.path.join(output_dir, f"{cell_type}_gsea_results.csv"))
        
        # Store the GSEA result in memory as well
        gsea_results[cell_type] = gsea_res

    return gsea_results

for dataset, cell_type_key in zip(datasets, cell_type_keys):
    print(f"Processing dataset: {dataset}")
    adata_path = f"{dataset}/lsvelo_{dataset}.h5ad"
    adata = sc.read_h5ad(adata_path)

    # Load the gene sets for GSEA (you can define these or use an external resource like MSigDB)
    gene_sets = 'Reactome_2022'  # Example, adjust as needed or provide custom sets

    output_dir = f"{dataset}/gsea_results/{gene_sets}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the function and save results
    gsea_results = compute_average_velocity_and_gsea(adata, cell_type_key, gene_sets, output_dir)

