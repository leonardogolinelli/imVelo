import scanpy as sc
import scvelo as scv
import pandas as pd
import seaborn as sns
import os

def preprocess(
    dataset_name,
    cell_type_key=None,
    n_highly_var_genes=None,
    smooth_k=None,
    ):


    os.makedirs(f"{dataset_name}", exist_ok=True)
    if dataset_name == "forebrain":
        # (4) Human embryo glutamatergic neurogenesis dataset
        adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_forebrain_K11_knn_rep_ve_best_key_None_0_kl_weight_1e-9_1e-08_20k/forebrain/K11/adata/adata_K11_dt_ve.h5ad"
        smooth_k = 200  # Use 200 nearest neighbors as per your description

        # Load the data
        adata = sc.read_h5ad(adata_path)
        del adata.layers["velocity"]
        del adata.layers["velocity_u"]
        # Set up layers
        adata.layers["unspliced"] = adata.layers["counts_unspliced"].copy()
        adata.layers["spliced"] = adata.layers["counts_spliced"].copy()
        adata.X = adata.layers["counts_spliced"].copy()

        # **Preprocessing steps:**
        # - Keep cells with at least 200 genes expressed
        sc.pp.filter_cells(adata, min_genes=200)
        
        # - Keep genes that are expressed in at least 3 cells
        sc.pp.filter_genes(adata, min_cells=3)
        
        # - Identify all highly variable genes with default settings
        scv.pp.filter_and_normalize(adata, n_highly_var_genes=2000)
        
        # - Subset the data to include only highly variable genes
        adata = adata[:, adata.var['highly_variable']]
        
        # - Normalize per cell and log-transform the data
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        
        # - Confirm the number of cells and genes
        print(f"Number of cells: {adata.n_obs}")  # Should be 1,054
        print(f"Number of genes: {adata.n_vars}")  # Should be 1,720
        
        # - Calculate the first and second moments (needed for velocity estimation)
        scv.pp.moments(adata, n_neighbors=smooth_k, n_highly_var_genes=2000)
    
        return adata

    elif dataset_name == "pancreas":
        # (1) Pancreatic endocrinogenesis data
        adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_pancreas_K11_knn_rep_ve_best_key_None_0_kl_weight_1e-9_1e-8/pancreas/K11/adata/adata_K11_dt_ve.h5ad"

        # Load the data
        adata = sc.read_h5ad(adata_path)
        del adata.layers["velocity"]
        del adata.layers["velocity_u"]
        
        # Set up layers
        adata.layers["unspliced"] = adata.layers["counts_unspliced"].copy()
        adata.layers["spliced"] = adata.layers["counts_spliced"].copy()
        adata.X = adata.layers["counts_spliced"].copy()

        # **Preprocessing steps:**
        # - Follow the method by Bergen et al. in the scVelo study
        # - Filter and normalize the data, keeping 2,000 highly variable genes
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
        
        # - Confirm the number of cells and genes
        print(f"Number of cells: {adata.n_obs}")  # Should be 3,696
        print(f"Number of genes: {adata.n_vars}")  # Should be 2,000
        
        # - Calculate moments with default parameters
        scv.pp.moments(adata)
        
        return adata

    elif dataset_name == "dentategyrus_lamanno_P5":
        # (2) Mouse hippocampal dentate gyrus neurogenesis data
        adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_dentategyrus_lamanno_P5_K31_knn_rep_pca_best_key_pca_unique_0_kl_weight_1e-9_1e-08_20k_12_july/dentategyrus_lamanno_P5/K31/adata/adata_K31_dt_pca.h5ad"

        # Load the data
        adata = sc.read_h5ad(adata_path)
        del adata.layers["velocity"]
        del adata.layers["velocity_u"]
        
        # Set up layers
        adata.layers["unspliced"] = adata.layers["unspliced"].copy()
        adata.layers["spliced"] = adata.layers["spliced"].copy()
        adata.X = adata.layers["spliced"].copy()

        # **Preprocessing steps:**
        # - Follow the gene and cell filtering methods by La Manno et al.
        # - Since specific filtering parameters are not detailed, we'll use reasonable defaults
        # - Filter cells and genes
        sc.pp.filter_cells(adata, min_counts=500)
        sc.pp.filter_genes(adata, min_counts=50)
        
        # - Identify highly variable genes, selecting 2,159 genes
        sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes=2159)
        
        # - Subset the data to include only highly variable genes
        adata = adata[:, adata.var['highly_variable']]
        
        # - Normalize per cell and log-transform the data
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        
        # - Confirm the number of cells and genes
        print(f"Number of cells: {adata.n_obs}")  # Should be 18,140
        print(f"Number of genes: {adata.n_vars}")  # Should be 2,159
        
        # - Calculate moments with default parameters
        scv.pp.moments(adata)
        
        return adata

    elif dataset_name == "gastrulation_erythroid":
        # (3) Erythroid lineage of the mouse gastrulation data
        adata_path = "/path/to/gastrulation_erythroid_data.h5ad"
        smooth_k = 100  # Use 100 nearest neighbors as per your description

        # Load the data
        adata = sc.read_h5ad(adata_path)
        del adata.layers["velocity"]
        del adata.layers["velocity_u"]
        
        # Set up layers
        adata.layers["unspliced"] = adata.layers["counts_unspliced"].copy()
        adata.layers["spliced"] = adata.layers["counts_spliced"].copy()
        adata.X = adata.layers["counts_spliced"].copy()

        # Ensure 'cell_type_key' and 'stage' are provided in adata.obs
        if cell_type_key is None or 'stage' not in adata.obs:
            raise ValueError("Please provide 'cell_type_key' and ensure 'stage' is in adata.obs")
                
        # - Confirm the number of cells
        print(f"Number of cells after filtering: {adata.n_obs}")  # Should be 12,329
        
        # - Follow standard preprocessing procedures in scVelo with default parameters
        scv.pp.filter_and_normalize(adata, min_shared_counts=20)
        
        # - Calculate moments with 100 nearest neighbors
        scv.pp.moments(adata, n_neighbors=smooth_k, n_highly_var_genes=2000)
        
        return adata

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
