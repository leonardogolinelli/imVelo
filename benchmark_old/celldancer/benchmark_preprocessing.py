import scanpy as sc
import scvelo as scv
import numpy as np
import scipy.sparse as sp

def preprocess(dataset_name):
    if dataset_name == "forebrain":
        adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/benchmark/imVelo/imVelo_forebrain/forebrain/K11/adata/adata_K11_dt_ve.h5ad"
        adata = sc.read_h5ad(adata_path)
        
        # Convert layers to float32 and ensure sparse matrices are correctly cast
        adata.layers["unspliced"] = adata.layers["counts_unspliced"].copy().astype(np.float32)
        adata.layers["spliced"] = adata.layers["counts_spliced"].copy().astype(np.float32)

        # If adata.X is sparse, ensure it's cast to float32
        if sp.issparse(adata.X):
            adata.X = adata.X.astype(np.float32)
        else:
            adata.X = adata.layers["counts_spliced"].copy().astype(np.float32)

        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)

        # Identify highly variable genes
        sc.pp.highly_variable_genes(adata, n_top_genes=1720, flavor="seurat_v3")
        adata = adata[:, adata.var['highly_variable']]

        # Calculate the first and second moments (needed for velocity estimation)
        scv.pp.moments(adata, n_neighbors=200)

        return adata
        
    elif dataset_name == "pancreas":
        adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_pancreas_K11_knn_rep_ve_best_key_None_0_kl_weight_1e-9_1e-8/pancreas/K11/adata/adata_K11_dt_ve.h5ad"
        adata = sc.read_h5ad(adata_path)
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(adata)
        
    elif dataset_name == "dentategyrus_lamanno_P5":
        # (2) Mouse hippocampal dentate gyrus neurogenesis data
        adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_dentategyrus_lamanno_P5_K31_knn_rep_pca_best_key_pca_unique_0_kl_weight_1e-9_1e-08_20k_12_july/dentategyrus_lamanno_P5/K31/adata/adata_K31_dt_pca.h5ad"
        adata = sc.read_h5ad(adata_path)

        # Convert layers to float32 and ensure sparse matrices are correctly cast
        adata.layers["unspliced"] = adata.layers["counts_unspliced"].copy().astype(np.float32)
        adata.layers["spliced"] = adata.layers["counts_spliced"].copy().astype(np.float32)

        if sp.issparse(adata.X):
            adata.X = adata.X.astype(np.float32)
        else:
            adata.X = adata.layers["counts_spliced"].copy().astype(np.float32)

        # Preprocessing steps
        # 1. Filter cells and genes
        sc.pp.filter_cells(adata, min_genes=500)
        sc.pp.filter_genes(adata, min_cells=10)
        
        # 2. Normalize spliced and unspliced counts per cell (per the paper)
        sc.pp.normalize_total(adata, target_sum=1e4, layers=['spliced', 'unspliced'])
        
        # 3. Log-transform both layers
        sc.pp.log1p(adata, layer='spliced')
        sc.pp.log1p(adata, layer='unspliced')

        # 4. Identify 3,000 highly variable genes
        sc.pp.highly_variable_genes(adata, n_top_genes=2159)
        adata = adata[:, adata.var['highly_variable']]

        adata.layers["Mu"] = adata.layers["unspliced"].copy().toarray()
        adata.layers["Ms"] = adata.layers["spliced"].copy().toarray()

    elif dataset_name == "gastrulation_erythroid":
        adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_gastrulation_erythroid_K11_knn_rep_ve_best_key_None_0_kl_weight_1e-9_1e-08_20k_256/gastrulation_erythroid/K11/adata/adata_K11_dt_ve.h5ad"
        adata = sc.read_h5ad(adata_path)
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(adata, n_neighbors=100)
        
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    
    return adata
