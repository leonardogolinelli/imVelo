import scanpy as sc
import numpy as np

def preprocess(
    dataset_name,
    n_highly_var_genes
    ):
        if dataset_name == "forebrain":
            adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_forebrain_K11_knn_rep_ve_best_key_None_0_kl_weight_1e-9_1e-08_20k/forebrain/K11/adata/adata_K11_dt_ve.h5ad"
        elif dataset_name == "pancreas":
            adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_pancreas_K11_knn_rep_ve_best_key_None_0_kl_weight_1e-9_1e-8/pancreas/K11/adata/adata_K11_dt_ve.h5ad"
        elif dataset_name == "gastrulation_erythroid":
            adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_gastrulation_erythroid_K11_knn_rep_ve_best_key_None_0_kl_weight_1e-9_1e-08_20k_256/gastrulation_erythroid/K11/adata/adata_K11_dt_ve.h5ad"
        elif dataset_name == "dentategyrus_lamanno_P5":
            adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_dentategyrus_lamanno_P5_K31_knn_rep_pca_best_key_pca_unique_0_kl_weight_1e-9_1e-08_20k_12_july/dentategyrus_lamanno_P5/K31/adata/adata_K31_dt_pca.h5ad"

        adata = sc.read_h5ad(adata_path)
        adata.uns['log1p'] = {'base': np.e}
        sc.pp.highly_variable_genes(adata, n_top_genes=n_highly_var_genes)
        adata = adata[:, adata.var['highly_variable']].copy()

        return adata