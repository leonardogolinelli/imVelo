import scanpy as sc
import scvelo as scv
import pandas as pd
import seaborn as sns

def preprocess(
    dataset_name,
    cell_type_key,
    n_highly_var_genes,
    smooth_k,
    ):
        if dataset_name == "forebrain":
            adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/benchmark/imVelo/imVelo_forebrain/forebrain/K11/adata/adata_K11_dt_ve.h5ad"
        elif dataset_name == "pancreas":
            adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_pancreas_K11_knn_rep_ve_best_key_None_0_kl_weight_1e-9_1e-8/pancreas/K11/adata/adata_K11_dt_ve.h5ad"
        elif dataset_name == "gastrulation_erythroid":
            adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_gastrulation_erythroid_K11_knn_rep_ve_best_key_None_0_kl_weight_1e-9_1e-08_20k_256/gastrulation_erythroid/K11/adata/adata_K11_dt_ve.h5ad"
        elif dataset_name == "dentategyrus_lamanno_P5":
            adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/outputs_dentategyrus_lamanno_P5_K31_knn_rep_pca_best_key_pca_unique_0_kl_weight_1e-9_1e-08_20k_12_july/dentategyrus_lamanno_P5/K31/adata/adata_K31_dt_pca.h5ad"

        adata = sc.read_h5ad(adata_path)
        adata.layers["unspliced"] = adata.layers["counts_unspliced"].copy()
        adata.layers["spliced"] = adata.layers["counts_spliced"].copy()
        adata.X = adata.layers["counts_spliced"].copy()
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=n_highly_var_genes) #filter and normalize
        scv.pp.moments(adata, n_neighbors=smooth_k)

        adata.obs[cell_type_key] = [str(cat) for cat in list(adata.obs[cell_type_key])]
        adata.obs[cell_type_key] = pd.Series(adata.obs[cell_type_key], dtype="category")
        unique_categories = adata.obs[cell_type_key].cat.categories
        rgb_colors = sns.color_palette("tab20", len(unique_categories))
        hex_colors = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b in rgb_colors]
        adata.uns[f"{cell_type_key}_colors"] = hex_colors

        return adata