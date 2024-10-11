import scanpy as sc

def preprocess(
    dataset_name
    ):
        if dataset_name == "forebrain":
            adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/benchmark/imVelo/forebrain/forebrain/K11/adata/adata_K11_dt_ve.h5ad"
        elif dataset_name == "pancreas":
            adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/benchmark/imVelo/pancreas/pancreas/K11/adata/adata_K11_dt_ve.h5ad"
        elif dataset_name == "gastrulation_erythroid":
            adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/benchmark/imVelo/gastrulation_erythroid/gastrulation_erythroid/K11/adata/adata_K11_dt_ve.h5ad"
        elif dataset_name == "dentategyrus_lamanno_P5":
            adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/benchmark/imVelo/dentategyrus_lamanno_P5/dentategyrus_lamanno_P5/K31/adata/adata_K31_dt_pca.h5ad"

        adata = sc.read_h5ad(adata_path)

        print(f"number of gene programs: {len(adata.uns['terms'])}")

        return adata