import scvelo as scv
import scanpy as sc
from scvelo_adapted_metrics import compute_scvelo_metrics

datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters","celltype", "clusters"]

for dataset, cell_type_key in zip(datasets, cell_type_keys):
    adata_path = f"../imVelo/{dataset}/imVelo_{dataset}.h5ad"
    adata = sc.read_h5ad(adata_path)
    scv.tl.velocity(adata, mode="stochastic")
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding_stream(adata, color=cell_type_key)
    compute_scvelo_metrics(adata, dataset, False, cell_type_key)
    #deg_genes(adata, dataset, cell_type_key=cell_type_key, n_deg_rows=5)
    adata.write_h5ad(f"{dataset}/stochastic_{dataset}.h5ad")
