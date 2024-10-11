import scanpy as sc
from velovi_adapted_metrics import compute_scvelo_metrics
from velovi_adapted_plots import plot_important_genes


adata = sc.read_h5ad("velovi_proc.h5ad")
compute_scvelo_metrics(adata, "pancreas", False, cell_type_key="clusters")
plot_important_genes(adata, "pancreas", cell_type_key="clusters")
adata.write_h5ad("../../outputs/final_anndatas/pancreas/velovi_proc.h5ad")

