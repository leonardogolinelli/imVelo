import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from baseline_adapted_metrics import compute_scvelo_metrics
from baseline_adapted_plots import plot_important_genes

adata = sc.read_h5ad("baseline.h5ad")
compute_scvelo_metrics(adata, "pancreas", False, "clusters")
plot_important_genes(adata, "pancreas", cell_type_key="clusters")
adata.write_h5ad("../../outputs/final_anndatas/pancreas/baseline.h5ad")