import numpy as np
import pandas as pd
import scvelo as scv
import scanpy as sc
from scvelo_adapted_utils import load_files
from scvelo_adapted_metrics import compute_scvelo_metrics
from scvelo_adapted_plots import plot_important_genes


#adata_path = "../../outputs/final_anndatas/pancreas/miVelo.h5ad"
#adata = sc.read_h5ad(adata_path)
adata = sc.read_h5ad("../../outputs/final_anndatas/pancreas/scvelo.h5ad")
sc.pp.neighbors(adata)
scv.tl.recover_dynamics(adata, n_jobs=-1)
scv.tl.velocity(adata, mode="dynamical")
scv.tl.velocity_graph(adata)
scv.pl.velocity_embedding_stream(adata, color="clusters")
#adata.write_h5ad("adata.h5ad")
#compute_scvelo_metrics(adata, "pancreas", False, "clusters")
plot_important_genes(adata, "pancreas", cell_type_key="clusters")
adata.write_h5ad("../../outputs/final_anndatas/pancreas/scvelo.h5ad")

