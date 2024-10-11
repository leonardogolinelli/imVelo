import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scvelo_adapted_utils import load_files
from scvelo_adapted_metrics import compute_scvelo_metrics

adata_mivelo, model = load_files("pancreas", 30)
gnames = list(adata_mivelo.var_names)
adata = scv.datasets.pancreas()
adata = adata[:,gnames] #only keep gnames present in adata
scv.pp.filter_and_normalize(adata, min_shared_counts=20)
sc.pp.neighbors(adata)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
#sc.tl.umap(adata)
scv.tl.recover_dynamics(adata, n_jobs=-1)
scv.tl.velocity(adata, mode="dynamical")
scv.tl.velocity_graph(adata)
adata.write_h5ad("scvelo.h5ad")
compute_scvelo_metrics(adata, "pancreas", False, "clusters")
adata.write_h5ad("scvelo.h5ad")
adata.write_h5ad("../../outputs/final_anndatas/pancreas/velovi.h5ad")
