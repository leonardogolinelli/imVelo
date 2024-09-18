import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import torch
from velovi import preprocess_data, VELOVI
import matplotlib.pyplot as plt
import seaborn as sns
import os
from velovi_adapted_plots import *
from velovi_adapted_metrics import *
from velovi_adapted_utils import *


datasets = ["pancreas", "gastrulation_erythroid", "forebrain", "dentategyrus_lamanno"]
cell_type_keys = ["clusters","celltype", "Clusters", "clusters"]
filter_genes = [True, False]
for filtered in filter_genes:
    for dataset, cell_type_key in zip(datasets, cell_type_keys):
        adata_path = os.path.expanduser(f"~/top_adatas/mivelo_{dataset}.h5ad")
        adata = sc.read_h5ad(adata_path)
        if dataset == "forebrain":
            adata.obs[cell_type_key] = [str(name) for name in adata.obs[cell_type_key]]
            adata.obs[cell_type_key] = pd.Series(adata.obs[cell_type_key], dtype="category")
        del adata.layers["velocity"]
        del adata.layers["velocity_u"]

        if filtered:
            print(f"filtering genes, initial number: {adata.shape[1]}")
            adata = preprocess_data(adata) #filter out poorly detected genes and min max scales data
            dataset += f"_filtered"
            print(f"filtering genes, final number: {adata.shape[1]}")
            print(f"new dataset name: {dataset}")
        VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
        vae = VELOVI(adata)
        vae.train(max_epochs=10, train_size=0.999)
        os.makedirs(f"{dataset}/stats", exist_ok=True)
        plot_training_stats(vae, dataset)
        adata = add_velovi_outputs_to_adata(adata, vae)
        sc.pp.neighbors(adata)
        scv.tl.velocity_graph(adata)
        save_files(adata, vae, dataset)
        plot_permutation_score(adata, vae, dataset, cell_type_key)
        compute_scvelo_metrics(adata, dataset, False, cell_type_key)
        save_files(adata, vae, dataset)
        deg_genes(adata, dataset, cell_type_key, n_deg_rows=1)
        save_files(adata, vae, dataset)
        plot_important_genes(adata, dataset, cell_type_key)
        adata = intrinsic_uncertainty(adata, vae, dataset, n_samples=100)
        save_files(adata, vae, dataset)
        adata = extrinsic_uncertainty(adata, vae, dataset, n_samples=25)
        save_files(adata, vae, dataset)
        adata.write_h5ad(os.path.expanduser(f"~/top_adatas/velovi_{dataset}.h5ad"))
        VELOVI.save(vae, os.path.expanduser(f"~/top_models/velovi_{dataset}.h5ad"))
        