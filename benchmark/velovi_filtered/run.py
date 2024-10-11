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

datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters","celltype", "clusters"]

for dataset, cell_type_key in zip(datasets, cell_type_keys):
    if not os.path.isdir(dataset):
        adata_path = f"../imVelo/{dataset}/imVelo_{dataset}.h5ad"
        adata = sc.read_h5ad(adata_path)
        del adata.layers["velocity"]
        del adata.layers["velocity_u"]
        print(f"filtering genes, initial number: {adata.shape[1]}")
        adata = preprocess_data(adata) #filter out poorly detected genes and min max scales data
        print(f"filtering genes, final number: {adata.shape[1]}")
        print(f"new dataset name: {dataset}")
        VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
        vae = VELOVI(adata)
        vae.train(max_epochs=1000, train_size=0.90)
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
