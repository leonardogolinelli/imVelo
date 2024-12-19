import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import sys
import os

# Add the parent directory to the system path
parent_directory = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(parent_directory)
from utils import return_gnames

from plotting import plot_phase_plane

datasets = ["forebrain", "pancreas", "gastrulation_erythroid"]
cell_type_keys = ["Clusters", "clusters", "celltype"]
model_names = ["lsvelo", "celldancer", "deepvelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "scvelo", "stochastic"]

datasets = ["dentategyrus_lamanno_P5"]
cell_type_keys = ["clusters"]

for model_name in model_names:
    print(f"model name: {model_name}")
    for dataset, cell_type_key in zip(datasets, cell_type_keys):
        print(f"dataset: {dataset}")
        adata = sc.read_h5ad(f"../../benchmark/{model_name}/{dataset}/{model_name}_{dataset}.h5ad")
        for gene_name in return_gnames():
            print(f"gene name: {gene_name}")
            os.makedirs(f"plots/{dataset}/{gene_name}", exist_ok=True)
            if gene_name in list(adata.var_names):
                plot_phase_plane(adata, gene_name, u_scale=0.1, s_scale=0.1, 
                                cell_type_key=cell_type_key, dataset=dataset, 
                                K=11, save_path= f"plots/{dataset}/{gene_name}/{model_name}.png", 
                                save_plot=True)
            else:
                print(f"skippin'")