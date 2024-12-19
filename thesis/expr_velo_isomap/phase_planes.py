import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import scanpy as sc
from utils import return_gnames
parent_directory = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(parent_directory)

from plotting import plot_phase_plane, plot_velocity_expression



models = ["imVelo", "lsvelo", "celldancer", "deepvelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "scvelo", "stochastic"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]


dataset = "pancreas"
cell_type_key = "clusters"
model_name = "imVelo"


adata = sc.read_h5ad(f"../../benchmark/{model_name}/{dataset}/{model_name}_{dataset}.h5ad")


plot_type = "spliced"
gene_name = "Rbfox3"
plot_plane = False


if plot_plane:
    plot_phase_plane(adata, gene_name, u_scale=0.1, s_scale=0.1, 
        cell_type_key=cell_type_key, dataset=dataset, 
        K=11, save_path= f"gene_phase_planes/{gene_name}/{dataset}/{model_name}.png", 
        save_plot=False)

