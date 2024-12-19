import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import sys
import os

# Add the parent directory to the system path
parent_directory = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(parent_directory)

from plotting import plot_phase_plane, plot_velocity_expression
from utils import return_gnames

dic = {
    "forebrain" : ["Gnas"],
    "gastrulation_erythroid" : ["Rap1b", "Hba-x"],
    "pancreas" : ["Gnas",
                "Gnao1"
                "Ank3",
                "Cald1",
                "Hsp90b1",
                "Pex5l",
                "Snrnp70", #spliced
                "Stx16", #spliced,
                "Cck",
                "Spp1"],
    "dentategyrus_lamanno_P5" : ["Ryr2"]
}



datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters", "celltype", "clusters"]
model_names = ["imVelo", "lsvelo", "celldancer", "deepvelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "scvelo"]


datasets = ["dentategyrus_lamanno_P5"]
model_names =  ["velovi", "velovi_filtered", "scvelo"]
cell_type_keys = ["clusters"]


"""
for dataset, cell_type_key in zip(datasets, cell_type_keys):
    for model_name in model_names:
        os.makedirs(f"plots/{dataset}/{model_name}/", exist_ok=True)
        flags = []
        for gene_name in dic[dataset]:
            flag = os.path.isfile(f"plots/{dataset}/{model_name}/{gene_name}.png")
            flags.append(flag)
        if all(flags):
            print("skipping")
            continue
        adata = sc.read_h5ad(f"../../benchmark/{model_name}/{dataset}/{model_name}_{dataset}.h5ad")
        for gene_name in dic[dataset]:
            if gene_name in list(adata.var_names) and not os.path.isfile(f"plots/{dataset}/{model_name}/{gene_name}.png"):
                plot_phase_plane(adata, gene_name, u_scale=0.1, s_scale=0.1, 
                    cell_type_key=cell_type_key, dataset=dataset, 
                    K=11, save_path= f"plots/{dataset}/{model_name}/{gene_name}.png", 
                    save_plot=True)
            else:
                print("skipping")
"""

for dataset, cell_type_key in zip(datasets, cell_type_keys):
    for model_name in model_names:
        plot_dir = f"plots/{dataset}/{model_name}/"
        os.makedirs(plot_dir, exist_ok=True)
        
        h5ad_file = f"../../benchmark/{model_name}/{dataset}/{model_name}_{dataset}.h5ad"
        
        # Load only var_names without loading the full dataset
        adata_backed = sc.read_h5ad(h5ad_file, backed='r')
        var_names = adata_backed.var_names
        gene_names_in_dataset = set(var_names)
        
        # Filter gene names to those present in the dataset
        gene_names = [gene_name for gene_name in return_gnames() if gene_name in gene_names_in_dataset]
        
        # Generate list of expected plot paths for genes in the dataset
        plot_paths = [f"{plot_dir}/{gene_name}.png" for gene_name in gene_names]
        
        # Check if all plot files exist
        plots_exist = all(os.path.isfile(plot_path) for plot_path in plot_paths)
        
        if plots_exist:
            print(f"All plots for {dataset} - {model_name} are already generated. Skipping dataset loading.")
            continue
        
        # Load the full dataset if any plot is missing
        adata = sc.read_h5ad(h5ad_file)
        
        for gene_name in return_gnames():
            plot_path = f"{plot_dir}/{gene_name}.png"
            if gene_name in adata.var_names and not os.path.isfile(plot_path):
                print(f"Working on gene: {gene_name}")
                plot_phase_plane(
                    adata, gene_name, u_scale=0.1, s_scale=0.1, 
                    cell_type_key=cell_type_key, dataset=dataset, 
                    K=11, save_path=plot_path, save_plot=True
                )
            else:
                print(f"Skipping {gene_name} as the plot already exists or gene not in dataset.")
        
        # Clean up the backed dataset to free resources
        adata_backed.file.close()
