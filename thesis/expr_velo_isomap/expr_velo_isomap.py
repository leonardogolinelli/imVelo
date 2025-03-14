import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import sys
import os

# Add the parent directory to the system path
parent_directory = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(parent_directory)

from plotting import plot_velocity_expression


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

for dataset, cell_type_key in zip(datasets, cell_type_keys):
    for model_name in model_names:
        os.makedirs(f"plots/{dataset}/{model_name}/", exist_ok=True)
        flags = []
        for gene_name in dic[dataset]:
            flag = os.path.isfile(f"plots/{dataset}/{model_name}/{gene_name}_spliced.png")
            flags.append(flag)
        if all(flags):
            print("skipping")
            continue
        adata = sc.read_h5ad(f"../../benchmark/{model_name}/{dataset}/{model_name}_{dataset}.h5ad")
        for plot_type in ["unspliced", "spliced"]:
            for gene_name in dic[dataset]:
                if gene_name in list(adata.var_names) and not os.path.isfile(f"plots/{dataset}/{model_name}/{gene_name}_{plot_type}.png"):
                    plot_velocity_expression(adata, 
                                            scale_velocity=1,
                                            shift_expression=1,
                                            gene_name=gene_name, plot_type=plot_type, 
                                            normalize_y=False, normalize_velocity=False,
                                            cell_type_key=cell_type_key, 
                                            save_path=f"plots/{dataset}/{model_name}/{gene_name}_{plot_type}.png", 
                                            save_plot=True)
            else:
                print("skipping")