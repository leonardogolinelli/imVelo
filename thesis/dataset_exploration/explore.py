import matplotlib.pyplot as plt
import scanpy as sc
import sys
import os
from plot import compute_cell_pops, plot_cell_type_distribution

# Add the parent directory to the system path
parent_directory = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(parent_directory)

datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
#datasets = ["gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters", "celltype", "clusters"]
#cell_type_keys = ["celltype", "clusters"]
batch_keys = [None, None, "stage", "Age"]
#batch_keys = ["stage", "Age"]

for dataset, cell_type_key, batch_key in zip(datasets, cell_type_keys, batch_keys):
    print(f"processing dataset: {dataset}")
    model_name = "imVelo"
    K = 31

    adata = sc.read_h5ad(f"../../benchmark/{model_name}/{dataset}/{model_name}_{dataset}.h5ad")

    os.makedirs(f"plots/{dataset}/cell_pops/", exist_ok=True)
    os.makedirs(f"plots/{dataset}/components/", exist_ok=True)
    os.makedirs(f"plots/{dataset}/latent_space/", exist_ok=True)

    for dim_red in ["isomap", "pca"]:
        for i in range(1,4):
            sc.pl.umap(adata, color=f"{dim_red}_{i}", cmap="gnuplot", title=f"{dim_red}_{i}_{dataset}")
            plt.savefig(f"plots/{dataset}/components/{dim_red}_{i}.png")

    plot_cell_type_distribution(adata, cell_type_key, dataset, save_path=f"plots/{dataset}/cell_pops/ctype_distribution.png")

    #if dataset in ["gastrulation_erythroid", "dentategyrus_lamanno_P5"]:
    #    compute_cell_pops(adata, cell_type_key, batch_key, save_path=f"plots/{dataset}/cell_pops/cell_pops.png")
    