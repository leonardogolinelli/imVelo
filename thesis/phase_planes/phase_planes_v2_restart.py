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

datasets = ["gastrulation_erythroid"]
cell_type_keys = ["celltype"]
#model_names = ["imVelo", "deepvelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "scvelo", "stochastic"]
model_names = ["celldancer"]
gene_names_gastrulation = [
    "Hspa8",
    "Hspa5",
    "Hsp90b1",
    "Actb",
    "Rap1b",
    "Rfc2",
    "Rpl18a",
    "Slc16a10",
    "Rap1b",
    "Hba-x",
    "Hba-a2",
    "Rpl18",
    "Psmc1"

]
# Track progress using a temporary file
progress_file = "progress.txt"

# Check where to start
if os.path.exists(progress_file):
    with open(progress_file, "r") as f:
        start_index = int(f.read().strip())
else:
    start_index = 0

for i, model_name in enumerate(model_names[start_index:], start=start_index):
    print(f"model name: {model_name}")
    
    for dataset, cell_type_key in zip(datasets, cell_type_keys):
        print(f"dataset: {dataset}")
        adata = sc.read_h5ad(f"../../benchmark/{model_name}/{dataset}/{model_name}_{dataset}.h5ad")
        
        for gene_name in gene_names_gastrulation:
            print(f"gene name: {gene_name}")
            os.makedirs(f"plots/{dataset}/{gene_name}", exist_ok=True)
            if gene_name in list(adata.var_names):
                plot_phase_plane(
                    adata, gene_name, u_scale=0.1, s_scale=0.1, 
                    cell_type_key=cell_type_key, dataset=dataset, 
                    K=11, save_path=f"plots/{dataset}/{gene_name}/{model_name}.png", 
                    save_plot=True
                )
            else:
                print(f"skippin'")
    
    # Save progress and restart the script
    with open(progress_file, "w") as f:
        f.write(str(i + 1))
    
    print("Restarting script to clear memory...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Clean up progress file when done
if os.path.exists(progress_file):
    os.remove(progress_file)
