import matplotlib.pyplot as plt
import scanpy as sc
import sys
import os

# Add the parent directory to the system path
parent_directory = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(parent_directory)

model_names = ["imVelo", "ivelo", "ivelo_filtered", "expimap"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters", "celltype", "clusters"]

progress_file = "progress_dataset.txt"

# Check where to start
if os.path.exists(progress_file):
    with open(progress_file, "r") as f:
        start_model, start_dataset = map(int, f.read().strip().split(","))
else:
    start_model, start_dataset = 0, 0

for i, model_name in enumerate(model_names[start_model:], start=start_model):
    for j, (dataset, cell_type_key) in enumerate(zip(datasets[start_dataset:], cell_type_keys[start_dataset:]), start=start_dataset):
        print(f"Processing dataset: {dataset} for model: {model_name}")
        K = 31

        # Read AnnData
        adata = sc.read_h5ad(f"../../benchmark/{model_name}/{dataset}/{model_name}_{dataset}.h5ad")

        # Create output directory
        os.makedirs(f"plots/{dataset}/latent_space/", exist_ok=True)

        # Process and save UMAP plot
        sc.pp.neighbors(adata, use_rep="z")
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=cell_type_key)
        plt.savefig(f"plots/{dataset}/latent_space/{model_name}.png", bbox_inches="tight")
        plt.close()  # Close the plot to free memory

        # Save progress and restart the script for the next dataset
        with open(progress_file, "w") as f:
            f.write(f"{i},{j + 1 if j + 1 < len(datasets) else 0}")

        print(f"Restarting script for the next dataset.")
        os.execv(sys.executable, [sys.executable] + sys.argv)

    # Reset dataset index when moving to the next model
    start_dataset = 0

# Clean up progress file when done
if os.path.exists(progress_file):
    os.remove(progress_file)
