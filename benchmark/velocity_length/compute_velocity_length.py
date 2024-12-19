import numpy as np
import os

# Define models and datasets
models = ["imVelo", "lsvelo", "celldancer", "deepvelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "scvelo", "stochastic"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters", "celltype", "clusters"]

# Path to the output directory
output_dir = "../matrices/matrix_folder"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop over each dataset and model
for dataset, cell_type_key in zip(datasets, cell_type_keys):
    for model in models:
        print(f"Processing dataset: {dataset}, model: {model}")

        # Define paths for the velocity matrix and velocity length output
        velocity_matrix_path = os.path.join("../matrices/matrix_folder", f"velocity_{model}_{dataset}.npy")
        velocity_length_path = os.path.join(output_dir, f"velocity_length_{model}_{dataset}.npy")

        # Check if velocity length matrix already exists
        if os.path.exists(velocity_length_path):
            print(f"Velocity length matrix already exists for {model} - {dataset}. Skipping...")
            continue

        # Load the velocity matrix
        if not os.path.exists(velocity_matrix_path):
            print(f"Velocity matrix not found for {model} - {dataset}. Skipping...")
            continue
        velocity_matrix = np.load(velocity_matrix_path)

        # Calculate the length of each velocity vector and save
        velocity_length = np.linalg.norm(velocity_matrix, axis=1)
        np.save(velocity_length_path, velocity_length)
        print(f"Saved: {velocity_length_path}")
