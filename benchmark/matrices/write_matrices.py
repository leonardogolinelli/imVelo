import scanpy as sc
import numpy as np
import os

# Create the outputs directory if it doesn't exist
output_dir = "matrix_folder"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

models = ["imVelo", "lsvelo", "celldancer", "deepvelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "scvelo", "stochastic", "expimap"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters", "celltype", "clusters"]

for model in models:
    print(f"Processing model: {model}")
    
    for dataset, cell_type_key in zip(datasets, cell_type_keys):
        print(f"Processing dataset: {dataset}")

        # Define all the paths for this dataset-model combination in the outputs directory
        cell_types_path = os.path.join(output_dir, f"celltypes_{dataset}.npy")
        velocity_path = os.path.join(output_dir, f"velocity_{model}_{dataset}.npy")
        confidence_path = os.path.join(output_dir, f"confidence_{model}_{dataset}.npy")
        z_path = os.path.join(output_dir, f"z_{model}_{dataset}.npy")
        intrinsic_uncertainty_path = os.path.join(output_dir, f"intrinsic_uncertainty_{model}_{dataset}.npy")
        extrinsic_uncertainty_path = os.path.join(output_dir, f"extrinsic_uncertainty_{model}_{dataset}.npy")
        
        # Paths for storing var_names and obs_names
        var_names_path = os.path.join(output_dir, f"var_names_{model}_{dataset}.npy")
        obs_names_path = os.path.join(output_dir, f"obs_names_{model}_{dataset}.npy")

        # Define paths for Ms and Mu layers
        Ms_path = os.path.join(output_dir, f"Ms_{dataset}.npy")
        Mu_path = os.path.join(output_dir, f"Mu_{dataset}.npy")

        # Check which files are missing
        missing_files = []

        if not os.path.exists(cell_types_path):
            missing_files.append("celltypes")
        
        if model != "expimap":
            if not os.path.exists(velocity_path):
                missing_files.append("velocity")
            if not os.path.exists(confidence_path):
                missing_files.append("confidence")
        
        if model in ["imVelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered"]:
            if not os.path.exists(intrinsic_uncertainty_path):
                missing_files.append("intrinsic_uncertainty")
            if not os.path.exists(extrinsic_uncertainty_path):
                missing_files.append("extrinsic_uncertainty")
        
        if model in ["imVelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "expimap"]:
            if not os.path.exists(z_path):
                missing_files.append("z")
        
        # Check if var_names and obs_names are missing
        if not os.path.exists(var_names_path):
            missing_files.append("var_names")
        
        if not os.path.exists(obs_names_path):
            missing_files.append("obs_names")

        # Check if Ms and Mu layers are missing
        if not os.path.exists(Ms_path):
            missing_files.append("Ms")
        if not os.path.exists(Mu_path):
            missing_files.append("Mu")

        # If all necessary files exist, skip to the next dataset
        if not missing_files:
            print(f"All files already processed for {model} - {dataset}, skipping...")
            continue

        # Load the AnnData object only once if there are missing files
        print(f"Loading data for missing files: {missing_files}")
        adata_path = f"../{model}/{dataset}/{model}_{dataset}.h5ad"
        adata = sc.read_h5ad(adata_path)

        # Now process the missing files
        if "celltypes" in missing_files:
            celltypes = adata.obs[cell_type_key]
            np.save(cell_types_path, celltypes)

        if "velocity" in missing_files:
            velocity = adata.layers["velocity"]
            np.save(velocity_path, velocity)

        if "confidence" in missing_files:
            confidence = adata.obs["velocity_confidence"]
            np.save(confidence_path, confidence)

        if "intrinsic_uncertainty" in missing_files:
            intrinsic_uncertainty = adata.obs["directional_cosine_sim_variance"]
            np.save(intrinsic_uncertainty_path, intrinsic_uncertainty)

        if "extrinsic_uncertainty" in missing_files:
            extrinsic_uncertainty = adata.obs["directional_cosine_sim_variance_extrinisic"]
            np.save(extrinsic_uncertainty_path, extrinsic_uncertainty)

        if "z" in missing_files:
            z = adata.obsm["z"]
            np.save(z_path, z)

        # Save var_names if missing
        if "var_names" in missing_files:
            var_names = adata.var_names.astype(str)
            np.save(var_names_path, var_names)

        # Save obs_names if missing
        if "obs_names" in missing_files:
            obs_names = adata.obs_names.astype(str)
            np.save(obs_names_path, obs_names)

        # Save Ms layer if missing
        if "Ms" in missing_files and "Ms" in adata.layers:
            print(f"Saving Ms layer for dataset {dataset}")
            np.save(Ms_path, adata.layers["Ms"])

        # Save Mu layer if missing
        if "Mu" in missing_files and "Mu" in adata.layers:
            print(f"Saving Mu layer for dataset {dataset}")
            np.save(Mu_path, adata.layers["Mu"])

        # If Ms or Mu is missing in adata, print a message
        if "Ms" in missing_files and "Ms" not in adata.layers:
            print(f"Ms layer not found in dataset {dataset}")
        
        if "Mu" in missing_files and "Mu" not in adata.layers:
            print(f"Mu layer not found in dataset {dataset}")
