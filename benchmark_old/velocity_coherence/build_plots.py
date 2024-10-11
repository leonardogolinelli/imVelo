import pickle
import os
import scanpy as sc
from plot_functions import plot_violin_coherence, plot_violin_per_cell_type, plot_cdf_models, plot_cdf_per_cell_type

# Model and dataset information
models = ["celldancer", "imVelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "scvelo", "stochastic"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters", "celltype", "clusters"]

# Path to the coherence dictionary

#coherence_keys = ["normal", "absolute"]
#coherence_paths = ["coherences.pkl", "absolute_coherences.pkl"]

#coherence_keys = ["normal_normalized", "absolute_normalized"]
#coherence_paths = ["coherences_normalized.pkl", "absolute_coherences_normalized.pkl"]

coherence_keys = ["normal_cosine", "absolute_cosine"]
coherence_paths = ["coherences_cosine.pkl", "absolute_coherences_cosine.pkl"]

for key, path in zip(coherence_keys, coherence_paths):
    dic_path = path
    out_folder = f"plots_{key}"

    # Load the coherence dictionary
    try:
        with open(dic_path, "rb") as pickle_file:
            coherences = pickle.load(pickle_file)
        print("Coherences dictionary loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {dic_path} not found. Please ensure the coherence dictionary is generated and saved.")
        exit()

    # Function to build all plots from the coherence dictionary
    def build_all_plots(coherences, datasets, models, cell_type_keys, overwrite_plot=False, out_folder=out_folder):
        for dataset, cell_type_key in zip(datasets, cell_type_keys):
            print(f"Generating plots for dataset: {dataset}")

            # Define the paths for the violin and CDF plots for the dataset
            violin_plot_path = f"./{out_folder}/{dataset}/velocity_coherence_violin_{dataset}.png"
            cdf_plot_path = f"./{out_folder}/{dataset}/velocity_coherence_cdf_models_{dataset}.png"

            flag = not os.path.exists(violin_plot_path) and not os.path.exists(cdf_plot_path)
            print(flag)

            # Check if both plots exist before proceeding
            #if not os.path.exists(violin_plot_path) and not os.path.exists(cdf_plot_path) and not overwrite_plot:
            print(f"Both violin and CDF plots for dataset {dataset} already exist. Skipping.")
            # Skip processing for this dataset if both plots exist

            # 1. Violin plot comparing velocity coherences across different models
            plot_violin_coherence(coherences, dataset, models, overwrite_plot=overwrite_plot, out_folder=out_folder)

            # 2. Overlaid CDF plot comparing models for each dataset
            plot_cdf_models(coherences, dataset, models, overwrite_plot=overwrite_plot, out_folder=out_folder)

            # Process each model for the dataset
            for model in models:
                print(f"Generating plots for model: {model} in dataset: {dataset}")

                # Check if the model-specific plots exist before loading the AnnData file
                violin_path = f"./{out_folder}/{dataset}/{model}/velocity_coherence_violin_{dataset}_{model}.png"
                cdf_path = f"./{out_folder}/{dataset}/{model}/velocity_coherence_cdf_celltypes_{dataset}_{model}.png"

                # Skip loading if both plots already exist and overwrite_plot is False
                if (os.path.exists(violin_path) and os.path.exists(cdf_path)) and not overwrite_plot:
                    print(f"Both plots for model {model} in dataset {dataset} already exist. Skipping.")
                    continue

                # Load the AnnData object only if needed
                adata_path = f"../{model}/{dataset}/{model}_{dataset}.h5ad"
                if not os.path.exists(adata_path):
                    print(f"Warning: AnnData file not found for {model} in {dataset}. Skipping.")
                    continue

                adata = sc.read_h5ad(adata_path)
                adata.obs["velocity_coherence"] = coherences[dataset][model]

                # 3. Violin plot for each cell type within the dataset-model combination
                plot_violin_per_cell_type(adata, dataset, model, cell_type_key, overwrite_plot=overwrite_plot, out_folder=out_folder)

                # 4. CDF plot comparing cell types for the dataset-model combination
                plot_cdf_per_cell_type(adata, dataset, model, cell_type_key, overwrite_plot=overwrite_plot, out_folder=out_folder)

    # Build all the plots
    build_all_plots(coherences, datasets, models, cell_type_keys, overwrite_plot=False, out_folder=out_folder)
