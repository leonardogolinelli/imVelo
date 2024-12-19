import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_functions import plot_violin_across_datasets, plot_violin_coherence, plot_violin_per_cell_type, plot_cdf_models, plot_cdf_per_cell_type


# Define the models and datasets
models = ["imVelo", "lsvelo", "celldancer", "deepvelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "scvelo", "stochastic"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
#datasets = ["dentategyrus_lamanno_P5"]
# Path to the matrix folder containing stored coherence matrices and cell types
matrix_folder = "../matrices/matrix_folder"

# List of coherence metrics
metrics = [
    "velocity_length"
]

# Function to load a stored metric matrix
def load_metric_matrix(metric_name, model, dataset):
    file_path = os.path.join(matrix_folder, f"{metric_name}_{model}_{dataset}.npy")
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        print(f"File not found: {file_path}")
        return None

# Function to load cell types for a dataset
def load_cell_types(dataset):
    celltypes_path = os.path.join(matrix_folder, f"celltypes_{dataset}.npy")
    if os.path.exists(celltypes_path):
        return np.load(celltypes_path, allow_pickle=True)
    else:
        print(f"Cell type file not found for dataset {dataset}")
        return None
    
# Function to plot violin plots for each model across all datasets for a specific metric


# Loop over each dataset and model to generate plots for each metric
for dataset in datasets:
    print(f"Processing dataset: {dataset}")

    # Load cell types for the dataset
    cell_types = load_cell_types(dataset)

    for metric in metrics:
        print(f"Processing metric: {metric}")

        # 1. Plot coherence for all models (ignoring cell types)
        coherences_all_models = {}
        for model in models:
            coherence_values = load_metric_matrix(metric, model, dataset)
            if coherence_values is not None:
                coherences_all_models[model] = coherence_values
        
        # Plot violin and CDF for all models together (ignoring cell types)
        if coherences_all_models:
            plot_violin_coherence(coherences_all_models, dataset, models, metric_name=metric)
            #plot_cdf_models(coherences_all_models, dataset, models, metric_name=metric)
        else:
            print(f"No coherence data found for {metric} in {dataset}. Skipping...")

        # 2. Plot coherence per cell type for each model separately
        for model in models:
            coherence_values = load_metric_matrix(metric, model, dataset)
            if coherence_values is not None and cell_types is not None:
                # Plot violin and CDF per cell type
                plot_violin_per_cell_type(coherence_values, cell_types, dataset, model, metric_name=metric)
                #plot_cdf_per_cell_type(coherence_values, cell_types, dataset, model, metric_name=metric)
            else:
                print(f"Skipping model {model} for {metric} in {dataset} due to missing data.")

# Loop through each metric and generate plots for each model across all datasets
for metric in metrics:
    print(f"Processing metric: {metric}")
    plot_violin_across_datasets(models, datasets, metric, matrix_folder)