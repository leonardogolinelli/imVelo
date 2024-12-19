import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plot_functions import (
    plot_violin_across_datasets,
    plot_violin_across_models,
    plot_violin_coherence,
    plot_violin_per_cell_type,
    plot_cdf_models,
    plot_cdf_per_cell_type
)

# Define models, datasets, and path
#models = ["imVelo", "lsvelo", "celldancer", "deepvelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "scvelo", "stochastic"]
models = ["imVelo", "lsvelo", "celldancer", "deepvelo", "ivelo", "velovi", "stochastic"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
matrix_folder = "../matrices/matrix_folder"
metrics = ["coherence_fraction"]

# Load all metric data once and store in a dictionary
coherence_data = {}
for metric in metrics:
    coherence_data[metric] = {}
    for model in models:
        coherence_data[metric][model] = {}
        for dataset in datasets:
            file_path = os.path.join(matrix_folder, f"{metric}_{model}_{dataset}.npy")
            if os.path.exists(file_path):
                coherence_data[metric][model][dataset] = np.load(file_path)
            else:
                print(f"File not found: {file_path}")
                coherence_data[metric][model][dataset] = None

# Function to load cell types for each dataset
def load_cell_types(dataset):
    celltypes_path = os.path.join(matrix_folder, f"celltypes_{dataset}.npy")
    if os.path.exists(celltypes_path):
        return np.load(celltypes_path, allow_pickle=True)
    else:
        print(f"Cell type file not found for dataset {dataset}")
        return None
"""
# Generate plots for each dataset
for dataset in datasets:
    print(f"Processing dataset: {dataset}")
    cell_types = load_cell_types(dataset)

    for metric in metrics:
        print(f"Processing metric: {metric}")
        coherences_all_models = {
            model: coherence_data[metric][model][dataset] 
            for model in models if coherence_data[metric][model][dataset] is not None
        }
        
        # Plot all models together if data exists
        if coherences_all_models:
            plot_violin_coherence(coherences_all_models, dataset, models, metric_name=metric)
            plot_cdf_models(coherences_all_models, dataset, models, metric_name=metric)
        
        # Plot per cell type for each model
        for model in models:
            coherence_values = coherence_data[metric][model][dataset]
            if coherence_values is not None and cell_types is not None:
                plot_violin_per_cell_type(coherence_values, cell_types, dataset, model, metric_name=metric)
                plot_cdf_per_cell_type(coherence_values, cell_types, dataset, model, metric_name=metric)
"""
# Generate across-dataset violin plots for each model using preloaded data
for metric in metrics:
    print(f"Processing metric across all datasets: {metric}")
    plot_violin_across_datasets(models, datasets, metric, coherence_data, out_folder="plots")

# Generate across-model violin plots for each dataset using preloaded data
for metric in metrics:
    for dataset in datasets:
        print(f"Processing {dataset} across all models for metric: {metric}")
        plot_violin_across_models(models, dataset, metric, coherence_data, out_folder="plots")
