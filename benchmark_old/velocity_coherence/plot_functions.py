import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import scanpy as sc

# Function to plot and save violin plots for velocity coherence across models for each dataset
def plot_violin_coherence(coherences, dataset, models, overwrite_plot=True, out_folder = "plots"):
    output_folder = f"./{out_folder}/{dataset}"
    os.makedirs(output_folder, exist_ok=True)
    plot_path = f"{output_folder}/velocity_coherence_violin_{dataset}.png"
    
    # Check if the plot exists and skip if overwrite_plot is False
    if os.path.exists(plot_path) and not overwrite_plot:
        print(f"Plot already exists: {plot_path}. Skipping.")
        return
    
    data = []
    labels = []
    for model in models:
        if model in coherences[dataset]:
            data.append(coherences[dataset][model])
            labels.append(model)
    
    plt.figure(figsize=(10, 6))
    plt.violinplot(data, showmeans=True)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
    plt.title(f"Velocity Coherence for {dataset}")
    plt.ylabel("Velocity Coherence")

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Violin plot saved at: {plot_path}")


# Function to plot and save violin plots per cell type for each dataset-model combination
def plot_violin_per_cell_type(adata, dataset, model, cell_type_key, overwrite_plot=True, out_folder = "plots"):
    output_folder = f"./{out_folder}/{dataset}/{model}"
    os.makedirs(output_folder, exist_ok=True)
    plot_path = f"{output_folder}/velocity_coherence_violin_{dataset}_{model}.png"
    
    # Check if the plot exists and skip if overwrite_plot is False
    if os.path.exists(plot_path) and not overwrite_plot:
        print(f"Plot already exists: {plot_path}. Skipping.")
        return

    plt.figure(figsize=(10, 6))
    sc.pl.violin(adata, keys='velocity_coherence', groupby=cell_type_key, rotation=45)
    
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Violin plot saved for dataset {dataset}, model {model} at: {plot_path}")

# Function to plot overlaid CDFs for each dataset across different models
def plot_cdf_models(coherences, dataset, models, overwrite_plot=True, out_folder = "plots"):
    output_folder = f"./{out_folder}/{dataset}"
    os.makedirs(output_folder, exist_ok=True)
    plot_path = f"{output_folder}/velocity_coherence_cdf_models_{dataset}.png"
    
    # Check if the plot exists and skip if overwrite_plot is False
    if os.path.exists(plot_path) and not overwrite_plot:
        print(f"Plot already exists: {plot_path}. Skipping.")
        return

    df_list = []
    for model in models:
        if model in coherences[dataset]:
            coherence_values = coherences[dataset][model]
            sorted_data = np.sort(coherence_values)
            cdf = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
            
            df = pd.DataFrame({
                'Velocity coherence': sorted_data,
                'CDF': cdf,
                'Model': model
            })
            df_list.append(df)
    
    coherence_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=coherence_df, x='Velocity coherence', y='CDF', hue='Model')
    
    plt.title(f'Overlaid CDF of Velocity Coherence Across Models in {dataset}')
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"CDF plot saved for dataset {dataset} at: {plot_path}")

# Function to plot the CDF per cell type for a given dataset-model combination
def plot_cdf_per_cell_type(adata, dataset, model, cell_type_key, overwrite_plot=False, out_folder = "plots"):
    output_folder = f"./{out_folder}/{dataset}/{model}"
    os.makedirs(output_folder, exist_ok=True)
    plot_path = f"{output_folder}/velocity_coherence_cdf_celltypes_{dataset}_{model}.png"
    
    # Check if the plot exists and skip if overwrite_plot is False
    if os.path.exists(plot_path) and not overwrite_plot:
        print(f"Plot already exists: {plot_path}. Skipping.")
        return

    df_list = []
    cell_types = adata.obs[cell_type_key].unique()

    for cell_type in cell_types:
        cells = adata.obs[adata.obs[cell_type_key] == cell_type].index
        coherence_values = adata[cells].obs['velocity_coherence']
        sorted_data = np.sort(coherence_values)
        cdf = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        
        df = pd.DataFrame({
            'Velocity coherence': sorted_data,
            'CDF': cdf,
            'Cell type': cell_type
        })
        df_list.append(df)
    
    coherence_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=coherence_df, x='Velocity coherence', y='CDF', hue='Cell type')
    
    plt.title(f'Empirical CDF of Velocity Coherence per Cell Type for {dataset} - {model}')
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"CDF plot saved for dataset {dataset}, model {model} at: {plot_path}")
