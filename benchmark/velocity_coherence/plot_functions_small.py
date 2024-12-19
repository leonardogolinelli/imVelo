import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import scanpy as sc

# Function to plot and save violin plots for different coherence metrics across models for each dataset
def plot_violin_coherence(coherences, dataset, models, metric_name, overwrite_plot=True, out_folder="plots"):
    output_folder = f"./{out_folder}/{dataset}"
    os.makedirs(output_folder, exist_ok=True)
    plot_path = f"{output_folder}/{metric_name}_violin_{dataset}.png"
    
    # Check if the plot exists and skip if overwrite_plot is False
    if os.path.exists(plot_path) and not overwrite_plot:
        print(f"Plot already exists: {plot_path}. Skipping.")
        return
    
    data = []
    labels = []
    for model in models:
        if model in coherences:
            data.append(coherences[model])
            labels.append(model)
    
    plt.figure(figsize=(10, 6))
    plt.violinplot(data, showmeans=True)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
    plt.title(f"{metric_name.replace('_', ' ').title()} for {dataset}")
    plt.ylabel(f"{metric_name.replace('_', ' ').title()}")

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Violin plot saved at: {plot_path}")


# Function to plot and save violin plots per cell type for each dataset-model combination
def plot_violin_per_cell_type(coherence_values, cell_types, dataset, model, metric_name, overwrite_plot=True, out_folder="plots"):
    output_folder = f"./{out_folder}/{dataset}/{model}"
    os.makedirs(output_folder, exist_ok=True)
    plot_path = f"{output_folder}/{metric_name}_violin_{dataset}_{model}.png"
    
    # Check if the plot exists and skip if overwrite_plot is False
    if os.path.exists(plot_path) and not overwrite_plot:
        print(f"Plot already exists: {plot_path}. Skipping.")
        return

    # Create a DataFrame for plotting
    df = pd.DataFrame({
        "coherence": coherence_values,
        "cell_type": cell_types
    })

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="cell_type", y="coherence")
    plt.title(f"{metric_name.replace('_', ' ').title()} Violin Plot for {dataset} - {model}")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Violin plot saved for dataset {dataset}, model {model} at: {plot_path}")

# Function to plot overlaid CDFs for each dataset across different models for each metric
def plot_cdf_models(coherences, dataset, models, metric_name, overwrite_plot=True, out_folder="plots"):
    output_folder = f"./{out_folder}/{dataset}"
    os.makedirs(output_folder, exist_ok=True)
    plot_path = f"{output_folder}/{metric_name}_cdf_models_{dataset}.png"
    
    # Check if the plot exists and skip if overwrite_plot is False
    if os.path.exists(plot_path) and not overwrite_plot:
        print(f"Plot already exists: {plot_path}. Skipping.")
        return

    df_list = []
    for model in models:
        if model in coherences:
            coherence_values = coherences[model]
            sorted_data = np.sort(coherence_values)
            cdf = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
            
            df = pd.DataFrame({
                f'{metric_name.replace("_", " ").title()}': sorted_data,
                'CDF': cdf,
                'Model': model
            })
            df_list.append(df)
    
    coherence_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=coherence_df, x=f'{metric_name.replace("_", " ").title()}', y='CDF', hue='Model')
    
    plt.title(f'Overlaid CDF of {metric_name.replace("_", " ").title()} Across Models in {dataset}')
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"CDF plot saved for dataset {dataset} at: {plot_path}")


# Function to plot the CDF per cell type for a given dataset-model combination
def plot_cdf_per_cell_type(coherence_values, cell_types, dataset, model, metric_name, overwrite_plot=False, out_folder="plots"):
    output_folder = f"./{out_folder}/{dataset}/{model}"
    os.makedirs(output_folder, exist_ok=True)
    plot_path = f"{output_folder}/{metric_name}_cdf_celltypes_{dataset}_{model}.png"
    
    # Check if the plot exists and skip if overwrite_plot is False
    if os.path.exists(plot_path) and not overwrite_plot:
        print(f"Plot already exists: {plot_path}. Skipping.")
        return

    df_list = []
    unique_cell_types = np.unique(cell_types)

    for cell_type in unique_cell_types:
        coherence_subset = coherence_values[cell_types == cell_type]
        sorted_data = np.sort(coherence_subset)
        cdf = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        
        df = pd.DataFrame({
            f'{metric_name.replace("_", " ").title()}': sorted_data,
            'CDF': cdf,
            'Cell Type': cell_type
        })
        df_list.append(df)
    
    coherence_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=coherence_df, x=f'{metric_name.replace("_", " ").title()}', y='CDF', hue='Cell Type')
    
    plt.title(f'Empirical CDF of {metric_name.replace("_", " ").title()} per Cell Type for {dataset} - {model}')
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"CDF plot saved for dataset {dataset}, model {model} at: {plot_path}")