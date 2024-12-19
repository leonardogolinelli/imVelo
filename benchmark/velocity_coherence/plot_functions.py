import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot and save violin plots for different coherence metrics across models for each dataset
def plot_violin_coherence(coherences, dataset, models, metric_name, overwrite_plot=True, out_folder="plots"):
    output_folder = os.path.join(out_folder, dataset)
    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, f"{metric_name}_violin_{dataset}.png")
    
    if os.path.exists(plot_path) and not overwrite_plot:
        print(f"Plot already exists: {plot_path}. Skipping.")
        return
    
    data, labels = [], []
    for model in models:
        if model in coherences:
            data.append(coherences[model])
            labels.append(model)
    
    plt.figure(figsize=(10, 6))
    plt.violinplot(data, showmeans=True)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, fontsize=18)
    plt.title(f"{metric_name.replace('_', ' ').title()} for {dataset}", fontsize=25)
    plt.ylabel(f"{metric_name.replace('_', ' ').title()}", fontsize=18)

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Violin plot saved at: {plot_path}")

# Function to plot and save violin plots per cell type for each dataset-model combination
def plot_violin_per_cell_type(coherence_values, cell_types, dataset, model, metric_name, overwrite_plot=True, out_folder="plots"):
    output_folder = os.path.join(out_folder, dataset, model)
    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, f"{metric_name}_violin_{dataset}_{model}.png")
    
    if os.path.exists(plot_path) and not overwrite_plot:
        print(f"Plot already exists: {plot_path}. Skipping.")
        return

    df = pd.DataFrame({"coherence": coherence_values, "cell_type": cell_types})

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="cell_type", y="coherence")
    plt.title(f"{metric_name.replace('_', ' ').title()} Violin Plot for {dataset} - {model}", fontsize=25)
    plt.xticks(rotation=45, fontsize=18)

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Violin plot saved for dataset {dataset}, model {model} at: {plot_path}")

# Function to plot overlaid CDFs for each dataset across different models for each metric
def plot_cdf_models(coherences, dataset, models, metric_name, overwrite_plot=True, out_folder="plots"):
    output_folder = os.path.join(out_folder, dataset)
    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, f"{metric_name}_cdf_models_{dataset}.png")
    
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
    
    plt.title(f'Overlaid CDF of {metric_name.replace("_", " ").title()} Across Models in {dataset}', fontsize=25)
    plt.xlabel(f'{metric_name.replace("_", " ").title()}', fontsize=18)
    plt.ylabel('CDF', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"CDF plot saved for dataset {dataset} at: {plot_path}")
    
    

def plot_violin_across_datasets(models, datasets, metric, coherence_data, out_folder="plots", overwrite_plot=True):
    """
    Plots and saves a violin plot for each model across all datasets for a specified metric.
    Saves all plots in the main `plots/` directory without creating subfolders for each model.

    Parameters:
    - models: List of model names
    - datasets: List of dataset names
    - metric: The coherence metric to plot (e.g., 'coherence_fraction')
    - coherence_data: Dictionary containing preloaded coherence data
    - out_folder: Output folder for saved plots (default is './plots')
    - overwrite_plot: Whether to overwrite existing plots (default is True)
    """
    os.makedirs(out_folder, exist_ok=True)
    
    for model in models:
        coherence_data_model = []
        dataset_labels = []

        # Collect coherence values and dataset labels for each model across datasets
        for dataset in datasets:
            coherence_values = coherence_data.get(metric, {}).get(model, {}).get(dataset)
            if coherence_values is not None:
                coherence_data_model.extend(coherence_values)
                dataset_labels.extend([dataset] * len(coherence_values))
            else:
                print(f"No data for {metric} in {model} - {dataset}")

        # Plot the violin plot if there is data
        if coherence_data_model:
            data = pd.DataFrame({"Coherence": coherence_data_model, "Dataset": dataset_labels})
            plot_path = os.path.join(out_folder, f"{metric}_violin_across_datasets_{model}.png")
            
            if os.path.exists(plot_path) and not overwrite_plot:
                print(f"Plot already exists: {plot_path}. Skipping.")
                continue

            plt.figure(figsize=(10, 6))
            sns.violinplot(data=data, x="Dataset", y="Coherence")
            plt.title(f"{model} - {metric.replace('_', ' ').title()} across datasets")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            print(f"Violin plot saved at: {plot_path}")
        else:
            print(f"No coherence data available for model {model} across datasets for metric {metric}")





def plot_violin_across_models(models, dataset_name, metric, coherence_data, out_folder="plots", overwrite_plot=True):
    """
    Plots and saves a violin plot for a specified dataset across all models for a given metric.
    Uses preloaded `coherence_data` dictionary to avoid repeated file reads.

    Parameters:
    - models: List of model names
    - dataset_name: The dataset for which to plot the metric across models
    - metric: The coherence metric to plot (e.g., 'coherence_fraction')
    - coherence_data: Dictionary containing preloaded coherence data
    - out_folder: Output folder for saved plots (default is './plots')
    - overwrite_plot: Whether to overwrite existing plots (default is True)
    """
    coherence_data_dataset = []
    model_labels = []

    # Collect coherence values and model labels for each model across a dataset
    for model in models:
        coherence_values = coherence_data.get(metric, {}).get(model, {}).get(dataset_name)
        if coherence_values is not None:
            coherence_data_dataset.extend(coherence_values)
            model_labels.extend([model] * len(coherence_values))
        else:
            print(f"No data for {metric} in {model} - {dataset_name}")

    # Plot the violin plot if there is data
    if coherence_data_dataset:
        data = pd.DataFrame({"Coherence": coherence_data_dataset, "Model": model_labels})
        dataset_folder = os.path.join(out_folder, dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)
        plot_path = os.path.join(dataset_folder, f"{metric}_violin_across_models.png")
        
        if os.path.exists(plot_path) and not overwrite_plot:
            print(f"Plot already exists: {plot_path}. Skipping.")
            return

        plt.figure(figsize=(10, 6))
        sns.violinplot(data=data, x="Model", y="Coherence")
        plt.title(f"{metric.replace('_', ' ').title()} for {dataset_name} across models")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Violin plot saved at: {plot_path}")
    else:
        print(f"No coherence data available for {dataset_name} across models for metric {metric}")

def plot_cdf_per_cell_type(coherence_values, cell_types, dataset, model, metric_name, overwrite_plot=False, out_folder="plots"):
    """
    Plots and saves a CDF plot for each cell type within a dataset-model combination.

    Parameters:
    - coherence_values: Array of coherence values for each cell
    - cell_types: Array of cell type labels corresponding to coherence_values
    - dataset: Name of the dataset
    - model: Name of the model
    - metric_name: Name of the metric being plotted
    - overwrite_plot: Whether to overwrite existing plots (default is False)
    - out_folder: Output folder for saved plots (default is './plots')
    """
    output_folder = os.path.join(out_folder, dataset, model)
    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, f"{metric_name}_cdf_celltypes_{dataset}_{model}.png")
    
    if os.path.exists(plot_path) and not overwrite_plot:
        print(f"Plot already exists: {plot_path}. Skipping.")
        return

    # Prepare data for plotting CDF by cell type
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
    
    # Set font sizes and plot title
    plt.title(f'Empirical CDF of {metric_name.replace("_", " ").title()} per Cell Type for {dataset} - {model}', fontsize=25)
    plt.xlabel(f'{metric_name.replace("_", " ").title()}', fontsize=18)
    plt.ylabel('CDF', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=12, title='Cell Type')  # Adjust legend font size for readability
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"CDF plot saved for dataset {dataset}, model {model} at: {plot_path}")