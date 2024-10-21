import pickle
import os
import matplotlib.pyplot as plt

# Load the existing gene-wise and cell-wise correlations
with open('gene_wise_correlations.pkl', 'rb') as f:
    gene_wise_correlations = pickle.load(f)

with open('cell_wise_correlations.pkl', 'rb') as f:
    cell_wise_correlations = pickle.load(f)

models = ["imVelo", "lsvelo", "celldancer", "deepvelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "scvelo", "stochastic"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]

# Function to plot and save the violin plots for "1 vs all"
def create_one_vs_all_plots(correlations, corr_type, dataset, model):
    # Prepare data for plotting
    all_correlations = []
    labels = []
    
    # Collect the correlations for the specific model compared to all others
    for other_model in models:
        if model != other_model:
            model_pair = tuple(sorted([model, other_model]))
            if model_pair in correlations[dataset]:
                corr = correlations[dataset][model_pair].flatten()  # Flatten the array for plotting
                all_correlations.append(corr)
                labels.append(f"{model} vs {other_model}")
    
    # If no correlations were found, skip the plotting
    if not all_correlations:
        print(f"No correlations found for {model} vs others in {dataset}. Skipping...")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.violinplot(all_correlations, showmeans=False, showmedians=True)
    
    # Set x-ticks and labels
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=90)
    
    # Add titles and labels
    ax.set_title(f'{corr_type.capitalize()} Correlations for {model} vs All in {dataset}')
    ax.set_xlabel('Model Comparisons')
    ax.set_ylabel(f'{corr_type.capitalize()} Correlations')
    
    # Save the plot as an image in a subfolder named after the model within the dataset folder
    folder = f"./{dataset}_correlation_plots/{model}_vs_all"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    plot_path = f"{folder}/{model}_vs_all_{corr_type}_correlations.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {corr_type} correlations plot for {model} vs all others in {dataset} at {plot_path}")

# Iterate through each dataset and create plots for both gene-wise and cell-wise correlations
for dataset in datasets:
    if dataset in gene_wise_correlations:
        print(f"Creating gene-wise correlation plots for dataset {dataset}...")
        for model in models:
            create_one_vs_all_plots(gene_wise_correlations, 'gene-wise', dataset, model)
        
    if dataset in cell_wise_correlations:
        print(f"Creating cell-wise correlation plots for dataset {dataset}...")
        for model in models:
            create_one_vs_all_plots(cell_wise_correlations, 'cell-wise', dataset, model)
