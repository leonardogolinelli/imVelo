import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define models and datasets
models = ["imVelo", "lsvelo", "celldancer", "deepvelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "scvelo", "stochastic"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
matrix_folder = "../matrices/matrix_folder"

# Metrics to plot
metrics = ["confidence", "intrinsic_uncertainty", "extrinsic_uncertainty"]

# Create output directories for each metric
output_dirs = {metric: f"./plots/{metric}_violin_plots" for metric in metrics}
for output_dir in output_dirs.values():
    os.makedirs(output_dir, exist_ok=True)

# Function to generate and save violin plots for a specific metric
def generate_violin_plots(dataset, metric):
    confidences = []
    model_labels = []

    for model in models:
        # Build the filename
        file_name = f"{metric}_{model}_{dataset}.npy"
        file_path = os.path.join(matrix_folder, file_name)
        
        if os.path.exists(file_path):
            # Load the values
            metric_values = np.load(file_path, allow_pickle=True)
            
            # Check if the loaded object is a NumPy array
            if isinstance(metric_values, np.ndarray):
                if metric_values.size > 0:
                    confidences.append(metric_values)
                    model_labels.append(model)
            elif isinstance(metric_values, np.lib.npyio.NpzFile):
                # If it's an npz file, extract arrays
                for key in metric_values.files:
                    data = metric_values[key]
                    if isinstance(data, np.ndarray) and data.size > 0:
                        confidences.append(data)
                        model_labels.append(f"{model}_{key}")

    # Check if there are any valid metric values to plot
    if len(confidences) > 0:
        # Create violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=confidences)
        plt.xticks(ticks=range(len(model_labels)), labels=model_labels, rotation=45)
        plt.title(f"{metric.capitalize()} Violin Plots for {dataset.capitalize()}")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Models")

        # Save plot
        output_file = os.path.join(output_dirs[metric], f"{dataset}_{metric}_violin_plot.png")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

# Generate plots for each dataset and each metric
for dataset in datasets:
    for metric in metrics:
        generate_violin_plots(dataset, metric)
