import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
model_names = ["velovi", "velovi_filtered", "ivelo", "ivelo_filtered", "imVelo"]

for dataset in datasets:
    # Dictionary to store MSE values for each model
    mse_data = {}
    
    for model in model_names:
        file_path = f"../matrices/matrix_folder/test_mse_{model}_{dataset}.npy"
        
        if os.path.exists(file_path):
            # Load the test MSE data
            cell_mse = np.log(np.load(file_path))
            mse_data[model] = cell_mse
        else:
            print(f"MSE file for {model} in {dataset} not found.")
            continue

    # Create the violin plot for the dataset
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=[mse_data[model] for model in model_names if model in mse_data], 
                   palette="Set2", 
                   inner="box")
    
    # Set plot labels and title
    plt.xticks(ticks=range(len(mse_data)), labels=mse_data.keys())
    plt.title(f"Test Cell Log MSE Distribution for {dataset}")
    plt.ylabel("MSE")
    plt.xlabel("Model")

    os.makedirs("plots", exist_ok=True)
    # Show and save the plot
    plt.savefig(f"plots/test_mse_violin_{dataset}.png")
    plt.show()
