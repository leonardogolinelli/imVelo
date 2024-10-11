import scanpy as sc
import pickle
import os

models = ["celldancer", "imVelo", "ivelo", "velovi", "scvelo", "stochastic"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
K_vals = [11, 11, 11, 31]

#models = ["scvelo", "stochastic"]
#datasets = ["dentategyrus_lamanno_P5"]
#K_vals = [11, 11, 11, 31]

# Load existing confidences dictionary if it exists
if os.path.exists('confidences.pkl'):
    with open('confidences.pkl', 'rb') as f:
        confidences = pickle.load(f)
        print("Loaded existing confidences dictionary.")
else:
    confidences = {}
    print("No existing confidences dictionary found, starting fresh.")

for dataset, K in zip(datasets, K_vals):
    print(f"Processing dataset: {dataset}")
    if dataset not in confidences:
        confidences[dataset] = {}
    
    for model in models:
        # Check if the dataset-model combination is already present
        if model in confidences[dataset] and model+"_filtered" in confidences[dataset]:
            print(f"Dataset {dataset} with model {model} already processed. Skipping...")
            continue
        
        print(f"Processing model: {model}")
        path = f"../{model}/{dataset}/{model}_{dataset}.h5ad"
        adata = sc.read_h5ad(path)
        confidences[dataset][model] = adata.obs["velocity_confidence"]

        try:
            path_filtered = f"../{model}/{dataset}/{model}_{dataset}_filtered.h5ad"
            adata = sc.read_h5ad(path_filtered)
            confidences[dataset][model+"_filtered"] = adata.obs["velocity_confidence"]
        except:
            print(f"No filtered file found for {model} in {dataset}")
            path_filtered = None

        # Save the updated confidences dictionary after each dataset
        with open('confidences.pkl', 'wb') as f:
            pickle.dump(confidences, f)
            print(f"Saved confidences for {dataset}")

print("All datasets processed and saved.")
