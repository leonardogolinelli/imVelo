import scanpy as sc
from velocity_coherence import *
import pickle
import scipy.sparse as sp
import numpy as np

# Models and datasets
models = ["celldancer", "imVelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "scvelo", "stochastic"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters", "celltype", "clusters"]

# Paths for saving the dictionaries
dic_path = "coherences.pkl"
abs_dic_path = "absolute_coherences.pkl"

# Load the coherence dictionaries from Pickle files if they exist
try:
    with open(dic_path, "rb") as pickle_file:
        coherences = pickle.load(pickle_file)
except FileNotFoundError:
    # Initialize an empty dictionary if Pickle file does not exist
    coherences = {}

try:
    with open(abs_dic_path, "rb") as abs_pickle_file:
        absolute_coherences = pickle.load(abs_pickle_file)
except FileNotFoundError:
    # Initialize an empty dictionary if Pickle file does not exist
    absolute_coherences = {}

# Process datasets and models
for dataset, cell_type_key in zip(datasets, cell_type_keys):
    print(f"Processing dataset: {dataset}")
    
    # Ensure the dataset exists in both dictionaries
    if dataset not in coherences:
        coherences[dataset] = {}
    if dataset not in absolute_coherences:
        absolute_coherences[dataset] = {}
    
    for model in models:
        # Check if the model has already been processed for the dataset in both dictionaries
        if model not in coherences[dataset] or model not in absolute_coherences[dataset]:
            print(f"Processing model: {model}")
            adata = sc.read_h5ad(f"../{model}/{dataset}/{model}_{dataset}.h5ad")
            
            # If model is 'scvelo', filter out genes with NaN values
            if model == 'scvelo':
                print("Filtering genes with NaN values for scvelo model")

                # Convert adata.X to dense if it's sparse
                if sp.issparse(adata.X):
                    adata.X = adata.X.toarray()

                adata = adata[:, adata.var["velocity_genes"]].copy()

                # Check for any remaining NaN values
                if np.isnan(adata.X).any():
                    raise ValueError("Warning: NaN values still present after filtering!")
                else:
                    print("No NaN values found after filtering.")

            # Compute velocity coherence and absolute velocity coherence
            compute_velocity_coherence(adata, dataset, cell_type_key, plot=False)
            
            # Store the standard velocity coherence in the dictionary
            coherences[dataset][model] = adata.obs["velocity_coherence"].tolist()  # Convert to list for Pickle compatibility
            
            # Store the absolute velocity coherence in the dictionary
            absolute_coherences[dataset][model] = adata.obs["absolute_velocity_coherence"].tolist()  # Convert to list
            
            # Save both dictionaries to Pickle files
            with open(dic_path, "wb") as pickle_file:
                pickle.dump(coherences, pickle_file)
            with open(abs_dic_path, "wb") as abs_pickle_file:
                pickle.dump(absolute_coherences, abs_pickle_file)

print("Processing complete and results saved.")