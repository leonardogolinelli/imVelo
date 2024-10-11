import numpy as np
import pandas as pd
import scvelo as scv
import scanpy as sc
from sklearn.metrics.pairwise import cosine_similarity


def compute_velocity_coherence(adata, dataset_name, cell_type_key):
    """
    Compute velocity coherence using cosine similarity and plot violin plot per cell type.
    
    Parameters:
    - adata: AnnData object
    - dataset_name: Name of the dataset (used in plot title)
    - cell_type_key: Key in adata.obs where cell types are stored
    
    Stores both velocity coherence (cosine similarity) and absolute velocity coherence 
    (without directionality) in adata.obs.
    """
    
    # Compute velocity graph
    sc.pp.neighbors(adata)
    scv.tl.velocity_graph(adata)

    # Get transition matrix
    tm = scv.utils.get_transition_matrix(
        adata, use_negative_cosines=True, self_transitions=True
    )
    tm.setdiag(0)

    # Extrapolate Ms
    adata.layers["Ms_extrap"] = tm @ adata.layers["Ms"]

    # Compute Ms_delta (gene expression change vector)
    adata.layers["Ms_delta"] = adata.layers["Ms_extrap"] - adata.layers["Ms"]

    # Compute cosine similarity in a vectorized way
    velocities = adata.layers["velocity"]
    ms_delta = adata.layers["Ms_delta"]

    # Calculate pairwise cosine similarity between ms_delta and velocity vectors (per cell)
    cosine_similarities = np.diag(cosine_similarity(ms_delta, velocities))

    # Store velocity coherence (cosine similarity)
    adata.obs["velocity_coherence"] = cosine_similarities

    # Compute absolute cosine similarity (without directionality)
    adata.obs["absolute_velocity_coherence"] = np.abs(cosine_similarities)
