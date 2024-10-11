import numpy as np
import pandas as pd
import scvelo as scv
import scanpy as sc

def compute_velocity_coherence(adata, dataset_name, cell_type_key, plot=True):
    """
    Compute velocity coherence and plot violin plot per cell type.

    Parameters:
    - adata: AnnData object
    - dataset_name: Name of the dataset (used in plot title)
    - cell_type_key: Key in adata.obs where cell types are stored

    Stores both velocity coherence (with directionality) and absolute velocity coherence 
    (without directionality) in adata.obs, and plots a violin plot per cell type.
    """
    import scvelo as scv
    import scanpy as sc
    import numpy as np

    # Compute velocity graph
    scv.tl.velocity_graph(adata)

    # Get transition matrix
    tm = scv.utils.get_transition_matrix(
        adata, use_negative_cosines=True, self_transitions=True
    )
    tm.setdiag(0)

    # Extrapolate Ms
    adata.layers["Ms_extrap"] = tm @ adata.layers["Ms"]

    # Compute Ms_delta
    adata.layers["Ms_delta"] = adata.layers["Ms_extrap"] - adata.layers["Ms"]

    # Compute product_score (with directionality)
    prod = adata.layers["Ms_delta"] * adata.layers["velocity"]
    adata.layers["product_score"] = prod

    # Compute mean product_score per cell (mean over genes, with directionality)
    mean_product_score_per_cell = prod.mean(axis=1)  # Convert matrix to array if necessary

    # Store velocity coherence (with directionality)
    adata.obs["velocity_coherence"] = mean_product_score_per_cell

    # Compute absolute product_score (without directionality)
    abs_prod = np.abs(prod)  # Taking the absolute value
    adata.layers["absolute_product_score"] = abs_prod

    # Compute mean absolute_product_score per cell (mean over genes, without directionality)
    mean_abs_product_score_per_cell = abs_prod.mean(axis=1)  # Convert matrix to array if necessary

    # Store absolute velocity coherence (without directionality)
    adata.obs["absolute_velocity_coherence"] = mean_abs_product_score_per_cell
