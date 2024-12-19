import numpy as np
import scvelo as scv
import scanpy as sc
from sklearn.metrics.pairwise import cosine_similarity

def compute_velocity_coherence(adata, dataset_name, cell_type_key):
    """
    Compute velocity coherence and plot violin plot per cell type.

    Parameters:
    - adata: AnnData object
    - dataset_name: Name of the dataset (used in plot title)
    - cell_type_key: Key in adata.obs where cell types are stored

    Stores both velocity coherence (with directionality) and absolute velocity coherence 
    (without directionality) in adata.obs, and plots a violin plot per cell type.
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

    # Compute Ms_delta
    adata.layers["Ms_delta"] = adata.layers["Ms_extrap"] - adata.layers["Ms"]

    # Compute product_score (with directionality)
    prod = adata.layers["Ms_delta"] * adata.layers["velocity"]
    adata.layers["product_score"] = prod

    velocity_norms = np.linalg.norm(adata.layers["velocity"], axis=1, keepdims=True)
    adata.layers["velocity_normalized"] = adata.layers["velocity"] / (velocity_norms + 1e-9)  # Avoid division by zero

    prod_normalized = adata.layers["Ms_delta"] * adata.layers["velocity_normalized"]
    adata.layers["product_score"] = prod_normalized

    adata.obs["coherence_hadamard"] = prod.mean(axis=1)  
    adata.obs["coherence_hadamard_normalized"] = prod_normalized.mean(axis=1)  
    adata.obs["coherence_cosine"] = np.diag(cosine_similarity(adata.layers["Ms_delta"], adata.layers["velocity"]))
    adata.obs["absolute_coherence_cosine"] = np.abs(adata.obs["coherence_cosine"])

    # Compute sign agreement fraction
    sign_agreement = np.sign(adata.layers["Ms_delta"]) == np.sign(adata.layers["velocity"])
    adata.obs["coherence_fraction"] = sign_agreement.mean(axis=1)  # Fraction of matching signs per cell

