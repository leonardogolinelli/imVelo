import scanpy as sc
import numpy as np
import pandas as pd

def least_squares_slope(time_vector, quantity_vector):
    
    """
    Computes the slope of the least squares line for the given time and quantity vectors.
    
    Parameters:
    time_vector (numpy.ndarray): The input time vector.
    quantity_vector (numpy.ndarray): The input quantity vector.
    
    Returns:
    float: The slope of the least squares line.
    """
    n = len(time_vector)
    sum_x = np.sum(time_vector)
    sum_y = np.sum(quantity_vector)
    sum_xy = np.sum(time_vector * quantity_vector)
    sum_x_squared = np.sum(time_vector ** 2)
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x_squared - sum_x ** 2
    
    if denominator == 0:
        raise ValueError("Denominator is zero, cannot compute slope.")
    
    slope = numerator / denominator
    return slope

def learn_and_store_velocity(adata, cell_type_key):
    adata.layers["velocity_u"] = np.zeros(adata.shape)
    adata.layers["velocity"] = np.zeros(adata.shape)
    for ctype in pd.unique(adata.obs[cell_type_key]):
        print(f"computing slopes for ctype: {ctype}..")
        ctype_obs = np.where(adata.obs[cell_type_key] == ctype)[0]
        for i,gene in enumerate(list(adata.var_names)):
            Mu = adata.layers["Mu"]
            Ms = adata.layers["Ms"]
            quantity_vector_u = Mu[ctype_obs,i]
            quantity_vector_s = Ms[ctype_obs,i]
            time_vector = adata.obs["pseudotime"][ctype_obs]
            velocity_u = least_squares_slope(time_vector, quantity_vector_u)
            velocity = least_squares_slope(time_vector, quantity_vector_s)
            adata.layers["velocity_u"][ctype_obs, i] = velocity_u
            adata.layers["velocity"][ctype_obs, i] = velocity

            """if f"slope_{gene}_u" not in adata.obs:
                adata.obs[f"slope_{gene}_u"] = 0
            if f"slope_{gene}_s" not in adata.obs:
                adata.obs[f"slope_{gene}_s"] = 0

            adata.obs[f"slope_{gene}_u"][ctype_obs] = velocity_u
            adata.obs[f"slope_{gene}_s"][ctype_obs] = velocity"""