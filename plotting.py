import scanpy as sc
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import scvelo as scv
import pandas as pd
from utils import return_gnames
import matplotlib.patches as mpatches
import seaborn as sns
from utils import fetch_relevant_terms
from sklearn.manifold import Isomap

def plot_phase_plane(adata, gene_name, dataset, K, u_scale=.01, s_scale=0.01, alpha=0.5, head_width=0.02, head_length=0.03, length_includes_head=False, log=False,
                        norm_velocity=True, filter_cells=False, smooth_expr=True, show_plot=True, save_plot=True, save_path=".",
                        cell_type_key="clusters"):

    if smooth_expr:
        unspliced_expression = adata.layers["Mu"][:, adata.var_names.get_loc(gene_name)].flatten() 
        spliced_expression = adata.layers["Ms"][:, adata.var_names.get_loc(gene_name)].flatten() 
    else:
        unspliced_expression = adata.layers["unspliced"][:, adata.var_names.get_loc(gene_name)].flatten()
        spliced_expression = adata.layers["spliced"][:, adata.var_names.get_loc(gene_name)].flatten()

    # Normalize the expression data
    unspliced_expression_min, unspliced_expression_max = np.min(unspliced_expression), np.max(unspliced_expression)
    spliced_expression_min, spliced_expression_max = np.min(spliced_expression), np.max(spliced_expression)

    # Min-Max normalization
    unspliced_expression = (unspliced_expression - unspliced_expression_min) / (unspliced_expression_max - unspliced_expression_min)
    spliced_expression = (spliced_expression - spliced_expression_min) / (spliced_expression_max - spliced_expression_min)

    # Extract the velocity data
    unspliced_velocity = adata.layers['velocity_u'][:, adata.var_names.get_loc(gene_name)].flatten()
    spliced_velocity = adata.layers['velocity'][:, adata.var_names.get_loc(gene_name)].flatten()

    def custom_scale(data):
        max_abs_value = np.max(np.abs(data))  # Find the maximum absolute value
        scaled_data = data / max_abs_value  # Scale by the maximum absolute value
        return scaled_data

    if norm_velocity:
        unspliced_velocity = custom_scale(unspliced_velocity)
        spliced_velocity = custom_scale(spliced_velocity)


    # Apply any desired transformations (e.g., log) here
    if log:
        # Apply log transformation safely, ensuring no log(0)
        unspliced_velocity = np.log1p(unspliced_velocity)
        spliced_velocity = np.log1p(spliced_velocity)

    # Generate boolean masks for conditions and apply them
    if filter_cells:
        valid_idx = (unspliced_expression > 0) & (spliced_expression > 0)
    else:
        valid_idx = (unspliced_expression >= 0) & (spliced_expression >= 0)

    # Filter data based on valid_idx
    unspliced_expression_filtered = unspliced_expression[valid_idx]
    spliced_expression_filtered = spliced_expression[valid_idx]
    unspliced_velocity_filtered = unspliced_velocity[valid_idx]
    spliced_velocity_filtered = spliced_velocity[valid_idx]

    # Also filter cell type information to match the filtered expressions
    # First, get unique cell types and their corresponding colors
    unique_cell_types = adata.obs[cell_type_key].cat.categories
    celltype_colors = adata.uns[f"{cell_type_key}_colors"]
    
    # Create a mapping of cell type to its color
    celltype_to_color = dict(zip(unique_cell_types, celltype_colors))

    # Filter cell types from the data to get a list of colors for the filtered data points
    cell_types_filtered = adata.obs[cell_type_key][valid_idx]
    colors = cell_types_filtered.map(celltype_to_color).to_numpy()
    plt.figure(figsize=(9, 6.5), dpi=100)
  # Lower dpi here if the file is still too large    scatter = plt.scatter(unspliced_expression_filtered, spliced_expression_filtered, c=colors, alpha=0.6)

    """# Plot velocity vectors
    for i in range(len(unspliced_expression_filtered)):
        cell_type_index = np.where(unique_cell_types == cell_types_filtered[i])[0][0]
        arrow_color = celltype_to_color[cell_types_filtered[i]]  # Use the color corresponding to the cell type
        plt.arrow(
            unspliced_expression_filtered[i], spliced_expression_filtered[i], 
            unspliced_velocity_filtered[i] * u_scale, spliced_velocity_filtered[i] * s_scale, 
            color=arrow_color, alpha=alpha, head_width=head_width, head_length=head_length, length_includes_head=length_includes_head
        )"""

    # Plot velocity vectors
    for i in range(len(unspliced_expression_filtered)):
        cell_type_index = np.where(unique_cell_types == cell_types_filtered[i])[0][0]
        arrow_color = celltype_to_color[cell_types_filtered[i]]  # Use the color corresponding to the cell type
        plt.arrow(
            spliced_expression_filtered[i], unspliced_expression_filtered[i], 
            spliced_velocity_filtered[i] * s_scale, unspliced_velocity_filtered[i] * u_scale, 
            color=arrow_color, alpha=alpha, head_width=head_width, head_length=head_length, length_includes_head=length_includes_head
        )

    plt.ylabel(f'Normalized Unspliced Expression of {gene_name}')
    plt.xlabel(f'Normalized Spliced Expression of {gene_name}')
    plt.title(f'Expression and Velocity of {gene_name} by Cell Type')

    # Create a legend
    patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=celltype_to_color[celltype], markersize=10, label=celltype) 
            for celltype in unique_cell_types]
    plt.legend(handles=patches, title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left')

    

    if save_plot:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # Check if show_plot is True, then display the plot
    if show_plot:
        plt.show()
    else:
        plt.close()

    plt.show()

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_losses(trainer, dataset, K, figsize=(20, 10)):
    # Setup the figure and axes for 8 subplots
    loss_register = trainer.loss_register

    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=figsize)
    fig.suptitle('Logarithmic Scale Training and Evaluation Losses per Epoch', fontsize=16)

    # Names of the loss components
    loss_names = ['total_loss', 'recon_loss', 'kl_loss', 'heuristic_loss', "uniform_p_loss", "kl_weight"]

    # Function to safely convert tensor to numpy
    def safe_to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.cpu().numpy()  # Ensure tensor is moved to CPU and converted to numpy
        return value

    # Plot training losses
    for i, loss_name in enumerate(loss_names):
        modality = "training"
        if modality in loss_register and f'{loss_name}' in loss_register[modality]:
            # Apply safe conversion for each value
            loss_data = [safe_to_numpy(loss) for loss in loss_register[modality][f'{loss_name}']]
            axes[0, i].plot(np.log1p(loss_data))
            axes[0, i].set_title(f'Training {loss_name.replace("_", " ").title()}')
            axes[0, i].set_xlabel('Epoch')
            axes[0, i].set_ylabel('Log Loss')
            axes[0, i].grid(True)

    # Plot evaluation losses
    for i, loss_name in enumerate(loss_names):
        modality = "evaluation"
        if modality in loss_register and f'{loss_name}' in loss_register[modality]:
            # Apply safe conversion for each value
            loss_data = [safe_to_numpy(loss) for loss in loss_register[modality][f'{loss_name}']]
            axes[1, i].plot(np.log1p(loss_data))
            axes[1, i].set_title(f'Evaluation {loss_name.replace("_", " ").title()}')
            axes[1, i].set_xlabel('Epoch')
            axes[1, i].set_ylabel('Log Loss')
            axes[1, i].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"outputs/{dataset}/K{K}/stats/training_stats.png"
    os.makedirs(f"outputs/{dataset}/K{K}/stats/", exist_ok=True)
    
    plt.savefig(save_path)  # Save using plt.savefig

    # Show the plots
    plt.show()


def plot_rank_velocity_genes(adata, cell_type_key="clusters", min_corr=.3):
    scv.tl.rank_velocity_genes(adata, groupby=cell_type_key, min_corr=min_corr)
    df = pd.DataFrame(adata.uns['rank_velocity_genes']['names'])
    return df
 
def plot_important_genes(adata, dataset, K, cell_type_key):
    for gname in return_gnames():
        gname = gname.lower().capitalize()
        if gname in list(adata.var_names):
            os.makedirs(f"outputs/{dataset}/K{K}/phase_planes/important_genes/", exist_ok=True)
            path = f"outputs/{dataset}/K{K}/phase_planes/important_genes/{gname}.png"
            plot_phase_plane(adata, gname, dataset, K, u_scale=.01, s_scale=.01, alpha=0.5, norm_velocity=True,
                            show_plot=False, save_plot=True, save_path=path, cell_type_key=cell_type_key)

def gpvelo_plots(adata, dataset, K, cell_type_key):
    z = adata.obsm["z"].copy()
    embedding_key_2d = "X_umap" if dataset == "dentategyrus_lamanno" else "X_umap"
    umap = adata.obsm[embedding_key_2d].copy() 
    obs = adata.obs.copy()
    var_names = adata.uns["terms"]
    gp_velocity = adata.obsm["gp_velo"].copy()
    adata_gp = sc.AnnData(X=z)
    adata_gp.obs = obs
    adata_gp.var_names = var_names
    adata_gp.obsm[embedding_key_2d] = umap
    adata_gp.layers["spliced"] = adata_gp.X
    adata_gp.layers["Ms"] = adata_gp.X
    adata_gp.layers["Mu"] = adata_gp.X
    adata_gp.uns = adata.uns.copy()

    adata_gp.layers["velocity"] = gp_velocity

    terms = fetch_relevant_terms(dataset)

    os.makedirs(f"outputs/{dataset}/K{K}/gpvelo/scatter_terms/", exist_ok=True)

    sc.pp.neighbors(adata_gp)#, use_rep="X")
    scv.tl.velocity_graph(adata_gp)
    scv.pl.velocity_embedding_stream(adata_gp, color=cell_type_key)
    
    plt.savefig(f"outputs/{dataset}/K{K}/gpvelo/gpvelo_stream.png", bbox_inches='tight')
    plt.close()

    scv.tl.velocity_confidence(adata_gp)
    
    #if not dataset in ["dentategyrus_lamanno", "dentategyrus_lamanno_P0", "dentategyrus_lamanno_P5", "gastrulation_erythroid"]:
    scv.tl.velocity_pseudotime(adata_gp)
    sc.pl.umap(adata_gp, color=["velocity_confidence"], color_map="coolwarm")
    
    plt.savefig(f"outputs/{dataset}/K{K}/gpvelo/gpvelo_confidence.png", bbox_inches='tight')
    plt.close()

    sc.pl.umap(adata_gp, color=["velocity_length"], color_map="coolwarm")
    
    plt.savefig(f"outputs/{dataset}/K{K}/gpvelo/gpvelo_length.png", bbox_inches='tight')
    plt.close()

    sc.pl.umap(adata_gp, color=["velocity_pseudotime"], color_map="gnuplot")
    
    plt.savefig(f"outputs/{dataset}/K{K}/gpvelo/gpvelo_pseudotime.png", bbox_inches='tight')
    plt.close()

    scv.tl.rank_velocity_genes(adata_gp, groupby=cell_type_key)
    scv.pl.rank_genes_groups(adata_gp, ncols=2, key="rank_velocity_genes")    
    
    plt.savefig(f"outputs/{dataset}/K{K}/gpvelo/gpvelo_deg.png", bbox_inches='tight')
    plt.close()

    """else:        
        plt.savefig(f"outputs/{dataset}/K{K}/gpvelo/gpvelo_confidence.png", bbox_inches='tight')
        plt.close()

        sc.pl.umap(adata_gp, color=["velocity_length"], color_map="coolwarm")
        
        plt.savefig(f"outputs/{dataset}/K{K}/gpvelo/gpvelo_length.png", bbox_inches='tight')
        plt.close()

        scv.tl.rank_velocity_genes(adata_gp, groupby=cell_type_key)
        scv.pl.rank_genes_groups(adata_gp, ncols=2, key="rank_velocity_genes")    
        
        plt.savefig(f"outputs/{dataset}/K{K}/gpvelo/gpvelo_deg.png", bbox_inches='tight')
        plt.close()  """      
    

    adata_gp.var_names = [term.replace("/", "-") for term in list(adata_gp.var_names)]
    terms = [term.replace("/", "-") for term in terms]

    not_present = []
    for term in terms:
        if term not in list(adata_gp.var_names):
            not_present.append(term)
            terms.remove(term)
            print(f"{term} removed from list of terms")

    """if len(not_present) > 0:
        for term in not_present:
            print(f"not present term: {term}")"""

    base_path = f"outputs/{dataset}/K{K}/gpvelo/terms_over_time/"
    os.makedirs(base_path, exist_ok=True)
    pseudotime_key = "isomap_1"

    for term in terms:
        if term in list(adata_gp.var_names):
            title_X = f"{term}_activation_over_time"
            title_velo = f"GP_velocity_of_{term}_over_time"
            path_X = base_path + f"{title_X}.png"
            path_velo = base_path + f"{title_velo}.png"

            sc.pl.scatter(adata_gp, x=pseudotime_key, y=f"{term}", color=cell_type_key, layers="X",
            title=title_X)
            
            plt.savefig(path_X, bbox_inches='tight')
            plt.close()
            sc.pl.scatter(adata_gp, x=pseudotime_key, y=f"{term}", color=cell_type_key, layers="velocity",
            title=title_velo)
            
            plt.savefig(path_velo, bbox_inches='tight')
            plt.close()

    term_dic = {}
    for i, term1 in enumerate(terms):
        for j, term2 in enumerate(terms):
            if (term1 in list(adata_gp.var_names)) and (term2 in list(adata_gp.var_names)):
                if (f"{term1}-{term2}" not in term_dic) or (f"{term2}-{term1}" not in term_dic) and (term1 != term2):
                    term_dic[f"{term1}-{term2}"] = 1
                    term_dic[f"{term2}-{term1}"] = 1
                    sc.pl.scatter(adata_gp, term1, term2, color=cell_type_key)
                    plt.savefig(f"outputs/{dataset}/K{K}/gpvelo/scatter_terms/{term1}-{term2}.png", bbox_inches='tight')

    os.makedirs(f"outputs/{dataset}/K{K}/gpvelo/terms_umap/", exist_ok=True)
    for term in terms[1:]:
        if term in list(adata_gp.var_names):
                sc.pl.umap(adata_gp, color=term, show=False)
                path = f"outputs/{dataset}/K{K}/gpvelo/terms_umap/term_{term}_umap.png"
                plt.savefig(path, bbox_inches="tight")

    adata_gp.write_h5ad(f"outputs/{dataset}/K{K}/gpvelo/adata_gp.h5ad")



def compute_celldeviation_maxstatevelo(adata, dataset, K, cell_type_key):
    # Assuming you have your matrices as follows:
    pp = adata.layers["pp"]
    nn = adata.layers["nn"]
    pn = adata.layers["pn"]
    np_matrix = adata.layers["np"]

    # Stack the matrices along a new axis
    stacked_matrices = np.stack((pp, nn, pn, np_matrix), axis=-1)

    # Use np.argmax to find the index of the maximum value along the new axis
    max_indices = np.argmax(stacked_matrices, axis=-1)

    # Compute aggregate deviation from uniform distribution (0.25 for each probability)
    uniform_prob = 0.25
    uniform_dist = np.full(stacked_matrices.shape, uniform_prob)
    aggregate_deviation_matrix = np.sum(np.abs(stacked_matrices - uniform_dist), axis=2)
    adata.layers["deviation_uniform"] = aggregate_deviation_matrix

    """biomarker_list = ["Gcg","Ins1","Ghrl", "Sst", "Sox9","Neurog3", "Hes1", "Neurod1"]

    for gene in biomarker_list:
        if gene in list(adata.var_names):
            sc.pl.umap(adata, layer="X", color=gene, title=f"gene: {gene}, layer: X", show=False)
            
            plt.savefig(f"outputs/{dataset}/K{K}/stats/uncertainty/probabilities/{gene}_X.png", bbox_inches='tight')
            plt.close()
            
            sc.pl.umap(adata, layer="velocity", color=gene, title=f"gene: {gene}, layer: velocity", show=False)
            
            plt.savefig(f"outputs/{dataset}/K{K}/stats/uncertainty/probabilities/{gene}_velocity.png", bbox_inches='tight')
            plt.close()
            
            sc.pl.umap(adata, layer="deviation_uniform", color=gene, title=f"gene: {gene}, layer: deviation from uniform", show=False)
            
            plt.savefig(f"outputs/{dataset}/K{K}/stats/uncertainty/probabilities/{gene}_deviation.png", bbox_inches='tight')
            plt.close()"""

    signs = np.array([(1, 1), (-1, -1), (1, -1), (-1, 1)])
    ########## maxstate ##############
    alpha = adata.layers["alpha"]
    beta = adata.layers["beta"]
    gamma = adata.layers["gamma"]
    u = adata.layers["Mu"]
    s = adata.layers["Ms"]

    # Extract signs based on `max_indices`
    sign_u = signs[max_indices][:, :, 0]
    sign_s = signs[max_indices][:, :, 1]

    # Compute new velocities
    new_velo_u = sign_u * (alpha - beta * u)
    new_velo = sign_s * (beta * u - gamma * s)

    adata.layers["new_velo"] = new_velo
    adata.layers["new_velo_u"] = new_velo_u

    sc.pp.neighbors(adata)#, use_rep="MuMs")
    #sc.tl.umap(adata)
    scv.tl.velocity_graph(adata, vkey="new_velo")
    scv.pl.velocity_embedding_stream(adata, vkey="new_velo", color=cell_type_key, show=False)
    
    plt.savefig(f"outputs/{dataset}/K{K}/embeddings/max_p_velo.png", bbox_inches='tight')


import numpy as np

def compute_velocity_sign_uncertainty(adata, aggregate_method='mean'):
    """
    Computes the velocity sign uncertainty at both the cell and gene levels.
    
    Parameters:
        pp: numpy array (cells x genes) - probabilities of positive unspliced and positive spliced RNA velocity.
        nn: numpy array (cells x genes) - probabilities of negative unspliced and negative spliced RNA velocity.
        pn: numpy array (cells x genes) - probabilities of positive unspliced and negative spliced RNA velocity.
        np_matrix: numpy array (cells x genes) - probabilities of negative unspliced and positive spliced RNA velocity.
        aggregate_method: str - Method to aggregate uncertainties across genes in each cell ('mean' or 'sum').
        
    Returns:
        gene_uncertainty: numpy array (genes) - average uncertainty per gene across cells.
        cell_uncertainty: numpy array (cells) - aggregated uncertainty per cell (mean or sum across genes).
    """

    pp = adata.layers["pp"]
    nn = adata.layers["nn"]
    pn = adata.layers["pn"]
    np_matrix = adata.layers["np"]

    # Stack the probabilities along the third dimension (axis=-1)
    stacked_matrices = np.stack((pp, nn, pn, np_matrix), axis=-1)
    
    # Define uniform probability (0.25 for each velocity sign configuration)
    uniform_prob = 0.25
    
    # Compute the deviation from uniform distribution for each configuration
    deviation = np.abs(stacked_matrices - uniform_prob)
    
    # Sum the deviations across the four configurations (axis=-1 gives sum per gene per cell)
    cell_gene_uncertainty = np.sum(deviation, axis=-1)
    
    # Compute gene-level uncertainty by averaging across cells (axis=0)
    gene_uncertainty = np.mean(cell_gene_uncertainty, axis=0)
    
    # Compute cell-level uncertainty by aggregating across genes for each cell (axis=1)
    if aggregate_method == 'mean':
        cell_uncertainty = np.mean(cell_gene_uncertainty, axis=1)
    elif aggregate_method == 'sum':
        cell_uncertainty = np.sum(cell_gene_uncertainty, axis=1)
    else:
        raise ValueError("Invalid aggregate_method. Use 'mean' or 'sum'.")
    
    adata.var["p_gene_uncertainty"] = gene_uncertainty
    adata.obs["p_cell_uncertainty"] = cell_uncertainty



def plot_embeddings(adata, dataset, K, cell_type_key="clusters"):
    sc.pp.neighbors(adata)
    scv.tl.velocity_graph(adata)

    #concatenated unspliced - spliced velocity adata object for computing the velocity embedding
    X = np.concatenate([adata.layers["velocity_u"], adata.layers["velocity"]], axis=1)
    adata_velocity = sc.AnnData(X=X)
    adata_velocity.obs = adata.obs.copy()
    adata_velocity.uns = adata.uns.copy()
    adata_velocity.layers["velocity"] = adata_velocity.X
    adata_velocity.layers["spliced"] = adata_velocity.X
    sc.pp.neighbors(adata_velocity)#, use_rep="X")
    sc.tl.umap(adata_velocity)
    save_path = f"outputs/{dataset}/K{K}/embeddings/concat_velocity_adata.png"
    scv.tl.velocity_graph(adata_velocity)
    scv.pl.velocity_embedding_stream(adata_velocity, color=cell_type_key, 
                                    show=False)
    
    os.makedirs(f"outputs/{dataset}/K{K}/embeddings/", exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')

    #anndata of spliced velocity embedding
    adata_velocity = sc.AnnData(X=adata.layers["velocity"].copy())
    adata_velocity.obs = adata.obs.copy()
    adata_velocity.uns = adata.uns.copy()
    adata_velocity.layers["velocity"] = adata_velocity.X
    adata_velocity.layers["spliced"] = adata_velocity.X
    save_path = f"outputs/{dataset}/K{K}/embeddings/velocity_adata.png"
    sc.pp.neighbors(adata_velocity)#, use_rep="X")
    sc.tl.umap(adata_velocity)
    scv.tl.velocity_graph(adata_velocity)
    scv.pl.velocity_embedding_stream(adata_velocity,color=cell_type_key, show=False)
    
    plt.savefig(save_path, bbox_inches='tight')

    #legit plot of gene expression data with velocity vectors overlayed
    save_path = f"outputs/{dataset}/K{K}/embeddings/velocity.png"
    sc.pp.neighbors(adata)
    #sc.tl.umap(adata)
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding_stream(adata, color=cell_type_key, 
                                    show=False)
    
    plt.savefig(save_path, bbox_inches='tight')

    #UMAP of predicted data according to velocity
    sc.pp.neighbors(adata, use_rep="prediction")
    #sc.tl.umap(adata)
    if dataset == "dentategyrus_lamanno":
        sc.pl.umap(adata, color=cell_type_key, show=False)
    else:
        sc.pl.umap(adata, color=cell_type_key, show=False)
    
    plt.savefig(f"outputs/{dataset}/K{K}/embeddings/prediction.png", bbox_inches='tight')  # Save using plt.savefig
    plt.close()
    
    #umap of latent space
    adata_z = sc.AnnData(X=adata.obsm["z"])
    adata_z.obs = adata.obs.copy()
    sc.pp.neighbors(adata_z)#, use_rep="X")
    sc.tl.umap(adata_z)
    sc.pl.umap(adata_z, color=cell_type_key, show=False)

    
    plt.savefig(f"outputs/{dataset}/K{K}/embeddings/z.png", bbox_inches='tight')  # Save using plt.savefig

    os.makedirs(f"outputs/{dataset}/K{K}/embeddings/",exist_ok=True)
    sc.pl.umap(adata, color="p_cell_uncertainty")
    plt.savefig(f"outputs/{dataset}/K{K}/embeddings/p_cell_uncertainty.png", bbox_inches='tight')
    plt.close()
    
def plot_probabilities(adata, dataset, gene_name = "Rbfox3", x_label = "pp", y_label = "nn", cell_type_key="clusters"):
    # Assuming 'adata' is defined and properly structured
    # Extracting data for a specific gene across all cells
    x = adata[:, gene_name].layers[x_label].flatten()  # Assuming layers are correctly structured
    y = adata[:, gene_name].layers[y_label].flatten()

    # Use the provided color palette and your predefined cluster order
    cluster_colors = adata.uns[f"{cell_type_key}_colors"]
    unique_clusters = ["Ductal", "Ngn3 low EP", "Ngn3 high EP", "Pre-endocrine", "Beta", "Alpha", "Epsilon", "Delta"]

    # Ensure the number of colors matches the number of unique cluster names
    if len(cluster_colors) != len(unique_clusters):
        raise ValueError("The number of colors does not match the number of unique clusters.")

    # Map each cluster to a color from the provided palette
    cluster_to_color = dict(zip(unique_clusters, cluster_colors))

    # Retrieve the cluster labels from adata.obs assuming 'clusters' corresponds to these labels
    clusters = adata.obs[cell_type_key].to_numpy()  # Ensure this key is correct

    # Color points by their cluster label
    point_colors = [cluster_to_color[cluster] for cluster in clusters]

    # Create scatter plot
    plt.scatter(x, y, color=point_colors)

    # Create a custom legend
    legend_handles = [mpatches.Patch(color=cluster_to_color[cluster], label=cluster) for cluster in unique_clusters]
    plt.legend(handles=legend_handles)

    # Label axes for clarity
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Show the plot
    plt.show()

def plot_isomaps(adata, dataset_name, K, cell_type_key):
    base_path = f"outputs/{dataset_name}/K{K}/isomap_plots/"
    os.makedirs(base_path, exist_ok=True)
    for rep in ["ve", "MuMs", "z", "velocity", "velocity_concat", "PCA"]:
        if rep == "ve":
            rep_matrix=adata.obsm["ve"]
        elif rep == "MuMs":
            rep_matrix=np.concatenate([adata.layers["Mu"], adata.layers["Ms"]], axis=1)
        elif rep == "z":
            rep_matrix=adata.obsm["z"]
        elif rep == "velocity":
            rep_matrix = adata.layers["velocity"]
        elif rep == "velocity_concat":
            rep_matrix = np.concatenate([adata.layers["velocity_u"], adata.layers["velocity"]], axis=1)

        isomap = Isomap(n_components=3, n_neighbors=5)#, n_jobs=-1)
        rep_embedding = isomap.fit_transform(rep_matrix)
        cats = adata.obs[cell_type_key].values
        unique_categories = adata.obs[cell_type_key].cat.categories
        colors = adata.uns[f"{cell_type_key}_colors"]
        category_to_color = dict(zip(unique_categories, colors))

        adata.obs[f"isomap_1_{rep}"] = x = rep_embedding[:,0]
        adata.obs[f"isomap_2_{rep}"] = y = rep_embedding[:,1]
        adata.obs[f"isomap_3_{rep}"] = z = rep_embedding[:,2]

        # Create a new figure for plotting
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each category with its corresponding color
        for category in unique_categories:
            mask = cats == category
            ax.scatter(x[mask], y[mask], z[mask], label=category, c=category_to_color[category], edgecolors='w')

        # Set labels for axes
        ax.set_xlabel(f'isomap_1_{rep}')
        ax.set_ylabel(f'isomap_2_{rep}')
        ax.set_zlabel(f'isomap_3_{rep}')

        title = f"{dataset_name}_3D isomaps"

        # Add legend
        ax.legend()
        ax.set_title(title)

        # Show the plot
        plt.show()
        plt.savefig(f"{base_path}_3D_isomap_{rep}")#, bbox_inches="tight")

        sc.pl.umap(adata, color=f"isomap_1_{rep}")
        plt.savefig(f"{base_path}_isomap_1_{rep}", bbox_inches="tight")
        sc.pl.umap(adata, color=f"isomap_2_{rep}")
        plt.savefig(f"{base_path}_isomap_2_{rep}", bbox_inches="tight")
        sc.pl.umap(adata, color=f"isomap_3_{rep}")
        plt.savefig(f"{base_path}_isomap_3_{rep}", bbox_inches="tight")



        sc.pl.scatter(adata, x=f"isomap_1_{rep}", y=f"isomap_2_{rep}", color=cell_type_key, title=f"{title}, isomap 1 vs 2")
        plt.savefig(f"{base_path}_isomap_1_vs_isomap_2_{rep}", bbox_inches="tight")

        sc.pl.scatter(adata, x=f"isomap_1_{rep}", y=f"isomap_3_{rep}", color=cell_type_key, title=f"{title}, isomap 2 vs 3")
        plt.savefig(f"{base_path}_isomap_1_vs_isomap_3_{rep}", bbox_inches="tight")

        sc.pl.scatter(adata, x=f"isomap_2_{rep}", y=f"isomap_3_{rep}", color=cell_type_key, title=f"{title}, isomap 3 vs 4")
        plt.savefig(f"{base_path}_isomap_2_vs_isomap_3_{rep}", bbox_inches="tight")


def plot_3d(adata, rep, rep_matrix, cell_type_key, dataset_name, base_path=None):
    # Create an Isomap instance and fit-transform the data
    isomap = Isomap(n_components=3, n_neighbors=5)
    rep_embedding = isomap.fit_transform(rep_matrix)

    # Extract categories and colors
    cats = adata.obs[cell_type_key].values
    unique_categories = adata.obs[cell_type_key].cat.categories
    colors = adata.uns[f"{cell_type_key}_colors"]
    category_to_color = dict(zip(unique_categories, colors))

    # Add the Isomap components to the AnnData object
    adata.obs[f"isomap_1_{rep}"] = rep_embedding[:, 0]
    adata.obs[f"isomap_2_{rep}"] = rep_embedding[:, 1]
    adata.obs[f"isomap_3_{rep}"] = rep_embedding[:, 2]

    # Create a new figure for plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each category with its corresponding color
    for category in unique_categories:
        mask = cats == category
        ax.scatter(rep_embedding[mask, 0], rep_embedding[mask, 1], rep_embedding[mask, 2], 
                   label=category, c=category_to_color[category], edgecolors='w')

    # Set labels for axes
    ax.set_xlabel(f'isomap_1_{rep}')
    ax.set_ylabel(f'isomap_2_{rep}')
    ax.set_zlabel(f'isomap_3_{rep}')

    # Add title and legend
    title = f"{dataset_name}_3D Isomap_rep_{rep}"
    ax.set_title(title)
    ax.legend()

    # Show the plot
    plt.show()

    # Save the plot if base_path is provided
    if base_path:
        fig.savefig(f"{base_path}_{dataset_name}_3D_isomap_{rep}.png")  # Save as .png file