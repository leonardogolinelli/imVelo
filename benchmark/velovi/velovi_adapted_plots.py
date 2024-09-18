import os, sys
current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))
import matplotlib.pyplot as plt
import numpy as np
from velovi_adapted_utils import return_gnames
import os
import torch
import scanpy as sc
import scvelo as scv
import pandas as pd
from velovi._model import _compute_directional_statistics_tensor
from scvi.utils import track
from contextlib import redirect_stdout
import io
import seaborn as sns


def plot_phase_plane(adata, gene_name, u_scale=.01, s_scale=0.01, alpha=0.5, head_width=0.02, head_length=0.03, length_includes_head=False, log=False,
                        norm_velocity=True, filter_cells=False, smooth_expr=True, show_plot=True, save_plot=True, save_path=".",
                        cell_type_key="clusters"):

    if not gene_name in list(adata.var_names):
        return 0
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
        plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Plot saved to {save_path}")

    # Check if show_plot is True, then display the plot
    if show_plot:
        plt.show()
    else:
        plt.close()

    plt.show()



def plot_important_genes(adata, dataset, cell_type_key):
    dir_path = f"{dataset}/interesting_genes/"
    os.makedirs(dir_path, exist_ok=True)
    for gname in return_gnames():
        if gname in list(adata.var_names):
            path = dir_path + f"{gname}.png"
            plot_phase_plane(adata, gname, u_scale=.01, s_scale=0.01, alpha=0.5, head_width=0.02, head_length=0.03, length_includes_head=False, log=False,
                            norm_velocity=True, filter_cells=False, smooth_expr=True, show_plot=True, save_plot=True, save_path=path,
                            cell_type_key=cell_type_key)



def plot_training_stats(vae, dataset):
    # Assuming 'model.history' is a dictionary with 12 different keys, each corresponding to a metric
    metrics = vae.history
    num_plots = len(metrics)  # Number of metrics to plot

    # Create a figure with 4 rows and 3 columns
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(15, 20))
    fig.tight_layout(pad=5.0)  # Adds padding between plots for clarity

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Loop through the metrics and create a plot for each
    for i, (key, val) in enumerate(metrics.items()):
        ax = axes[i]
        ax.plot(val, label=f'{key}', marker='o', linestyle='-')  # You can customize the line style
        ax.set_title(key)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')

    # Hide any unused axes if the number of metrics is less than the number of subplots
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    # Show the plotWWWW
    
    plt.savefig(f"{dataset}/training_stats.png", bbox_inches="tight")
    plt.close()


def intrinsic_uncertainty(adata, vae, dataset, n_samples):
    uncertainty_df, _ = vae.get_directional_uncertainty(n_samples=n_samples)
    #uncertainty_df.head()
    for c in uncertainty_df.columns:
        adata.obs[c] = np.log10(uncertainty_df[c].values)
    sc.pl.umap(
        adata, 
        color="directional_cosine_sim_variance",
        cmap="Greys",
        vmin="p1",
        vmax="p99",
    )
    
    plt.savefig(f"{dataset}/stats/intrinsic_uncertainty.png", bbox_inches="tight")
    plt.close()
    return adata



def extrinsic_uncertainty(adata, vae, dataset, n_samples=25) -> pd.DataFrame:
    extrapolated_cells_list = []
    for i in track(range(n_samples)):
        with io.StringIO() as buf, redirect_stdout(buf):
            vkey = "velocities_velovi_{i}".format(i=i)
            v = vae.get_velocity(n_samples=1, velo_statistic="mean")
            adata.layers[vkey] = v
            scv.tl.velocity_graph(adata, vkey=vkey, sqrt_transform=False, approx=True)
            t_mat = scv.utils.get_transition_matrix(
                adata, vkey=vkey, self_transitions=True, use_negative_cosines=True
            )
            extrapolated_cells = np.asarray(t_mat @ adata.layers["Ms"])
            extrapolated_cells_list.append(extrapolated_cells)
    extrapolated_cells = np.stack(extrapolated_cells_list)
    ext_uncertainty_df, _ = _compute_directional_statistics_tensor(extrapolated_cells, n_jobs=-1, n_cells=adata.n_obs)
    for c in ext_uncertainty_df.columns:
        adata.obs[c + "_extrinisic"] = np.log10(ext_uncertainty_df[c].values)

    sc.pl.umap(
    adata, 
    color="directional_cosine_sim_variance_extrinisic",
    vmin="p1", 
    vmax="p99", 
)
    
    
    plt.savefig(f"{dataset}/stats/extrinsic_uncertainty.png", bbox_inches="tight")
    plt.close()
    return adata

def plot_permutation_score(adata, vae, dataset, cell_type_key):
    perm_df, _ = vae.get_permutation_scores(labels_key=cell_type_key)
    adata.var["permutation_score"] = perm_df.max(1).values
    sns.kdeplot(data=adata.var, x="permutation_score")
    
    plt.savefig(f"{dataset}/stats/permutation_score.png", bbox_inches="tight")
    plt.close()