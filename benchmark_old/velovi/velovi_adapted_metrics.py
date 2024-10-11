import os, sys
current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import scvelo as scv
from velovi_adapted_plots import plot_phase_plane

def deg_genes(adata, dataset, cell_type_key="clusters", n_deg_rows=5):
    print(f"Computing DEG genes..")
    import os
    wilcoxon_path = f"{dataset}/deg_genes/wilcoxon/"
    phase_plane_path = f"{dataset}/deg_genes/phase_plane/"
    os.makedirs(wilcoxon_path, exist_ok=True)
    os.makedirs(phase_plane_path, exist_ok=True)
    if dataset == "forebrain":
        adata.obs["Clusters"] = pd.Series(adata.obs["Clusters"], dtype="category")

    for layer in ["Ms", "Mu"]:
        sc.tl.rank_genes_groups(adata, groupby=cell_type_key, layer=layer, method='wilcoxon')
        sc.pl.rank_genes_groups(adata, ncols=2, show=False)
        save_path = f"{wilcoxon_path}wilcoxon_on_{layer}.png"
        
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        # Extract DEGs
        degs = pd.DataFrame()
        for cluster in adata.obs[cell_type_key].unique():
            if dataset == "forebrain":
                cluster = str(cluster)
            degs[cluster] = pd.DataFrame(adata.uns['rank_genes_groups']['names'])[cluster]
        
        # Save the DataFrame in adata.uns
        adata.uns[f'deg_genes_{layer}'] = degs

    for layer in ["velocity", "velocity_u"]:
        scv.tl.rank_velocity_genes(adata, vkey=layer, groupby=cell_type_key, min_corr=.3)
        scv.pl.rank_genes_groups(adata, ncols=2, key="rank_velocity_genes")
        save_path = f"{wilcoxon_path}wilcoxon_on_{layer}.png"
        
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        # Extract DEGs
        degs = pd.DataFrame()
        for cluster in adata.obs[cell_type_key].unique():
            cluster = str(cluster)
            degs[cluster] = pd.DataFrame(adata.uns['rank_genes_groups']['names'])[cluster]
        
        # Save the DataFrame in adata.uns
        adata.uns[f'deg_genes_{layer}'] = degs

    deg_df = adata.uns[f"deg_genes_velocity"]
    i=0
    while i < n_deg_rows:
        for gene in deg_df.iloc[i]:
            save_path = f"{phase_plane_path}{gene}.png"
            plot_phase_plane(adata, gene, save_path=save_path, show_plot=False, cell_type_key=cell_type_key)
        i+=1
    
    return adata

def compute_scvelo_metrics(adata, dataset, show=False, cell_type_key="clusters"):
    print("computing scvelo metrics..")
    scv.tl.velocity_confidence(adata)
    scv.tl.velocity_pseudotime(adata)

    confidence_path = f"{dataset}/scvelo_metrics/confidence/"
    s_genes_path = f"{dataset}/scvelo_metrics/s_genes/"
    g2m_genes_path = f"{dataset}/scvelo_metrics/g2m_genes/"
    for path in [confidence_path, s_genes_path, g2m_genes_path]:
        os.makedirs(path, exist_ok = True)

    keys = cell_type_key, 'velocity_length', 'velocity_confidence', 'velocity_pseudotime'
    cmaps = [None, 'coolwarm', 'coolwarm', 'gnuplot', 'gnuplot']
    for i, key in enumerate(keys):
        sc.pl.umap(adata, color=key, color_map=cmaps[i], show=show)
        
        plt.savefig(f"{confidence_path}{key}.png", bbox_inches='tight')

    for i, key in enumerate(keys):
        scv.pl.velocity_embedding_stream(adata, color=key, color_map=cmaps[i], show=show)
        
        plt.savefig(f"{confidence_path}{key}_stream.png", bbox_inches='tight')
    
    scv.tl.rank_velocity_genes(adata, groupby=cell_type_key, min_corr=.3)
    s_genes, g2m_genes = scv.utils.get_phase_marker_genes(adata)
    if len(s_genes) > 0:
        s_genes = scv.get_df(adata[:, s_genes], 'spearmans_score', sort_values=True).index
    if len(g2m_genes) > 0:
        g2m_genes = scv.get_df(adata[:, g2m_genes], 'spearmans_score', sort_values=True).index

    if len(s_genes) > 0:
        for gene in s_genes:
            save_path = f"{s_genes_path}{gene}"
            scv.pl.velocity(adata, gene, 
                            add_outline=True, show=False) 
            
            plt.savefig(save_path, bbox_inches='tight')

    if len(g2m_genes) > 0:
        for gene in g2m_genes:
            save_path = f"{g2m_genes_path}{gene}"
            scv.pl.velocity(adata, gene,
                            add_outline=True, show=False)
            
            plt.savefig(save_path, bbox_inches='tight')

import numpy as np
import pandas as pd
import scvelo as scv
import scanpy as sc

def plot_velocity_coherence(adata, dataset_name, cell_type_key):
    """
    Compute velocity coherence and plot violin plot per cell type.

    Parameters:
    - adata: AnnData object
    - dataset_name: Name of the dataset (used in plot title)
    - cell_type_key: Key in adata.obs where cell types are stored

    Stores velocity coherence in adata.obs["velocity_coherence"] and plots a violin plot per cell type.
    """
    import scvelo as scv
    import scanpy as sc
    import numpy as np

    # Compute velocities if not already computed
    if 'velocity' not in adata.layers:
        scv.tl.velocity(adata)

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

    # Compute product_score
    prod = adata.layers["Ms_delta"] * adata.layers["velocity"]
    adata.layers["product_score"] = prod

    # Compute mean product_score per cell (mean over genes)
    mean_product_score_per_cell = prod.mean(axis=1) # Convert matrix to array if necessary

    # Store in adata.obs["velocity_coherence"]
    adata.obs["velocity_coherence"] = mean_product_score_per_cell

    # Plot violin plot per cell type
    sc.pl.violin(
        adata,
        keys="velocity_coherence",
        groupby=cell_type_key,
        stripplot=False,
        jitter=False,
        xlabel='Cell type',
        ylabel='Velocity coherence',
        #title=f'Velocity Coherence per Cell Type in {dataset_name}',
        show=True
    )

def plot_velocity_coherence_cdf(adata, dataset_name, cell_type_key):
    """
    Compute velocity coherence and plot the empirical CDF per cell type.

    Parameters:
    - adata: AnnData object
    - dataset_name: Name of the dataset (used in plot title)
    - cell_type_key: Key in adata.obs where cell types are stored

    Computes velocity coherence per gene per cell type and plots the empirical CDFs.
    """
    import scvelo as scv
    import scanpy as sc
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.sparse import issparse

    # Ensure velocities are computed
    if 'velocity' not in adata.layers:
        scv.tl.velocity(adata)

    # Compute velocity graph
    scv.tl.velocity_graph(adata)

    # Get transition matrix
    tm = scv.utils.get_transition_matrix(
        adata, use_negative_cosines=True, self_transitions=True
    )
    tm.setdiag(0)

    # Extrapolate Ms
    Ms = adata.layers["Ms"]
    Ms_extrap = tm @ Ms
    adata.layers["Ms_extrap"] = Ms_extrap

    # Compute Ms_delta (empirical displacement)
    Ms_delta = Ms_extrap - Ms
    adata.layers["Ms_delta"] = Ms_delta

    # Compute coherence score (Hadamard product)
    prod = Ms_delta * adata.layers["velocity"]
    adata.layers["coherence_score"] = prod

    # Prepare DataFrames for each cell type
    df_list = []
    cell_types = adata.obs[cell_type_key].unique()

    all_coherence_scores = []  # Collect all coherence scores

    for cell_type in cell_types:
        # Select cells of this cell type
        cells = adata.obs[adata.obs[cell_type_key] == cell_type].index
        adata_cell_type = adata[cells]

        # Compute mean coherence score per gene over cells
        if issparse(adata_cell_type.layers["coherence_score"]):
            mean_coherence_per_gene = np.array(
                adata_cell_type.layers["coherence_score"].mean(axis=0)
            ).flatten()
        else:
            mean_coherence_per_gene = adata_cell_type.layers["coherence_score"].mean(axis=0)

        # Collect all coherence scores for percentile calculation
        all_coherence_scores.extend(mean_coherence_per_gene)

        # Sort the data for CDF
        sorted_data = np.sort(mean_coherence_per_gene)

        # Compute empirical CDF
        cdf = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

        # Create a dataframe
        df = pd.DataFrame({
            'Velocity coherence': sorted_data,
            'CDF': cdf,
            'Cell type': cell_type
        })

        df_list.append(df)

    # Concatenate dataframes
    coherence_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    coherence_df['Dataset'] = dataset_name

    # Compute desired percentiles for x-axis limits
    lower_percentile = np.percentile(all_coherence_scores, 5)
    upper_percentile = np.percentile(all_coherence_scores, 95)

    # Plot the empirical CDFs
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=coherence_df,
        x='Velocity coherence',
        y='CDF',
        hue='Cell type',
        ax=ax,
    )

    # Set x-axis limits based on percentiles
    plt.xlim(lower_percentile, upper_percentile)

    ax.set_title(f'Empirical CDF of Velocity Coherence per Cell Type in {dataset_name}')
    ax.legend()
    plt.tight_layout()
    plt.show()
