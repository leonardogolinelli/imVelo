import torch
import pandas as pd
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from utils import get_velocity, load_files, return_gnames, fetch_relevant_terms
from typing import Tuple
from joblib import Parallel, delayed
import scanpy as sc
import scvelo as scv
from dataloaders import setup_dataloaders
from plotting import plot_phase_plane
from plotting import compute_celldeviation_maxstatevelo
import os
import seaborn as sns
import mplscience


#!/usr/bin/env python3
def compute_bayes_factors(adata, cell_type_key):
    print("computing bayes factors")
    clusters = pd.unique(adata.obs[cell_type_key])
    terms = adata.uns["terms"]

    bayes_factors = {}
    for query_group in clusters:
        bayes_factors[query_group]={}
        bool_query = adata.obs[cell_type_key] == query_group
        for i, term in enumerate(terms):
            z_query = adata[bool_query].obsm["z"][:,i]
            z_all_other = adata[-bool_query].obsm["z"][:,i]

            numerator = -1 * (z_query.mean(0) - z_all_other.mean(0))
            denominator = z_query.var(0) + z_all_other.var(0)
            denominator = np.sqrt(2*denominator) + 1e-7
            ratio = numerator/denominator

            bayes_factor = np.absolute(np.log(0.5 * special.erfc(ratio)))
            
            bayes_factors[query_group][term] = bayes_factor

    adata.uns["bayes_factors"] = bayes_factors

    return bayes_factors

def plot_bayes_factors(adata, bayes_factors, cell_type_key, top_N, dataset, K, show_plot, save_plot):
    clusters = pd.unique(adata.obs[cell_type_key])
    #top_N = top_N[0]
    for cluster in clusters:
        factor_list = []
        term_list = []
        for term, factor in bayes_factors[cluster].items():
            factor_list.append(factor)
            term_list.append(term)

        
        indices = np.argsort(factor_list)[-top_N:]
        sorted_factors = np.array(factor_list)[indices][::-1]  # Sort descending
        sorted_terms = np.array(term_list)[indices][::-1]

        # Rank terms from 1 to 10, higher factor means lower rank number
        ranks = np.arange(1, top_N+1)

        # Create a new plot for each cluster
        plt.figure(figsize=(10, 5))

        # Plotting each term beside its corresponding point
        for rank, term, factor in zip(ranks, sorted_terms, sorted_factors):
            plt.text(factor, rank, f"{rank}. {term}", verticalalignment='center', horizontalalignment="left")

        plt.xlim(min(min(sorted_factors),2.3)-0.2, max(sorted_factors) + 1)  # Extend x-axis a bit more for text
        plt.ylim(0, top_N+0.5)  # Extend y-axis a bit more for clarity
        plt.gca().invert_yaxis()  # Invert y-axis so rank 1 is at the top
        plt.axvline(x=2.3, color='red', linestyle='--', label='Threshold at 2.3')

        plt.title(f'Top Terms in {cluster} by Bayes Factor')
        plt.xlabel('Absolute Log Bayes Factors')
        plt.ylabel('Rank')
        plt.yticks(ranks, labels=[f"{rank}"for rank in ranks])
        os.makedirs(f"outputs/{dataset}/K{K}/stats/bayes_scores/", exist_ok=True)

        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(f"outputs/{dataset}/K{K}/stats/bayes_scores/scores_{cluster}.png", bbox_inches='tight')

def bayes_factors(adata, cell_type_key, top_N, dataset, K, show_plot, save_plot):
    bayes_factors = compute_bayes_factors(adata, cell_type_key)
    plot_bayes_factors(adata, bayes_factors, cell_type_key, top_N, dataset, K, show_plot, save_plot)

#uncertainty estimation part.. 
def _compute_directional_statistics_tensor(
    tensor: np.ndarray, n_jobs: int, n_cells: int
) -> pd.DataFrame:
    df = pd.DataFrame(index=np.arange(n_cells))
    df["directional_variance"] = np.nan
    df["directional_difference"] = np.nan
    df["directional_cosine_sim_variance"] = np.nan
    df["directional_cosine_sim_difference"] = np.nan
    df["directional_cosine_sim_mean"] = np.nan
    print(n_jobs)
    results = Parallel(n_jobs=n_jobs, verbose=3)(
        delayed(_directional_statistics_per_cell)(tensor[:, cell_index, :])
        for cell_index in range(n_cells)
    )
    # cells by samples
    cosine_sims = np.stack([results[i][0] for i in range(n_cells)])
    df.loc[:, "directional_cosine_sim_variance"] = [
        results[i][1] for i in range(n_cells)
    ]
    df.loc[:, "directional_cosine_sim_difference"] = [
        results[i][2] for i in range(n_cells)
    ]
    df.loc[:, "directional_variance"] = [results[i][3] for i in range(n_cells)]
    df.loc[:, "directional_difference"] = [results[i][4] for i in range(n_cells)]
    df.loc[:, "directional_cosine_sim_mean"] = [results[i][5] for i in range(n_cells)]

    return df, cosine_sims

def _directional_statistics_per_cell(
    tensor: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Internal function for parallelization.

    Parameters
    ----------
    tensor
        Shape of samples by genes for a given cell.
    """
    n_samples = tensor.shape[0]
    # over samples axis
    mean_velocity_of_cell = tensor.mean(0)
    cosine_sims = [
        _cosine_sim(tensor[i, :], mean_velocity_of_cell) for i in range(n_samples-1)
    ]

    angle_samples = [np.arccos(el) for el in cosine_sims]
    return (
        cosine_sims,
        np.var(cosine_sims),
        np.percentile(cosine_sims, 95) - np.percentile(cosine_sims, 5),
        np.var(angle_samples),
        np.percentile(angle_samples, 95) - np.percentile(angle_samples, 5),
        np.mean(cosine_sims),
    )

def _centered_unit_vector(vector: np.ndarray) -> np.ndarray:
    """Returns the centered unit vector of the vector."""
    vector = vector - np.mean(vector)
    return vector / np.linalg.norm(vector)

def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Returns cosine similarity of the vectors."""
    v1_u = _centered_unit_vector(v1)
    v2_u = _centered_unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)

def get_directional_uncertainty(
        adata,
        model,
        dataloader,
        n_samples: int = 25,
        n_jobs: int = 1,
        show: bool = False,
        dataset: str = "pancreas",
        K: int = 10,
    ):
        print("computing directional uncertainty..")

        velocities_all = get_velocity(
                adata=adata,
                model=model,
                n_samples=n_samples,
                full_data_loader=dataloader,
                return_mean = False)

        df, cosine_sims = _compute_directional_statistics_tensor(
            tensor=velocities_all, n_jobs=n_jobs, n_cells=adata.n_obs
        )
        df.index = adata.obs_names

        for c in df.columns:
            adata.obs[c] = np.log10(df[c].values)

        if not dataset == "dentategyrus_lamanno":
            sc.pl.umap(
                adata, 
                color="directional_cosine_sim_variance",
                cmap="Greys",
                vmin="p1",
                vmax="p99",
                show=show,
            )

        else:
            sc.pl.umap(
                adata, 
                color="directional_cosine_sim_variance",
                cmap="Greys",
                vmin="p1",
                vmax="p99",
                show=show,
            )


        path = f"outputs/{dataset}/K{K}/stats/uncertainty/intrinsic_uncertainty.png"
        os.makedirs(f"outputs/{dataset}/K{K}/stats/uncertainty/", exist_ok=True)
        plt.savefig(path, bbox_inches='tight')  # Save using plt.savefig
        plt.close()

def compute_extrinisic_uncertainty(adata,
                                    model,
                                    dataset,
                                    K,
                                    dataloader,
                                    n_samples,
                                    show=False) -> pd.DataFrame:
    from contextlib import redirect_stdout
    import io
    print("computing extrinsic uncertainty..")

    sc.pp.neighbors(adata)
    #sc.tl.umap(adata)

    extrapolated_cells_list = []
    for i in range(n_samples):
        with io.StringIO() as buf, redirect_stdout(buf):
            vkey = f"velocities_miVelo_{i}"
            v = get_velocity(
                adata=adata,
                model=model,
                n_samples=1,
                full_data_loader=dataloader,
                return_mean = True
            )
            v = v[-1]
            adata.layers[vkey] = v
            scv.tl.velocity_graph(adata, vkey=vkey, sqrt_transform=False, approx=True)
            t_mat = scv.utils.get_transition_matrix(
                adata, vkey=vkey, self_transitions=True, use_negative_cosines=True
            )
            extrapolated_cells = np.asarray(t_mat @ adata.layers["Ms"])
            extrapolated_cells_list.append(extrapolated_cells)
    extrapolated_cells = np.stack(extrapolated_cells_list)
    ext_uncertainty_df, _ = _compute_directional_statistics_tensor(extrapolated_cells, n_jobs=1, n_cells=adata.n_obs)

    for c in ext_uncertainty_df.columns:
        adata.obs[c + "_extrinisic"] = np.log10(ext_uncertainty_df[c].values)

    if not dataset == "dentategyrus_lamanno":
        sc.pl.umap(
            adata, 
            color="directional_cosine_sim_variance_extrinisic",
            vmin="p1", 
            vmax="p99",
            show=show,
            )
    else:
        sc.pl.umap(
            adata, 
            color="directional_cosine_sim_variance_extrinisic",
            vmin="p1", 
            vmax="p99",
            show=show,
        )

    path = f"outputs/{dataset}/K{K}/stats/uncertainty/extrinsic_uncertainty.png"
    
    plt.savefig(path, bbox_inches='tight') # Save using plt.savefig
    plt.close()


def estimate_uncertainty(
        adata,
        model,
        batch_size=256,
        n_jobs=1,
        show=False,
        dataset="pancreas",
        K=10,

):
    
    model = model.cpu()
    with torch.no_grad():
        _, _, full_dl = setup_dataloaders(adata, batch_size=batch_size, train_size=.8, split_data=False)
        get_directional_uncertainty(adata, model, n_samples=50, n_jobs=n_jobs, 
                                                    dataloader=full_dl,show=show,
                                                    dataset=dataset, K=K)
        compute_extrinisic_uncertainty(adata, model, n_samples=25, 
                                                    dataloader=full_dl,show=show,
                                                    dataset=dataset, K=K)
    

def compute_sign_variance(adata, model, n_samples=50, dataloader=None):
    # Use your method to get velocities with multiple samples
    v_stack = get_velocity(
        adata=adata, 
        model=model, 
        n_samples=n_samples, 
        full_data_loader=dataloader, 
        return_mean=False
    )

    pos_freq = (v_stack >= 0).mean(0)
    adata.layers["velocity"] = v_stack.mean(0)

    var_freq = pos_freq * (1 - pos_freq)
    adata.obs["sign_var"] = var_freq.mean(1)

    adata.layers["sign_var"] = var_freq
    adata.layers["variance"] = v_stack.var(0)


def compute_sign_var_score(adata, labels_key, model, n_samples=50, dataloader=None):
    # Reuse the modified compute_sign_variance
    compute_sign_variance(adata, model, n_samples=n_samples, dataloader=dataloader)

    sign_var_df = pd.DataFrame(adata.layers["sign_var"], index=adata.obs_names)
    expr_df = pd.DataFrame(adata.layers["Ms"], index=adata.obs_names)

    # Calculate the product of sign variance and absolute expression
    prod_df = sign_var_df * np.abs(expr_df)
    prod_df[labels_key] = adata.obs[labels_key]
    prod_df = prod_df.groupby(labels_key).mean()

    sign_var_df[labels_key] = adata.obs[labels_key]
    sign_var_df = sign_var_df.groupby(labels_key).mean()

    return sign_var_df.mean(0)


def gene_rank(adata, vkey="velocity"):
    from scipy.stats import rankdata

    # Calculate velocity graph
    scv.tl.velocity_graph(adata, vkey=vkey)

    # Generate the transition matrix
    tm = scv.utils.get_transition_matrix(
        adata, vkey=vkey, use_negative_cosines=True, self_transitions=True
    )
    tm.setdiag(0)

    # Extrapolate RNA counts
    adata.layers["Ms_extrap"] = tm @ adata.layers["Ms"]
    adata.layers["Ms_delta"] = adata.layers["Ms_extrap"] - adata.layers["Ms"]

    # Product score: delta RNA counts multiplied by velocities
    prod = adata.layers["Ms_delta"] * adata.layers[vkey]
    ranked = rankdata(prod, axis=1)

    adata.layers["product_score"] = prod
    adata.layers["ranked_score"] = ranked



import scanpy as sc
import pandas as pd

def compute_and_plot_velocity_coherence(
    adata, 
    model, 
    dataloader,
    dataset_name: str, 
    cell_type_key: str, 
    save_figures: bool = False, 
    fig_dir: str = "."
):
    # Compute sign variance score
    compute_sign_var_score(adata, cell_type_key, model, n_samples=50, dataloader=dataloader)
    
    # Rank genes based on velocity coherence
    gene_rank(adata)
    
    # Initialize list for coherence data
    coherence_data = []

    # Get unique cell types from the given key in adata.obs
    clusters = adata.obs[cell_type_key].unique()

    for cluster in clusters:
        cluster_cells = adata.obs.query(f"{cell_type_key} == '{cluster}'").index
        cluster_data = adata[cluster_cells]
        
        # Per cell and per gene velocity coherence
        cluster_data.obs[f'mean_product_score_per_cell_{cluster.lower()}'] = cluster_data.layers['product_score'].mean(axis=1)
        cluster_data.var[f'mean_product_score_per_gene_{cluster.lower()}'] = cluster_data.layers['product_score'].mean(axis=0)

        # Store the coherence values in adata.obs for the whole dataset
        adata.obs.loc[cluster_cells, 'velocity_coherence'] = cluster_data.var[f'mean_product_score_per_gene_{cluster.lower()}'].mean()

    # Now plot using Scanpy's violin function
    sc.pl.violin(
        adata, 
        keys="velocity_coherence", 
        groupby=cell_type_key, 
        jitter=True, 
        rotation=90, 
        save=f"_{dataset_name}_velocity_coherence.png" if save_figures else None
    )

    # Optionally save the figures if required
    if save_figures:
        save_path = os.path.join(fig_dir, dataset_name)
        os.makedirs(save_path, exist_ok=True)
        sc.pl.violin(
            adata, 
            keys="velocity_coherence", 
            groupby=cell_type_key, 
            jitter=True, 
            rotation=90, 
            save=os.path.join(save_path, f"velocity_coherence_{dataset_name}.svg")
        )



def deg_genes(adata, dataset, K, cell_type_key="clusters", n_deg_rows=5):
    print(f"Computing DEG genes..")
    import os
    wilcoxon_path = f"outputs/{dataset}/K{K}/deg_genes/wilcoxon/"
    phase_plane_path = f"outputs/{dataset}/K{K}/deg_genes/phase_plane/"
    os.makedirs(wilcoxon_path, exist_ok=True)
    os.makedirs(phase_plane_path, exist_ok=True)
    if dataset == "forebrain":
        adata.obs["Clusters"] = pd.Series(adata.obs["Clusters"], dtype="category")

    for layer in ["Ms", "Mu"]:
        sc.tl.rank_genes_groups(adata, groupby=cell_type_key, layer=layer, method='wilcoxon')
        sc.pl.rank_genes_groups(adata, ncols=2, show=False)
        save_path = f"{wilcoxon_path}wilcoxon_on_{layer}.png"
        plt.tight_layout()
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
        plt.tight_layout()
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
            plot_phase_plane(adata, gene, dataset, K, save_path=save_path, show_plot=False, cell_type_key=cell_type_key)
        i+=1
    
    return adata

def compute_scvelo_metrics(adata, dataset, K, show=False, cell_type_key="clusters"):
    print("computing scvelo metrics..")
    scv.tl.velocity_confidence(adata)
    #if dataset not in ["dentategyrus_lamanno", "dentategyrus_lamanno_P0", "dentategyrus_lamanno_P5", "gastrulation_erythroid"]:
    scv.tl.velocity_pseudotime(adata)

    confidence_path = f"outputs/{dataset}/K{K}/scvelo_metrics/confidence/"
    s_genes_path = f"outputs/{dataset}/K{K}/scvelo_metrics/s_genes/"
    g2m_genes_path = f"outputs/{dataset}/K{K}/scvelo_metrics/g2m_genes/"
    for path in [confidence_path, s_genes_path, g2m_genes_path]:
        os.makedirs(path, exist_ok = True)

    keys = [cell_type_key, 'velocity_length', 'velocity_confidence', 'velocity_pseudotime']
    cmaps = [None, 'coolwarm', 'coolwarm', 'gnuplot', 'gnuplot']
    #if dataset in ["dentategyrus_lamanno", "dentategyrus_lamanno_P0", "dentategyrus_lamanno_P5", "gastrulation_erythroid"]:
    #    keys = keys[:-2]         
    for i, key in enumerate(keys):
        sc.pl.umap(adata, color=key, color_map=cmaps[i], show=show)
        plt.tight_layout()
        plt.savefig(f"{confidence_path}{key}.png", bbox_inches='tight')

    for i, key in enumerate(keys):
        scv.pl.velocity_embedding_stream(adata, color=key, color_map=cmaps[i], show=show)
        plt.tight_layout()
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
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')

    if len(g2m_genes) > 0:
        for gene in g2m_genes:
            save_path = f"{g2m_genes_path}{gene}"
            scv.pl.velocity(adata, gene,
                            add_outline=True, show=False)
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')








if __name__ == '__main__':
    adata, model = load_files(dataset="pancreas", K=11)
    """bayes_factors(adata, cell_type_key="clusters", top_N=30, dataset="pancreas", 
                  K=11, show_plot=False, save_plot=True)"""
    """_, _, full_dl = setup_dataloaders(adata, batch_size=256, train_size=.8, split_data=False)
    get_directional_uncertainty(adata, model, n_samples=50, n_jobs= 8, 
                                                dataloader=full_dl,show=False,
                                                dataset="pancreas", K=10)
    compute_extrinisic_uncertainty(adata, model, n_samples=50, 
                                                dataloader=full_dl,show=False,
                                                dataset="pancreas", K=10)"""
    compute_scvelo_metrics(adata, "pancreas", 11, show=False)


    