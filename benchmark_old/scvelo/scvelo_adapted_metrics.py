import os, sys
current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import scvelo as scv
from scvelo_adapted_plots import plot_phase_plane


def deg_genes(adata, dataset, cell_type_key="clusters", n_deg_rows=5):
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
            plot_phase_plane(adata, gene, dataset, save_path=save_path, show_plot=False, cell_type_key=cell_type_key)
        i+=1
    
    return adata


def compute_scvelo_metrics(adata, dataset, show=False, cell_type_key="clusters"):
    scv.tl.velocity_confidence(adata)
    scv.tl.velocity_pseudotime(adata)
    #scv.tl.latent_time(adata)


    confidence_path = f"{dataset}/scvelo_metrics/confidence/"
    s_genes_path = f"{dataset}/scvelo_metrics/s_genes/"
    g2m_genes_path = f"{dataset}/scvelo_metrics/g2m_genes/"
    for path in [confidence_path, s_genes_path, g2m_genes_path]:
        os.makedirs(path, exist_ok = True)

    keys = cell_type_key, 'velocity_length', 'velocity_confidence', 'velocity_pseudotime', 'latent_time'
    cmaps = [None, 'coolwarm', 'coolwarm', 'gnuplot', 'gnuplot']
    for i, key in enumerate(keys[:-2]):
        sc.pl.umap(adata, color=key, color_map=cmaps[i], show=show)
        
        plt.savefig(f"{confidence_path}{key}.png", bbox_inches='tight')

    for i, key in enumerate(keys[:-2]):
        scv.pl.velocity_embedding_stream(adata, color=key, color_map=cmaps[i], show=show)
        
        plt.savefig(f"{confidence_path}{key}_stream.png", bbox_inches='tight')
    
    scv.tl.rank_velocity_genes(adata, groupby=cell_type_key, min_corr=.3)
    s_genes, g2m_genes = scv.utils.get_phase_marker_genes(adata)
    s_genes = scv.get_df(adata[:, s_genes], 'spearmans_score', sort_values=True).index
    g2m_genes = scv.get_df(adata[:, g2m_genes], 'spearmans_score', sort_values=True).index

    for gene in s_genes:
        save_path = f"{s_genes_path}{gene}"
        scv.pl.velocity(adata, gene, 
                        add_outline=True, show=False) 
        plt.savefig(save_path)

    for gene in g2m_genes:
        save_path = f"{g2m_genes_path}{gene}"
        scv.pl.velocity(adata, gene,
                        add_outline=True, show=False)
        plt.savefig(save_path)
