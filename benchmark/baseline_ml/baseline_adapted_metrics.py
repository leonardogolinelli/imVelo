import os, sys
current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import scvelo as scv
from baseline_adapted_plots import plot_phase_plane

def deg_wilcoxon(adata, dataset, cell_type_key="clusters"):
    wilcoxon_path = f"{dataset}/deg_genes/wilcoxon/"
    phase_plane_path = f"{dataset}/deg_genes/phase_plane/"
    os.makedirs(wilcoxon_path, exist_ok=True)
    os.makedirs(phase_plane_path, exist_ok=True)

    for layer in ["Ms", "Mu", "velocity", "velocity_u"]:
        sc.tl.rank_genes_groups(adata, groupby=cell_type_key, layer=layer, method='wilcoxon')
        sc.pl.rank_genes_groups(adata, ncols=2, show=False)
        save_path = f"{wilcoxon_path}wilcoxon_on_{layer}.png"
        
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        
        # Extract DEGs
        degs = pd.DataFrame()
        for cluster in adata.obs[cell_type_key].unique():
            degs[cluster] = pd.DataFrame(adata.uns['rank_genes_groups']['names'])[cluster]
        
        # Save the DataFrame in adata.uns
        adata.uns[f'deg_genes_{layer}'] = degs

    deg_df = adata.uns[f"deg_genes_velocity"]
    i=0
    while i < 10:
        for gene in deg_df.iloc[i]:
            save_path = f"{phase_plane_path}{gene}.png"
            plot_phase_plane(adata, gene, save_path=save_path, show_plot=False, cell_type_key=cell_type_key)
        i+=1
    
    return adata


def compute_scvelo_metrics(adata, dataset, show=False, cell_type_key="clusters"):
    scv.tl.velocity_confidence(adata)
    scv.tl.velocity_pseudotime(adata)

    confidence_path = f"{dataset}/scvelo_metrics/confidence/"
    s_genes_path = f"{dataset}/scvelo_metrics/s_genes/"
    g2m_genes_path = f"{dataset}/scvelo_metrics/g2m_genes/"
    for path in [confidence_path, s_genes_path, g2m_genes_path]:
        os.makedirs(path, exist_ok = True)

    keys = 'clusters', 'velocity_length', 'velocity_confidence', 'velocity_pseudotime'
    cmaps = [None, 'coolwarm', 'coolwarm', 'gnuplot']
    for i, key in enumerate(keys):
        sc.pl.umap(adata, color=key, color_map=cmaps[i], show=show)
        
        plt.savefig(f"{confidence_path}{key}.png", bbox_inches='tight')

    for i, key in enumerate(keys):
        scv.pl.velocity_embedding_stream(adata, color=key, color_map=cmaps[i], show=show)
        
        plt.savefig(f"{confidence_path}{key}_stream.png", bbox_inches='tight')
    
    deg_wilcoxon(adata, dataset, cell_type_key=cell_type_key)

    scv.tl.rank_velocity_genes(adata, groupby=cell_type_key, min_corr=.3)
    s_genes, g2m_genes = scv.utils.get_phase_marker_genes(adata)
    s_genes = scv.get_df(adata[:, s_genes], 'spearmans_score', sort_values=True).index
    g2m_genes = scv.get_df(adata[:, g2m_genes], 'spearmans_score', sort_values=True).index

    for gene in s_genes:
        save_path = f"{s_genes_path}{gene}"
        scv.pl.velocity(adata, gene, 
                        add_outline=True, show=False) 
        
        plt.savefig(save_path, bbox_inches="tight")

    for gene in g2m_genes:
        save_path = f"{g2m_genes_path}{gene}"
        scv.pl.velocity(adata, gene,
                        add_outline=True, show=False)
        
        plt.savefig(save_path, bbox_inches="tight")
