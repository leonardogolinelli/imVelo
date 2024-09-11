import scanpy as sc
import scvelo as scv
import scarches as sca
import numpy as np
import gdown
import os
import pandas as pd
from utils import rename_duplicate_terms, manifold_and_neighbors
import seaborn as sns

def setup_adata(dataset_name='pancreas', 
                preproc_adata=True,
                smooth_k=30,
                n_highly_var_genes=2000,
                cell_type_key = "clusters",
                save_umap = True,
                show_umap = False,
                unspliced_key = "unspliced",
                spliced_key = "spliced",
                filter_on_r2=False,
                knn_rep=True,
                n_components=3,
                n_knn_search=10,
                best_key=None,
                K = None,
                ve_layer="6"):
    
    if preproc_adata:
        if (unspliced_key != "unspliced") and (spliced_key != "spliced"):
            adata.rename({unspliced_key: 'unspliced'})
            adata.rename({spliced_key: 'spliced'})
    
        #os makedir annotation

        #Download dataset
        output_adata_path = f"outputs/adata_preproc/adata_{dataset_name}_Ksmooth{smooth_k}.h5ad"
        output_plot_path = f"outputs/adata_preproc/umap_{dataset_name}_Ksmooth{smooth_k}.png"
        
        try:
            os.remove(output_adata_path)
        except:
            print("")

        if dataset_name == "pancreas":
            adata = scv.datasets.pancreas()
        elif dataset_name == "gastrulation_erythroid":
            adata = scv.datasets.gastrulation_erythroid()
            #adata = adata[adata.obs["stage"]=="E8.25"].copy()
            #dataset_name +="_E8.25"
        elif dataset_name == "forebrain":
            adata = scv.datasets.forebrain()
        elif dataset_name == "dentategyrus_lamanno":
            adata = scv.datasets.dentategyrus_lamanno()
        elif dataset_name == "dentategyrus_lamanno_P0":
            adata = scv.datasets.dentategyrus_lamanno()
            adata = adata[adata.obs["Age"] == "P0"].copy()
        elif dataset_name == "dentategyrus_lamanno_P5":
            adata = scv.datasets.dentategyrus_lamanno()
            adata = adata[adata.obs["Age"] == "P5"].copy()

        adata.obs[cell_type_key] = [str(cat) for cat in list(adata.obs[cell_type_key])]
        adata.obs[cell_type_key] = pd.Series(adata.obs[cell_type_key], dtype="category")
        unique_categories = adata.obs[cell_type_key].cat.categories
        rgb_colors = sns.color_palette("tab20", len(unique_categories))
        hex_colors = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b in rgb_colors]
        adata.uns[f"{cell_type_key}_colors"] = hex_colors
        print(dataset_name)
        adata.layers['counts_unspliced'] = adata.layers["unspliced"].copy()
        adata.layers['counts_spliced'] = adata.layers["spliced"].copy()

        reactome_path = "inputs/annotations/reactome.gmt"
        panglao_path = "inputs/annotations/panglao.gmt"

        if not os.path.exists(reactome_path):
            file_id = '1b8996Lmrt-7rws3nzD1KoGmR_I3LXqJV'
            output = reactome_path
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output, quiet=False)

        if not os.path.exists(panglao_path):
            file_id = '1q-UjBRMP558MBxJGXpH7LEfLpCEGvBSC'
            output = panglao_path
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output, quiet=False)
            
        #Annotate part 1
        sca.utils.add_annotations(adata, ["inputs/annotations/panglao.gmt", "inputs/annotations/reactome.gmt"], min_genes=12, clean=True) #keep terms with at least 12 genes
        adata._inplace_subset_var(adata.varm['I'].sum(1) > 0) #remove genes that are not annotated

        #Filter and normalize
        #scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=n_highly_var_genes) #filter and normalize
        #scv.pp.moments(adata, n_neighbors=smooth_k)
        if dataset_name in ["forebrain"]:
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        elif dataset_name in ["dentategyrus_lamanno", "dentategyrus_lamanno_P0", "dentategyrus_lamanno_P5"]:
            adata.obsm["X_umap"] = adata.obsm["X_tsne"].copy()

        if filter_on_r2:
            scv.tl.velocity(adata, mode="deterministic")

            adata = adata[
                :, np.logical_and(adata.var.velocity_r2 > 0, adata.var.velocity_gamma > 0)
            ].copy()
            adata = adata[:, adata.var.velocity_genes].copy()
        print(f"adata shape preproc: {adata.shape}")

        #Annotate part 2
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=n_highly_var_genes) #filter and normalize
        scv.pp.moments(adata, n_neighbors=smooth_k)
        print(f"adata shape preproc: {adata.shape}")
        select_terms = adata.varm['I'].sum(0) > 12 #remove the terms that contain less than 12 genes AFTER gene filtering
        adata.uns['terms'] = np.array(adata.uns['terms'])[select_terms].tolist()
        adata.varm['I'] = adata.varm['I'][:, select_terms]
        adata._inplace_subset_var(adata.varm['I'].sum(1) > 0) #remove the genes that were only present in terms that have been filtered out in the select_term part
        rename_duplicate_terms(adata)
        mask = adata.varm['I']
        mask_double = np.concatenate([mask,mask],axis=0)
        adata.uns["mask"] = mask_double
        adata.var_names = [name.lower().capitalize() for name in list(adata.var_names)]
        print(f"adata shape preproc: {adata.shape}")
        adata.obsm["MuMs"] = np.concatenate([adata.layers["Mu"], adata.layers["Ms"]], axis=1)
        distances, indices = manifold_and_neighbors(adata, n_components, n_knn_search, dataset_name, K, knn_rep, best_key, ve_layer)
        adata.uns["distances"] = distances
        adata.uns["indices"] = indices

        print(f"Hard mask shape: {adata.varm['I'].shape}")
        #adata.write_h5ad(output_adata_path)

    else:
        adata_path = f"outputs/adata_preproc/adata_{dataset_name}_Ksmooth{smooth_k}.h5ad"
        adata = sc.read_h5ad(adata_path)
    
    return adata

if __name__ == "__main__":
    print(setup_adata())
