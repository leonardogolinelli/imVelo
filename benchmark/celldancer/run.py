import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import os
import gc
from benchmark_preprocessing import preprocess

datasets = ["forebrain", "pancreas", "gastrulation_erythroid",  "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters","celltype",  "clusters"]
generate_adata = True #celldancer environment
adata_to_df = False
downstream = False
n_highly_var_genes = 2000
smooth_k = 30

#datasets = ["forebrain"]
#cell_type_keys = ["Clusters"]

#celldancer environment path
if generate_adata:
    for dataset, cell_type_key in zip(datasets, cell_type_keys):
        print(f"processing dataset: {dataset}")
        os.makedirs(dataset, exist_ok=True)
        adata = preprocess(dataset, cell_type_key, n_highly_var_genes, smooth_k)
        adata.write_h5ad(f"{dataset}/celldancer_{dataset}")

if adata_to_df:
    import celldancer as cd
    import celldancer.utilities as cdutil
    for dataset, cell_type_key in zip(datasets, cell_type_keys):
        cdutil.adata_to_df_with_embed(adata,
                                us_para=['Mu','Ms'],
                                cell_type_para=cell_type_key,
                                embed_para='X_umap',
                                save_path=f"{dataset}/cd_df.csv")
        
        df = pd.read_csv(f"{dataset}/cd_df.csv")
        df["cellIndex"] = df.cellID
        
        loss_df_velo, cd_df_velo = cd.velocity(df,\
                    gene_list=None,\
                    permutation_ratio=0.125,\
                    n_jobs=1,
                    speed_up=False)

        cd_df_velo.to_csv(f"{dataset}/cd_df_velo.csv")
        print(f"cd_df succesfully written to path {dataset}/cd_df_velo.csv")
        loss_df_velo.to_csv(f"{dataset}/loss_df_velo.csv")
        print(f"loss_df succesfully written to path {dataset}/loss_df_velo.csv")

        # Initialize velocity layers with zeros
        adata.layers["velocity"] = np.zeros(adata.shape)
        adata.layers["velocity_u"] = np.zeros(adata.shape)

        #cd_df_velo = pd.read_csv(f"{dataset}/cd_df_velo.csv")

        # Iterate over the DataFrame and update the layers
        for i, cell_idx in enumerate(cd_df_velo.cellIndex):
            gene_name = cd_df_velo.gene_name[i]
            alpha = cd_df_velo.alpha[i]
            beta = cd_df_velo.beta[i]
            gamma = cd_df_velo.gamma[i]
            unspliced = cd_df_velo.unsplice[i]
            spliced = cd_df_velo.splice[i]
            #pseudotime = cellDancer_df_u_s.pseudotime[i]

            velocity_u = alpha - beta*unspliced
            velocity = beta*unspliced - gamma*spliced
            
            # Update the velocity layers correctly
            adata.layers["velocity"][cell_idx, adata.var_names.get_loc(gene_name)] = velocity
            adata.layers["velocity_u"][cell_idx, adata.var_names.get_loc(gene_name)] = velocity_u
            #adata.obs["pseudotime"] = pseudotime
            
            # Print progress every 1000 iterations
            if i % 1000 == 0:
                print(f"{np.round(i/len(cd_df_velo)*100,2)}%")
        print(adata)
        # Save the updated AnnData object
        gc.collect()
        adata.write_h5ad(f"{dataset}/cd_{dataset}.h5ad")

downstream = False
#scvelo environment part
if downstream:
    datasets = ["dentategyrus_lamanno"]
    cell_type_keys = ["clusters"]
    import scvelo as scv
    from celldancer_adapted_metrics import compute_scvelo_metrics, deg_genes
    from celldancer_adapted_plots import plot_important_genes
    for dataset, cell_type_key in zip(datasets, cell_type_keys):
        adata_path = os.path.expanduser(f"~/top_adatas/cd_{dataset}.h5ad")
        print(f"processing dataset: {dataset}")
        os.makedirs(dataset, exist_ok=True)
        adata = sc.read_h5ad(f"{dataset}/cd_{dataset}.h5ad")
        if dataset == "gastrulation_erythroid":
            adata.layers["velocity"] *=-1
            adata.layers["velocity_u"] *=-1
        sc.pp.neighbors(adata)
        scv.tl.velocity_graph(adata)
        compute_scvelo_metrics(adata, dataset, False, cell_type_key)
        adata.write_h5ad(adata_path)
        deg_genes(adata, dataset, cell_type_key=cell_type_key, n_deg_rows=1)
        adata.write_h5ad(adata_path)
        plot_important_genes(adata, dataset, cell_type_key)

        # Clean up
        del adata
        gc.collect()

