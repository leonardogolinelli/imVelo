import numpy as np
import pandas as pd
import scanpy as sc
import os
import gc

datasets = ["forebrain", "pancreas", "gastrulation_erythroid",  "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters","celltype",  "clusters"]
generate_adata = False #celldancer environment
compute_velocity = False
downstream = True

if compute_velocity:
        import celldancer as cd
        import celldancer.utilities as cdutil
        for dataset, cell_type_key in zip(datasets, cell_type_keys):
            adata_path = f"../imVelo/{dataset}/imVelo_{dataset}.h5ad"
            adata = sc.read_h5ad(adata_path)
            os.makedirs(f"{dataset}", exist_ok=True)
            if not os.path.isdir(f"{dataset}/cd_df.csv"):
                cdutil.adata_to_df_with_embed(adata,
                                        us_para=['Mu','Ms'],
                                        cell_type_para=cell_type_key,
                                        embed_para='X_umap',
                                        save_path=f"{dataset}/cd_df.csv")
            
            df = pd.read_csv(f"{dataset}/cd_df.csv")
            df["cellIndex"] = df.cellID
            permutation_ratio = 0.125 #if not dataset == "dentategyrus_lamanno_P5" else 0.05
            loss_df_velo, cd_df_velo = cd.velocity(df,\
                        gene_list=None,\
                        permutation_ratio=permutation_ratio,\
                        n_jobs=5,
                        speed_up=False)

            cd_df_velo.to_csv(f"{dataset}/cd_df_velo.csv")
            print(f"cd_df succesfully written to path {dataset}/cd_df_velo.csv")
            loss_df_velo.to_csv(f"{dataset}/loss_df_velo.csv")
            print(f"loss_df succesfully written to path {dataset}/loss_df_velo.csv")

            # Initialize velocity layers with zeros
            adata.layers["velocity"] = np.zeros(adata.shape)
            adata.layers["velocity_u"] = np.zeros(adata.shape)

            #cd_df_velo = pd.read_csv(f"{dataset}/cd_df_velo.csv")
            cd_df_velo = pd.read_csv(f"{dataset}/cd_df_velo.csv")
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
            adata.write_h5ad(f"{dataset}/celldancer_{dataset}.h5ad")

#scvelo environment part
if downstream:
    import scvelo as scv
    from celldancer_adapted_metrics import compute_scvelo_metrics, deg_genes
    from celldancer_adapted_plots import plot_important_genes
    datasets = ["pancreas"]
    cell_type_keys = ["clusters"]
    for dataset, cell_type_key in zip(datasets, cell_type_keys):
        print(f"processing dataset: {dataset}")
        os.makedirs(dataset, exist_ok=True)
        adata = sc.read_h5ad(f"{dataset}/celldancer_{dataset}.h5ad")
        if dataset == "gastrulation_erythroid" or dataset == "pancreas":
            adata.layers["velocity"] *=-1
            adata.layers["velocity_u"] *=-1
        sc.pp.neighbors(adata)
        scv.tl.velocity_graph(adata)
        compute_scvelo_metrics(adata, dataset, False, cell_type_key)
        adata.write_h5ad(f"{dataset}/celldancer_{dataset}_final.h5ad")
        deg_genes(adata, dataset, cell_type_key=cell_type_key, n_deg_rows=1)
        adata.write_h5ad(f"{dataset}/celldancer_{dataset}_final.h5ad")
        plot_important_genes(adata, dataset, cell_type_key)

        # Clean up
        del adata
        gc.collect()