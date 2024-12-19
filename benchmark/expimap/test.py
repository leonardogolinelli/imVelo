import scarches as sca
import scanpy as sc
import numpy as np

datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]

# Specify the path to the dataset and model directory
for dataset in datasets:
    print(f"dataset: {dataset}")
    model = "expimap"
    model_path = f"{dataset}/model"

    # Load the saved EXPIMAP model
    adata_path = f"{dataset}/{model}_{dataset}.h5ad"
    adata = sc.read_h5ad(adata_path)

    # Assuming 'MuMs' and other necessary data are in `adata`
    data = adata.obsm["MuMs"]
    adata_expimap = sc.AnnData(X=data)
    adata_expimap.uns["terms"] = adata.uns["terms"].copy()
    adata_expimap.obs = adata.obs.copy()
    mask = adata.varm["I"]
    adata_expimap.varm["I"] = np.concatenate([mask, mask], axis=0)

    # Add a dummy 'study' column to adata_expimap.obs as required by EXPIMAP
    adata_expimap.obs["study"] = "0"

    # Load the saved EXPIMAP model
    intr_cvae = sca.models.EXPIMAP.load(model_path, adata=adata_expimap)
    adata.obsm['z'] = intr_cvae.get_latent(mean=False, only_active=False)

    print(f"building adata gp")
    # Now the model is loaded, and you can perform further analysis or use its methods
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
    adata_gp.write(f"{dataset}/adata_gp.h5ad")