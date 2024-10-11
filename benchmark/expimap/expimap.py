import numpy as np
import pandas as pd
import scvelo as scv
import scanpy as sc
import os 
import scarches as sca
import matplotlib.pyplot as plt
import torch


sc.set_figure_params(frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)


datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters", "celltype", "clusters"]

for dataset, cell_type_key in zip(datasets, cell_type_keys):
    print(f"running expimap on dataset: {dataset}")
    os.makedirs(dataset, exist_ok=True)

    adata_path = f"../imVelo/{dataset}/imVelo_{dataset}.h5ad"
    adata = sc.read_h5ad(adata_path)

    data = adata.obsm["MuMs"]
    adata_expimap = sc.AnnData(X=data)
    adata_expimap.uns["terms"] = adata.uns["terms"].copy()
    adata_expimap.obs = adata.obs.copy()
    assert adata.obs[cell_type_key].notna().any()
    assert adata_expimap.obs[cell_type_key].notna().any()

    mask = adata.varm["I"]
    adata_expimap.varm["I"] = np.concatenate([mask,mask],axis=0)

    print(f"Hard mask shape: {adata_expimap.varm['I'].shape}")
    adata_expimap.obs["study"] = "0"
    intr_cvae = sca.models.EXPIMAP(
        adata=adata_expimap,
        condition_key='study',
        hidden_layer_sizes=[512, 512, 512],
        recon_loss='mse',
        soft_mask = False,
        n_ext = 0,
        use_hsic=False,
    )

    ALPHA = .7
    early_stopping_kwargs = {
        "early_stopping_metric": "val_unweighted_loss", # val_unweighted_loss
        "threshold": 0,
        "patience": 50,
        "reduce_lr": False,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }

    intr_cvae.train(
        n_epochs=1000,
        alpha_epoch_anneal=500,
        alpha=ALPHA,
        alpha_kl=1e-5,
        weight_decay=1e-4,
        early_stopping_kwargs=early_stopping_kwargs,
        use_early_stopping=False,
        monitor_only_val=False,
        seed=2020,
        train_frac=.9,
        print_stats=True
    )

    adata.obsm['z'] = intr_cvae.get_latent(mean=False, only_active=True)
    sc.pp.neighbors(adata, use_rep='z')
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=[cell_type_key], frameon=False)
    
    plt.savefig(f"{dataset}/z.png", bbox_inches="tight")
    intr_cvae.latent_directions(adata=adata_expimap)
    intr_cvae.latent_enrich(groups=cell_type_key, use_directions=False, adata=adata_expimap)

    adata.write_h5ad(f"{dataset}/expimap_{dataset}.h5ad")
    intr_cvae.save(f"{dataset}/model")
