import scanpy as sc

import scib_metrics.benchmark
from scib_metrics.benchmark import Benchmarker

from functions import * 

models = ["expimap","velovi", "velovi_filtered", "ivelo", "ivelo_filtered", "imVelo"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters", "celltype", "clusters"]
"""
models = ["expimap","velovi", "velovi_filtered", "ivelo", "ivelo_filtered", "imVelo"]
datasets = ["forebrain", "pancreas", "gastrulation_erythroid"]
cell_type_keys = ["Clusters", "clusters", "celltype", "clusters"]

for dataset, cell_type_key in zip(datasets, cell_type_keys):
    adata_latent = None
    print(f"loading dataset: {dataset}")
    for model in models:
        print(f"loading model: {model}")
        adata_path = f"../{model}/{dataset}/{model}_{dataset}.h5ad"
        adata = sc.read_h5ad(adata_path)
        if not adata_latent:
            adata_latent = sc.AnnData(X=adata.obsm["z"])
            adata_latent.obs[cell_type_key] = adata.obs[cell_type_key].copy().tolist()
        z = adata.obsm["z"]
        adata_latent.obsm[f"{dataset}-{model}"] = z

    adata_latent.write_h5ad(f"adata_latent_{dataset}.h5ad")

"""

metric_names = ["isolated_labels","nmi_ari_cluster_labels_kmeans", "silhouette_label", "klisi_knn"]
#metric_names = ["nmi_ari_cluster_labels_kmeans"]


for dataset, cell_type_key in zip(datasets, cell_type_keys):
    print(f"processing dataset: {dataset}")
    adata = sc.read_h5ad(f"adata_latent_{dataset}.h5ad")
    z_names = [f"{dataset}-{model}" for model in models]
    save_path = f"{dataset}.png"

    bio_metrics = scib_metrics.benchmark.BioConservation()
    batch_metrics = scib_metrics.benchmark.BatchCorrection(
        False, False, False, False, False
    )
    adata.obs["batch"] = 0
    bm = Benchmarker(
        adata,
        batch_key="batch",
        label_key=cell_type_key,
        embedding_obsm_keys=z_names,
        bio_conservation_metrics=bio_metrics,
        batch_correction_metrics=batch_metrics,
        n_jobs=1
    )
    bm.benchmark()

    df = get_results(bm)
    plot_results_table(df,bm, save_path)