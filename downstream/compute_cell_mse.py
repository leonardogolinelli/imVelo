
import os
import scanpy as sc

datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters","celltype", "clusters"]

#models = ["celldancer", "imVelo", "ivelo", "ivelo_filtered", "velovi", "velovi_filtered", "scvelo", "stochastic"]

dataset_name = "forebrain"
cell_type_key = "Clusters"
K = 11
knn_rep = "ve"

base_path = f"{dataset_name}"
adata_path = f"{base_path}/{dataset_name}/K{K}/adata/adata_K{K}_dt_{knn_rep}.h5ad"
model_path = f"{base_path}/{dataset_name}/model_checkpoints/model_epoch_{checkpoint}.pt"
new_folder_name = f"{base_path}_{checkpoint}"
trainer_path = f"{base_path}/{dataset_name}/K{K}/trainer/trainer_K{K}_dt_{knn_rep}.pkl"

if not os.path.isdir(new_folder_name):
    adata = sc.read_h5ad(adata_path)
    #adata = sc.read_h5ad("/mnt/data2/home/leonardo/git/multilineage_velocity/checkpoints/pancreas/pancreas/K11/adata/adata_K11_dt_ve.h5ad")

    model = VAE(adata, 512, "cpu")
    model = load_model_checkpoint(adata, model, model_path)
