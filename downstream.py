import scanpy as sc
from train import Trainer
import pickle
from model import VAE
import torch
from metrics import *
from plotting import *
from utils import * 

# Preprocessing parameters
dataset_name = "gastrulation_erythroid"
preproc_adata = True
smooth_k = 200
n_highly_var_genes = 4000
cell_type_key = "celltype"
save_umap = False
show_umap = False
unspliced_key = "unspliced"
spliced_key = "spliced"
filter_on_r2 = False
knn_rep = "ve"
n_components = 10
n_knn_search = 10
best_key = None
K = 11
ve_layer = "None"

# Training parameters
model_hidden_dim = 512
K= K
train_size = 1
batch_size = 1024
n_epochs = 2500
first_regime_end = 2000
kl_start = 1e-5
base_lr = 1e-4
recon_loss_weight = 1
empirical_loss_weight = 1
kl_weight_upper = 1e-4
p_loss_weight = 1e-1
split_data = False
weight_decay = 1e-4
load_last = True

for ve_256 in [True, False]:
    ve_256_str = "256_" if ve_256 else ""
    for checkpoint in [2499, 2400, 2300, 2200, 2100, 2000, 1999]:
        for i in range(0,5):
            # load desired model and adata, then extract model outputs to adata
            new_folder_name = f"onlyoutputs_{dataset_name}_K{K}_{ve_256_str}{i}_checkpoint_{checkpoint}"
            if not os.path.isdir(new_folder_name):
                adata = sc.read_h5ad(f"outputs_{dataset_name}_K{K}_{ve_256_str}{i}/{dataset_name}/K{K}/adata/adata_K{K}_dt_ve.h5ad")
                with open(f'outputs_{dataset_name}_K{K}_{ve_256_str}{i}/{dataset_name}/K{K}/trainer/trainer_K{K}_dt_ve.pkl', 'rb') as file:
                    trainer = pickle.load(file)

                model_path = f"outputs_{dataset_name}_K{K}_{ve_256_str}{i}/{dataset_name}/model_checkpoints/model_epoch_{checkpoint}.pt"
                model = VAE(adata, 512, "cpu")
                model = load_model_checkpoint(adata, model, model_path)
                trainer.device = "cpu"
                trainer.model = model
                trainer.adata = adata
                trainer.self_extract_outputs()

                # Rerun downstream of interest
                #plot_losses(trainer, dataset_name, K,figsize=(20, 10))
                plot_isomaps(adata, dataset_name, K, cell_type_key)
                #if backward_velocity:
                #    self_backward_velocity()
                plot_embeddings(adata, dataset_name, K, cell_type_key)
                compute_scvelo_metrics(adata, dataset_name, K, show=False, cell_type_key = cell_type_key)
                gpvelo_plots(adata, dataset_name, K, cell_type_key)
                plot_important_genes(adata, dataset_name, K, cell_type_key)
                deg_genes(adata, dataset_name, K, cell_type_key, n_deg_rows=5)
                bayes_factors(adata, cell_type_key, top_N=10, dataset=dataset_name, K=K, show_plot=False, save_plot=True)
                #estimate_uncertainty(adata, model, batch_size=256, n_jobs=1, show=False, dataset=dataset_name, K=K)
                #save_adata(adata, dataset_name, K, knn_rep, save_first_regime=False)
                os.rename("outputs", new_folder_name)