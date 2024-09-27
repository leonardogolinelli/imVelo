import scanpy as sc
from train import Trainer
import pickle
from model import VAE
import torch
from metrics import *
from plotting import *
from utils import * 
import gc

# Preprocessing parameters
dataset_name = "dentategyrus_lamanno_P5"
preproc_adata = True
smooth_k = 200
n_highly_var_genes = 4000
cell_type_key = "clusters"
save_umap = False
show_umap = False
unspliced_key = "unspliced"
spliced_key = "spliced"
filter_on_r2 = False
knn_rep = "pca"
n_components = 100
n_knn_search = 10
best_key = "pca_unique"
K = 31 
ve_layer = "None"

# Training parameters
model_hidden_dim = 512
K= K
train_size = 1
batch_size = 256
n_epochs = 20500
first_regime_end = 20000
kl_start = 1e-9
kl_weight_upper = 1e-8
base_lr = 1e-4
recon_loss_weight = 1
empirical_loss_weight = 1
p_loss_weight = 1e-1
split_data = False
weight_decay = 1e-4
load_last = True

checkpoints = list(range(20050, 20500, 50))
checkpoints.append(20499)
checkpoints.append(19999)

for checkpoint in checkpoints:
    for i in range(2):
        input_folder_name = f"outputs_{dataset_name}_K{K}_knn_rep_{knn_rep}_best_key_{best_key}_{i}_kl_weight_1e-9_{kl_weight_upper}_20k_12_july"
        if os.path.isdir(input_folder_name):
            # load desired model and adata, then extract model outputs to adata
            new_folder_name = f"{input_folder_name}_checkpoint_{checkpoint}"
            if not os.path.isdir(new_folder_name):
                print(f"processing folder: {new_folder_name}")
                adata = sc.read_h5ad(f"{input_folder_name}/{dataset_name}/K{K}/adata/adata_K{K}_dt_{knn_rep}.h5ad")
                with open(f'{input_folder_name}/{dataset_name}/K{K}/trainer/trainer_K{K}_dt_{knn_rep}.pkl', 'rb') as file:
                    trainer = pickle.load(file)

                model_path = f"{input_folder_name}/{dataset_name}/model_checkpoints/model_epoch_{checkpoint}.pt"
                model = VAE(adata, 512, "cpu")
                model = load_model_checkpoint(adata, model, model_path)
                trainer.device = "cpu"
                trainer.model = model
                trainer.adata = adata
                trainer.self_extract_outputs()

                #BACKWARD VELOCITY SNIPPET HERE <------------------------
                #adata = backward_velocity(adata)

                velocity_u = adata.layers["velocity_u"]
                velocity = adata.layers["velocity"]
                z = adata.obsm["z"]

                os.makedirs("outputs", exist_ok=True)

                np.save("outputs/velocity_u.npy", velocity_u)
                np.save("outputs/velocity.npy", velocity)
                np.save("outputs/z.npy", z)

                # Rerun downstream of interest
                #plot_losses(trainer, dataset_name, K,figsize=(20, 10))
                plot_isomaps(adata, dataset_name, K, cell_type_key)
                if not checkpoint == 20000:
                    plot_embeddings(adata, dataset_name, K, cell_type_key)
                    #compute_scvelo_metrics(adata, dataset_name, K, show=False, cell_type_key = cell_type_key)
                    gpvelo_plots(adata, dataset_name, K, cell_type_key)
                    #plot_important_genes(adata, dataset_name, K, cell_type_key)
                    #deg_genes(adata, dataset_name, K, cell_type_key, n_deg_rows=5)
                    #bayes_factors(adata, cell_type_key, top_N=10, dataset=dataset_name, K=K, show_plot=False, save_plot=True)
                    #estimate_uncertainty(adata, model, batch_size=256, n_jobs=1, show=False, dataset=dataset_name, K=K)
                    #save_adata(adata, dataset_name, K, knn_rep, save_first_regime=False)
                    os.rename("outputs", new_folder_name)

                    # Clear memory after each iteration
                    del adata, trainer, model, velocity_u, velocity, z
                    gc.collect()
                else:
                    plot_embeddings(adata, dataset_name, K, cell_type_key)
                    os.rename("outputs", new_folder_name)
                    # Clear memory after each iteration
                    del adata, trainer, model, velocity_u, velocity, z
                    gc.collect()

            else:
                print("folder already exists")