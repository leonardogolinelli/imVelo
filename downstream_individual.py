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
ve_hidden_nodes = "12_july"

# Training parameters
model_hidden_dim = 512
K= K
train_size = 1
batch_size = 256
n_epochs = 2500
first_regime_end = 2000
kl_start = 1e-9
kl_weight_upper = 1e-8
base_lr = 1e-4
recon_loss_weight = 1
empirical_loss_weight = 1
p_loss_weight = 1e-1 ########################### TESTING 0 instead of 1e-1
split_data = False
weight_decay = 1e-4
load_last = True

#checkpoints = range(2000, 2500, 50)
#checkpoints = [2100, 2200, 2250, 2300, 2350, 2400, 2450, 2499]
#checkpoints = list(range(2000, 2500, 50))
checkpoints = [2499]
K=11
for checkpoint in checkpoints:
    base_path = "gastrulation_erythroid_test_1"
    adata_path = f"{base_path}/{dataset_name}/K{K}/adata/adata_K{K}_dt_{knn_rep}.h5ad"
    model_path = f"{base_path}/{dataset_name}/model_checkpoints/model_epoch_{checkpoint}.pt"
    new_folder_name = f"{base_path}_{checkpoint}"
    trainer_path = f"{base_path}/{dataset_name}/K{K}/trainer/trainer_K{K}_dt_{knn_rep}.pkl"

    if not os.path.isdir(new_folder_name):
        adata = sc.read_h5ad(adata_path)
        #adata = sc.read_h5ad("/mnt/data2/home/leonardo/git/multilineage_velocity/checkpoints/pancreas/pancreas/K11/adata/adata_K11_dt_ve.h5ad")

        model = VAE(adata, 512, "cpu")
        model = load_model_checkpoint(adata, model, model_path)

        with open(trainer_path, 'rb') as file:
            trainer = pickle.load(file)
            trainer.device = "cpu"
            trainer.model = model
            trainer.adata = adata
            trainer.self_extract_outputs()

        velocity_u = adata.layers["velocity_u"]
        velocity = adata.layers["velocity"]
        z = adata.obsm["z"]

        os.makedirs("outputs", exist_ok=True)

        #np.save("outputs/velocity_u.npy", velocity_u)
        #np.save("outputs/velocity.npy", velocity)
        #np.save("outputs/z.npy", z)

        # Rerun downstream of interest
        plot_losses(trainer, dataset_name, K,figsize=(20, 10))
        plot_isomaps(adata, dataset_name, K, cell_type_key)
        if dataset_name == "forebrain":
            adata = backward_velocity(adata)
        compute_velocity_sign_uncertainty(adata, aggregate_method='mean')
        plot_embeddings(adata, dataset_name, K, cell_type_key)
        #compute_scvelo_metrics(adata, dataset_name, K, show=False, cell_type_key = cell_type_key)
        gpvelo_plots(adata, dataset_name, K, cell_type_key)
        plot_important_genes(adata, dataset_name, K, cell_type_key)
        deg_genes(adata, dataset_name, K, cell_type_key, n_deg_rows=5)
        bayes_factors(adata, cell_type_key, top_N=10, dataset=dataset_name, K=K, show_plot=False, save_plot=True)
        estimate_uncertainty(adata, model, batch_size=256, n_jobs=1, show=False, dataset=dataset_name, K=K)
        save_adata(adata, dataset_name, K, knn_rep, save_first_regime=False)
        os.rename("outputs", new_folder_name)
    else:
        print(f"{new_folder_name} already exists!")
