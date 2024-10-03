import scanpy as sc
from train import Trainer
import pickle
from model import VAE
import torch
from metrics import *
from plotting import *
from utils import * 
import gc


datasets = ["forebrain", "pancreas", "gastrulation_erythroid",  "dentategyrus_lamanno_P5"]
checkpoints = [20035, 10400, 20050, 20050]

datasets = ["forebrain"]
checkpoints = [20035]

for dataset_name, checkpoint in zip(datasets, checkpoints):
    params = get_parameters(dataset_name, True)

    # Unpack parameters into individual variables
    dataset_name = params['dataset_name']
    preproc_adata = params['preproc_adata']
    smooth_k = params['smooth_k']
    n_highly_var_genes = params['n_highly_var_genes']
    cell_type_key = params['cell_type_key']
    save_umap = params['save_umap']
    show_umap = params['show_umap']
    unspliced_key = params['unspliced_key']
    spliced_key = params['spliced_key']
    filter_on_r2 = params['filter_on_r2']
    knn_rep = params['knn_rep']
    n_components = params['n_components']
    n_knn_search = params['n_knn_search']
    best_key = params['best_key']
    K = params['K']
    ve_layer = params['ve_layer']

    model_hidden_dim = params['model_hidden_dim']
    train_size = params['train_size']
    batch_size = params['batch_size']
    n_epochs = params['n_epochs']
    first_regime_end = params['first_regime_end']
    kl_start = params['kl_start']
    kl_weight_upper = params['kl_weight_upper']
    base_lr = params['base_lr']
    recon_loss_weight = params['recon_loss_weight']
    empirical_loss_weight = params['empirical_loss_weight']
    p_loss_weight = params['p_loss_weight']
    split_data = params['split_data']
    weight_decay = params['weight_decay']
    load_last = params['load_last']
    optimizer_lr_factors = params['optimizer_lr_factors']

    input_folder_name = f"checkpoints/{dataset_name}"

    if os.path.isdir(input_folder_name):
        # load desired model and adata, then extract model outputs to adata
        new_folder_name = f"benchmark/imVelo/imVelo_{dataset_name}"
        if not os.path.isdir(new_folder_name):
            print(f"processing folder: {new_folder_name}")
            adata = sc.read_h5ad(f"{input_folder_name}/{dataset_name}/K{K}/adata/adata_K{K}_dt_{knn_rep}.h5ad")
            if dataset_name == "forebrain":
                adata = add_cell_types_to_adata(adata)
                print(adata)
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
            if dataset_name == "forebrain":
                adata = backward_velocity(adata)

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
            plot_embeddings(adata, dataset_name, K, cell_type_key)
            compute_scvelo_metrics(adata, dataset_name, K, show=False, cell_type_key = cell_type_key)
            gpvelo_plots(adata, dataset_name, K, cell_type_key) #new feature
            plot_important_genes(adata, dataset_name, K, cell_type_key)
            deg_genes(adata, dataset_name, K, cell_type_key, n_deg_rows=5)
            compute_velocity_sign_uncertainty(adata, aggregate_method="mean") #new feature
            bayes_factors(adata, cell_type_key, top_N=10, dataset=dataset_name, K=K, show_plot=False, save_plot=True)
            estimate_uncertainty(adata, model, batch_size=256, n_jobs=1, show=False, dataset=dataset_name, K=K)
            save_adata(adata, dataset_name, K, knn_rep, save_first_regime=False)
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