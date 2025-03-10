from preprocessing import setup_adata
from train import Trainer
from utils import *
from plotting import *
from metrics import * 

# Preprocessing parameters
dataset_name = "forebrain"
preproc_adata = True
smooth_k = 200
n_highly_var_genes = 4000
cell_type_key = "clusters"
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
train_size = .9
batch_size = 256
n_epochs = 20500
first_regime_end = 20000
kl_start = 1e-9
kl_weight_upper = 1e-8
base_lr = 1e-4
recon_loss_weight = 1
empirical_loss_weight = 1
p_loss_weight = 1e-1 ########################### TESTING 0 instead of 1e-1
split_data = True
weight_decay = 1e-4
load_last = True

datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters","celltype", "clusters"]
K_vals = [11,11,11,31]

datasets = ["forebrain"]
cell_type_keys = ["Clusters"]
K_vals = [11]
batch_size = 1548 
n_highly_var_genes = 6000

for dataset_name, cell_type_key, K in zip(datasets, cell_type_keys, K_vals):
    #new_folder_name = f"forebrain_kl_upper_{kl_weight_upper}_epoch_20000"
    new_folder_name = f"{dataset_name}"
    #new_folder_name = f"outputs_{dataset_name}_K{K}_knn_rep_{knn_rep}_best_key_{best_key}_{i}_kl_weight_1e-9_{kl_weight_upper}_20k_6000genes"
    if not os.path.isdir(new_folder_name):
        # Run preprocessing
        """adata = setup_adata(dataset_name=dataset_name,
                                preproc_adata=preproc_adata,
                                smooth_k=smooth_k,
                                n_highly_var_genes=n_highly_var_genes,
                                cell_type_key=cell_type_key,
                                save_umap=save_umap,
                                show_umap=show_umap,
                                unspliced_key=unspliced_key,
                                spliced_key=spliced_key,
                                filter_on_r2=filter_on_r2,
                                knn_rep=knn_rep,
                                n_components = n_components,
                                n_knn_search=n_knn_search,
                                best_key=best_key,
                                K = K,
                                ve_layer= ve_layer,
                                ve_hidden_nodes=ve_hidden_nodes)"""

        adata = preprocess(dataset_name)
        distances, indices = manifold_and_neighbors(adata, n_components, n_knn_search, dataset_name, K, knn_rep, best_key, ve_layer, ve_hidden_nodes)
        adata.uns["distances"] = distances
        adata.uns["indices"] = indices

        ### Initialize Trainer and run
        trainer = Trainer(
            adata = adata,
            model_hidden_dim= model_hidden_dim,
            K= K,
            train_size= train_size,
            batch_size= batch_size,
            n_epochs= n_epochs,
            first_regime_end= first_regime_end,
            kl_start= kl_start,
            base_lr= base_lr,
            recon_loss_weight= recon_loss_weight,
            empirical_loss_weight= empirical_loss_weight,
            kl_weight_upper= kl_weight_upper,
            p_loss_weight= p_loss_weight,
            split_data= split_data,
            weight_decay= weight_decay,
            dataset_name=dataset_name,
            load_last = load_last
        )

        print("running first and second regime")
        model = trainer.train()
        trainer.self_extract_outputs()
        save_adata(adata, dataset_name, K, knn_rep, save_first_regime=False)
        save_model(model, dataset_name, K, knn_rep, save_first_regime=False)
        save_trainer(trainer, dataset_name, K, knn_rep, save_first_regime=False)

        ### Downstream analysis
        show = False
        top_N = 15

        plot_losses(trainer, dataset_name, K,figsize=(20, 10))
        plot_isomaps(adata, dataset_name, K, cell_type_key)
        if dataset_name == "forebrain":
            adata = backward_velocity(adata)
        plot_embeddings(adata, dataset_name, K, cell_type_key)
        compute_scvelo_metrics(adata, dataset_name, K, show, cell_type_key = cell_type_key)
        gpvelo_plots(adata, dataset_name, K, cell_type_key)
        plot_important_genes(adata, dataset_name, K, cell_type_key)
        deg_genes(adata, dataset_name, K, cell_type_key, n_deg_rows=5)
        bayes_factors(adata, cell_type_key, top_N, dataset_name, K, show_plot=False, save_plot=True)
        #estimate_uncertainty(adata, model, batch_size=256, n_jobs=1, show=show, dataset=dataset_name, K=K)
        save_adata(adata, dataset_name, K, knn_rep, save_first_regime=False)
        os.rename("outputs", "imVelo_evaluation/"+new_folder_name)

