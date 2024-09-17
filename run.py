from preprocessing import setup_adata
from train import Trainer
from utils import *
from plotting import *
from metrics import * 

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

for K in [31,11]:
    for i in range(5):
        # Run preprocessing
        adata = setup_adata(dataset_name=dataset_name,
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
                                ve_layer= ve_layer)

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
        #if backward_velocity:
        #    self_backward_velocity()
        plot_embeddings(adata, dataset_name, K, cell_type_key)
        """compute_scvelo_metrics(adata, dataset_name, K, show, cell_type_key = cell_type_key)
        gpvelo_plots(adata, dataset_name, K, cell_type_key)
        plot_important_genes(adata, dataset_name, K, cell_type_key)
        deg_genes(adata, dataset_name, K, cell_type_key, n_deg_rows=5)
        bayes_factors(adata, cell_type_key, top_N, dataset_name, K, show_plot=False, save_plot=True)
        estimate_uncertainty(adata, model, batch_size=256, n_jobs=1, show=show, dataset=dataset_name, K=K)
        save_adata(adata, dataset_name, K, knn_rep, save_first_regime=False)"""
        os.rename("outputs", f"outputs_{dataset_name}_K{K}_{i}")