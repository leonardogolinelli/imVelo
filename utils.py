import os
import scanpy as sc
import torch
import numpy as np
from model import VAE
import pickle

def return_gnames():
    gnames = [
        "Eya2",
        "St3gal6",
        "St18",
        "Gng12",
        "Pdx1",
        "Hspa8",
        "Gnas",
        "Ghrl",
        "Rbfox3",
        "Wfs1",
        "Cald1",
        "Ptn",
        "Tshz1",
        "Rap1b",
        "Slc16a10",
        "Nxph1",
        "Mapt",
        "Marcks",
        "Pax4",
        "Tpm4",
        "Actb",
        "Pou6f2",
        "Tpm1",
        "Rplp0",
        "Phactr1",
        "Isl1",
        "Foxo1",
        "Papss2",
        "Rpl18a",
        "Gnao1",
        "Enpp1",
        "Camk2b",
        "Hsp90b1",
        "Idh2",
        "Fgd2",
        "Syne1",
        "Hspa5",
        "Abcc8",
        "Pyy",
        "Top2a",
        "Rfc2",
        "Kif23",
        "LMO4".lower().capitalize(),
        "NPAS3".lower().capitalize(),
        "AP3M1".lower().capitalize(),
        "ZSWIM4".lower().capitalize(),
        "G2E3".lower().capitalize(),
        "SRGAP2".lower().capitalize()
    ]
    return gnames


def create_dir_tree(dataset, K):
    os.makedirs(f"inputs/annotations/", exist_ok=True)
    os.makedirs(f"outputs/adata_preproc/", exist_ok=True)
    os.makedirs(f"outputs/{dataset}/K{K}/embeddings/", exist_ok=True)
    os.makedirs(f"outputs/{dataset}/K{K}/phase_planes/important_genes/", exist_ok=True)
    os.makedirs(f"outputs/{dataset}/K{K}/adata/", exist_ok=True)
    os.makedirs(f"outputs/{dataset}/K{K}/model/", exist_ok=True)
    os.makedirs(f"outputs/{dataset}/K{K}/trainer/", exist_ok=True)
    os.makedirs(f"outputs/{dataset}/K{K}/stats/bayes_scores/", exist_ok=True)
    os.makedirs(f"outputs/{dataset}/K{K}/stats/uncertainty/", exist_ok=True)
    os.makedirs(f"outputs/{dataset}/K{K}/stats/uncertainty/probabilities", exist_ok=True)
    os.makedirs(f"outputs/{dataset}/K{K}/gpvelo/gp_scatter/", exist_ok=True)
    os.makedirs(f"outputs/{dataset}/K{K}/stats/scvelo_metrics/deg_genes/", exist_ok=True)
    os.makedirs(f"outputs/{dataset}/K{K}/stats/scvelo_metrics/expression/s_genes/", exist_ok=True)
    os.makedirs(f"outputs/{dataset}/K{K}/stats/scvelo_metrics/expression/g2m_genes/", exist_ok=True)
    os.makedirs(f"outputs/{dataset}/K{K}/phase_planes/deg_genes/", exist_ok=True)


def save_adata(adata, dataset, K, knn_rep, save_first_regime=False):
    adata_path = f"outputs/{dataset}/K{K}/adata/"
    os.makedirs(adata_path, exist_ok=True)

    if not save_first_regime:
        path = f"outputs/{dataset}/K{K}/adata/adata_K{K}_dt_{knn_rep}.h5ad"
        adata.write_h5ad(path)
    else:
        path = f"outputs/{dataset}/K{K}/adata/adata_K{K}_dt_{knn_rep}_first_regime.h5ad"
        adata.write_h5ad(path)

def save_model(model, dataset, K, knn_rep, save_first_regime=False):
    model_path = f"outputs/{dataset}/K{K}/model/"
    os.makedirs(model_path, exist_ok=True)

    if not save_first_regime:
        path = f"outputs/{dataset}/K{K}/model/model_K{K}_dt_{knn_rep}.pth"
        torch.save(model.state_dict(), path)
    else:
        path = f"outputs/{dataset}/K{K}/model/model_K{K}_dt_{knn_rep}_first_regime.pth"
        torch.save(model.state_dict(), path)


def save_trainer(trainer, dataset, K, knn_rep, save_first_regime=False):
    trainer_path = f"outputs/{dataset}/K{K}/trainer/"
    os.makedirs(trainer_path, exist_ok=True)

    if not save_first_regime:
        path = f"outputs/{dataset}/K{K}/trainer/trainer_K{K}_dt_{knn_rep}.pkl"
        with open(path, 'wb') as file:
            pickle.dump(trainer, file)
    else:
        path = f"outputs/{dataset}/K{K}/trainer/trainer_K{K}_dt_{knn_rep}_first_regime.pkl"
        with open(path, 'wb') as file:
            pickle.dump(trainer, file)


def get_velocity(adata, model, n_samples, full_data_loader, return_mean=True):
    model.eval()

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    model = model.to(device)

    velocities = torch.zeros((n_samples+return_mean, adata.shape[0], adata.shape[1])).to(device)
    with torch.no_grad():
        for i in range(n_samples):
            print(i)
            for x_batch, idx_batch in full_data_loader:
                x_batch = x_batch.to(device)
                model(x_batch, idx_batch, learn_kinetics=True)
                velocity = model.kinetics_decoder.s_rate#.cpu().numpy()
                velocities[i, idx_batch,:] = velocity
    
    if return_mean:
        velocities[-1] = velocities.mean(0)

    return -1 * velocities.cpu().numpy()

def load_model(model, epoch, model_path):
    # Load the specific model checkpoint for the given epoch
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded model from epoch {epoch} with loss {checkpoint['loss']}")

    return model

def extract_outputs(adata, model, full_data_loader, device):
    # Collect outputs from the model in eval mode
    model.eval()

    with torch.no_grad():
        adata.obsm["z"] = np.zeros((adata.shape[0], adata.uns["mask"].shape[1]))
        adata.obsm["z_mean"] = np.zeros_like(adata.obsm["z"])
        adata.obsm["z_log_var"] = np.zeros_like(adata.obsm["z"])
        adata.obsm["recons"] = np.zeros((adata.shape[0], adata.shape[1]*2))
        adata.obsm["prediction"] = np.zeros_like(adata.obsm["recons"])
        adata.layers["alpha"] = np.zeros(adata.shape)
        adata.layers["beta"] = np.zeros(adata.shape)
        adata.layers["gamma"] = np.zeros(adata.shape)
        adata.layers["velocity_u"] = np.zeros(adata.shape)
        adata.layers["velocity"] =  np.zeros(adata.shape)
        adata.layers["pp"] = np.zeros(adata.shape)
        adata.layers["nn"] = np.zeros(adata.shape)
        adata.layers["pn"] = np.zeros(adata.shape)
        adata.layers["np"] = np.zeros(adata.shape)

        for x_batch, idx_batch in full_data_loader:
            x_batch = x_batch.to(device)
            model(x_batch, idx_batch, learn_kinetics=True)
            adata.obsm["z"][idx_batch,:] = model.encoder.z.cpu().numpy()
            adata.obsm["z_mean"][idx_batch,:] = model.encoder.z_mean.cpu().numpy()
            adata.obsm["z_log_var"][idx_batch,:] = model.encoder.z_log_var.cpu().numpy()
            adata.obsm["recons"][idx_batch,:] = model.linear_decoder.recons.cpu().numpy()
            adata.obsm["prediction"][idx_batch,:] = model.kinetics_decoder.prediction.cpu().numpy()
            adata.layers["alpha"][idx_batch,:] = model.kinetics_decoder.alpha.cpu().numpy()
            adata.layers["beta"][idx_batch,:] = model.kinetics_decoder.beta.cpu().numpy()
            adata.layers["gamma"][idx_batch,:] = model.kinetics_decoder.gamma.cpu().numpy()
            #inverse sign
            adata.layers["velocity_u"][idx_batch,:] = -1 * model.kinetics_decoder.u_rate.cpu().numpy()
            adata.layers["velocity"][idx_batch,:] =  -1 * model.kinetics_decoder.s_rate.cpu().numpy()
            #inverse sign
            adata.layers["nn"][idx_batch,:] = model.kinetics_decoder.pp.cpu().numpy()
            adata.layers["pp"][idx_batch,:] = model.kinetics_decoder.nn.cpu().numpy()
            adata.layers["np"][idx_batch,:] = model.kinetics_decoder.pn.cpu().numpy()
            adata.layers["pn"][idx_batch,:] = model.kinetics_decoder.np.cpu().numpy()

        linear_weights = model.linear_decoder.linear.cpu().numpy()
        adata.uns["linear_weights"] = linear_weights
        weights_u, weights_s = np.split(linear_weights, 2,axis=0)
        gp_velo_u = np.matmul(adata.layers["velocity_u"],weights_u)
        gp_velo = np.matmul(adata.layers["velocity"],weights_s)
        adata.obsm["gp_velo_u"] = gp_velo_u
        adata.obsm["gp_velo"] = gp_velo

    #checking that a cell sums to one with respect to its probabilities
    probs_sum = (adata.layers["pp"] + adata.layers["nn"] + adata.layers["pn"] + adata.layers["np"])
    if np.isclose(probs_sum, 1, atol=1e-2).all():
        print("Confirmed that cell probabilities sum to one for all cells")
    else:
        print(probs_sum)
        raise ValueError("Not all probabilities sum to one, recheck the code")
    
def load_files(dataset, K, knn_rep, hidden_dim, load_first_regime):
    path = f"outputs/{dataset}/K{K}/"
    if not load_first_regime:
        adata_path = path + f"adata/adata_K{K}_dt_{knn_rep}.h5ad"
        model_path = path + f"model/model_K{K}_dt_{knn_rep}.pth"
        trainer_path = path + f"trainer/trainer_K{K}_dt_{knn_rep}.pkl"
    else:
        adata_path = path + f"adata/adata_K{K}_dt_{knn_rep}_first_regime.h5ad"
        model_path = path + f"model/model_K{K}_dt_{knn_rep}_first_regime.pth"
        trainer_path = path + f"trainer/trainer_K{K}_dt_{knn_rep}_first_regime.pkl"
    
    # Check if files exist and are not empty
    if not (os.path.exists(adata_path) and os.path.getsize(adata_path) > 0):
        raise FileNotFoundError(f"Adata file not found or is empty: {adata_path}")
    if not (os.path.exists(model_path) and os.path.getsize(model_path) > 0):
        raise FileNotFoundError(f"Model file not found or is empty: {model_path}")
    if not (os.path.exists(trainer_path) and os.path.getsize(trainer_path) > 0):
        raise FileNotFoundError(f"Trainer file not found or is empty: {trainer_path}")

    try:
        adata = sc.read_h5ad(adata_path)
        model = VAE(adata, hidden_dim=hidden_dim)
        model.load_state_dict(torch.load(model_path))
        with open(trainer_path, 'rb') as file:
            trainer = pickle.load(file)
    except EOFError as e:
        raise EOFError(f"Error loading trainer file {trainer_path}: {e}")
    
    return adata, model, trainer

def rename_duplicate_terms(adata):
    terms = adata.uns["terms"]
    term_dic = {}
    for i,term in enumerate(terms):
        if term not in term_dic:
            term_dic[term] = 0
        else:
            term_dic[term] +=1
            terms[i] = f"{term}_{term_dic[term]}"

    adata.uns["terms"] = terms

    double_idx = np.where(np.array([list(terms).count(term) for term in terms])>1)[0]
    assert len(double_idx) == 0

def latent_directions(adata, method="sum", get_confidence=False, key_added='directions'):
        """Get directions of upregulation for each latent dimension.
           Multipling this by raw latent scores ensures positive latent scores correspond to upregulation.
           Parameters
           ----------
           method: String
                Method of calculation, it should be 'sum' or 'counts'.
           get_confidence: Boolean
                Only for method='counts'. If 'True', also calculate confidence
                of the directions.
           adata: AnnData
                An AnnData object to store dimensions. If 'None', self.adata is used.
           key_added: String
                key of adata.uns where to put the dimensions.
        """
        
        terms_weights = adata.uns["linear_weights"]

        if method == "sum":
            signs = terms_weights.sum(0).cpu().numpy()
            signs[signs>0] = 1.
            signs[signs<0] = -1.
            confidence = None
        elif method == "counts":
            num_nz = torch.count_nonzero(terms_weights, dim=0)
            upreg_genes = torch.count_nonzero(terms_weights > 0, dim=0)
            signs = upreg_genes / (num_nz+(num_nz==0))
            signs = signs.cpu().numpy()
            num_nz = num_nz.cpu().numpy()

            confidence = signs.copy()
            confidence = np.abs(confidence-0.5)/0.5
            confidence[num_nz==0] = 0

            signs[signs>0.5] = 1.
            signs[signs<0.5] = -1.

            signs[signs==0.5] = 0
            signs[num_nz==0] = 0
        else:
            adata.uns[key_added] = signs

            if get_confidence and confidence is not None:
                adata.uns[key_added + '_confindence'] = confidence


def backward_velocity(adata):
    adata.layers["velocity"] *= -1
    adata.layers["velocity_u"] *= -1
    adata.obsm["gp_velo"] *=-1
    adata.obsm["gp_velo_u"] *=-1
    return adata


def fetch_relevant_terms(dataset):
    if dataset == "pancreas":
        terms = ["DUCTAL_CELLS", "ALPHA_CELLS", "BETA_CELLS", "DELTA_CELLS"]

    elif dataset == "gastrulation_erythroid":
        terms = ["ERYTHROID-LIKE_AND_ERYTHROID_P",
                  "DEVELOPMENTAL_BIOLOGY", "PLATELET_AGGREGATION_PLUG_FORM",
                  "IRON_UPTAKE_AND_TRANSPORT"]
        
    elif dataset in ["dentategyrus_lamanno", "dentategyrus_lamanno_P0", "dentategyrus_lamanno_P5"]:
        terms = [
            'ASTROCYTES',
            'BERGMANN_GLIA',
            'CAJAL-RETZIUS_CELLS',
            'EPENDYMAL_CELLS',
            'IMMATURE_NEURONS',
            'INTERNEURONS',
            'NEURAL_STEM/PRECURSOR_CELLS',
            'NEUROBLASTS',
            'NEUROENDOCRINE_CELLS',
            'NEURONS',
            'OLIGODENDROCYTE_PROGENITOR_CEL',
            'OLIGODENDROCYTES',
            'PURKINJE_NEURONS',
            'PYRAMIDAL_CELLS',
            'RADIAL_GLIA_CELLS',
            'SCHWANN_CELLS',
            'TRIGEMINAL_NEURONS',
            'AXON_GUIDANCE',
            'NEURONAL_SYSTEM',
            'NEUROTRANSMITTER_RELEASE_CYCLE',
            'SIGNALLING_BY_NGF',
            'SIGNALING_BY_BMP',
            'SIGNALING_BY_WNT',
            'CELL_CYCLE',
            'G1_S_TRANSITION',
            'S_PHASE',
            'G2_M_CHECKPOINTS',
            'DNA_REPLICATION',
            'SIGNALING_BY_NOTCH',
            'REGULATION_OF_APOPTOSIS',
            'TRANSCRIPTION',
            'RNA_POL_II_TRANSCRIPTION',
            'RNA_POL_III_TRANSCRIPTION'
        ]

    elif dataset == "forebrain":
        terms = [
            'EMBRYONIC_STEM_CELLS',
            'NEURAL_STEM/PRECURSOR_CELLS',
            'IMMATURE_NEURONS',
            'NEUROBLASTS',
            'NEURONS',
            'INTERNEURONS',
            'CAJAL-RETZIUS_CELLS',
            'PURKINJE_NEURONS',
            'PYRAMIDAL_CELLS',
            'TRIGEMINAL_NEURONS'
        ]

    return terms

def manifold_and_neighbors(adata, n_components, n_knn_search, dataset_name, K, knn_rep, best_key, ve_layer, ve_hidden_nodes):
    from sklearn.manifold import Isomap
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors

    MuMs = adata.obsm["MuMs"]

    print("computing isomap 1...")
    isomap = Isomap(n_components=n_components, n_neighbors=n_knn_search).fit_transform(MuMs)
    print("computing isomap 2..")
    isomap_unique = Isomap(n_components=1, n_neighbors=n_knn_search).fit_transform(MuMs)
    pca_runner = PCA(n_components=n_components)
    pca = pca_runner.fit_transform(MuMs)
    pca_unique = PCA(n_components=1).fit_transform(MuMs)
    adata.uns["PCA_weights"] = pca_runner.components_
    ve_path = f"/mnt/data2/home/leonardo/git/dim_reduction/{ve_hidden_nodes}/embeddings/6layer_{dataset_name}_smooth_K_{ve_layer}.npy"
    #ve = np.load(f"../dim_reduction/outputs/saved_z_matrices/{dataset_name}_z{ve_layer[0]}.npy")
    ve = np.load(ve_path)
    print(f"ve shape: {ve.shape}")
    print(f"adata shape: {adata.shape}")
    for rep, name in zip([isomap, isomap_unique, pca, pca_unique, ve], ["isomap", "isomap_unique", "pca", "pca_unique", "ve"]):
        adata.obsm[name] = rep
        base_path = f"outputs/{dataset_name}/K{K}/embeddings/time_umaps/"
        os.makedirs(base_path, exist_ok=True)
        if name in ["isomap", "pca"]:
            fname = f"{name}_1"
            adata.obs[fname] = rep[:,0]
            #sc.pl.umap(adata, color=fname)
            #plt.savefig(f"{base_path}{fname}", bbox_inches="tight")

            if n_components > 1:
                fname = f"{name}_2"
                adata.obs[fname] = rep[:,1]
                #sc.pl.umap(adata, color=fname)
                #plt.savefig(f"{base_path}{fname}", bbox_inches="tight")

                fname = f"{name}_3"
                adata.obs[fname] = rep[:,2]
                #sc.pl.umap(adata, color=fname)
                #plt.savefig(f"{base_path}{fname}", bbox_inches="tight")

                fname = f"{name}_1+2"
                adata.obs[fname] = rep[:,0] + rep[:,1]
                #sc.pl.umap(adata, color=fname)
                #plt.savefig(f"{base_path}{fname}", bbox_inches="tight")

                fname = f"{name}_1+3"
                adata.obs[fname] = rep[:,0] + rep[:,2]
                #sc.pl.umap(adata, color=fname)
                #plt.savefig(f"{base_path}{fname}", bbox_inches="tight")

                fname = f"{name}_2+3"
                adata.obs[fname] = rep[:,1] + rep[:,2]
                #sc.pl.umap(adata, color=fname)
                #plt.savefig(f"{base_path}{fname}", bbox_inches="tight")

    print(f"n_components: {n_components}")
    print(f"n_neighbors: {n_knn_search}")
    print(f"knn rep used: {knn_rep}")

    if knn_rep == "isomap":
        print("isomap key used")
        embedding = isomap

        if best_key:
            print("best key used")
            embedding = np.array(adata.obsm[best_key]).reshape(-1,1)

    elif knn_rep == "isomap_unique":
        print("isomap unique key used")
        embedding = isomap_unique
        
    elif knn_rep == "ve":
        print("ve key used")
        embedding = ve

    elif knn_rep == "pca":
        print("pca key used")
        embedding = pca

        if best_key:
            print("best key used")
            embedding = np.array(adata.obsm[best_key]).reshape(-1,1)
        
    nbrs = NearestNeighbors(n_neighbors=adata.shape[0], metric='euclidean')
    nbrs.fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)

    return distances, indices

def load_model_checkpoint(adata, model, model_path):
    # Load checkpoint with strict state checking to ensure compatibility
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to('cpu')  # Ensure the model is set to CPU as expected

    # Ensure all gradients are disabled (if not needed)
    for param in model.parameters():
        param.requires_grad = False

    print("Model checkpoint loaded successfully.")
    return model


def get_parameters(dataset_name, learn_kinetics):
    # Common parameters
    preproc_adata = True
    save_umap = False
    show_umap = False
    unspliced_key = "unspliced"
    spliced_key = "spliced"
    filter_on_r2 = False
    n_knn_search = 10
    ve_layer = "None"
    split_data = False
    weight_decay = 1e-4
    load_last = True
    train_size = 1
    recon_loss_weight = 1
    empirical_loss_weight = 1
    p_loss_weight = 1e-1
    kl_start = 1e-9
    kl_weight_upper = 1e-8

    # Parameters dependent on the dataset
    if dataset_name == "pancreas":
        # Preprocessing parameters
        smooth_k = 200
        n_highly_var_genes = 4000
        cell_type_key = "clusters"
        knn_rep = "ve"
        n_components = 10
        best_key = None
        K = 11

        # Training parameters
        model_hidden_dim = 512
        batch_size = 256
        n_epochs = 10400
        first_regime_end = 10000
        base_lr = 1e-4

        # Learning rate adjustments for optimizer param groups
        optimizer_lr_factors = {
            'encoder': 1e-1 if learn_kinetics else 1,
            'linear_decoder': 1e-1 if learn_kinetics else 1,
            'kinetics_decoder': 1e-1 if learn_kinetics else 0
        }

    elif dataset_name == "forebrain":
        # Preprocessing parameters
        smooth_k = 200
        n_highly_var_genes = 6000
        cell_type_key = "Clusters"
        knn_rep = "ve"
        n_components = 10
        best_key = None
        K = 11

        # Training parameters
        model_hidden_dim = 512
        batch_size = 1720
        n_epochs = 20100
        first_regime_end = 20000
        base_lr = 1e-4

        # Learning rate adjustments for optimizer param groups
        optimizer_lr_factors = {
            'encoder': 1e-3 if learn_kinetics else 1,
            'linear_decoder': 1e-3 if learn_kinetics else 1,
            'kinetics_decoder': 1e-3 if learn_kinetics else 0
        }

    elif dataset_name == "gastrulation_erythroid":
        # Preprocessing parameters
        smooth_k = 200
        n_highly_var_genes = 4000
        cell_type_key = "celltype"
        knn_rep = "ve"
        n_components = 10
        best_key = None
        K = 11

        # Training parameters
        model_hidden_dim = 512
        batch_size = 1024
        n_epochs = 20500
        first_regime_end = 20000
        base_lr = 1e-4

        # Learning rate adjustments for optimizer param groups
        optimizer_lr_factors = {
            'encoder': 1e-3 if learn_kinetics else 1,
            'linear_decoder': 1e-3 if learn_kinetics else 1,
            'kinetics_decoder': 1e-3 if learn_kinetics else 0
        }

    elif dataset_name == "dentategyrus_lamanno_P5":
        # Preprocessing parameters
        smooth_k = 200
        n_highly_var_genes = 4000
        cell_type_key = "clusters"
        knn_rep = "pca"
        n_components = 100
        best_key = "pca_unique"
        K = 31

        # Training parameters
        model_hidden_dim = 512
        batch_size = 256
        n_epochs = 20500
        first_regime_end = 20000
        base_lr = 1e-4

        # Learning rate adjustments for optimizer param groups
        optimizer_lr_factors = {
            'encoder': 1e-3 if learn_kinetics else 1,
            'linear_decoder': 1-3 if learn_kinetics else 1,
            'kinetics_decoder': 1-3 if learn_kinetics else 0
        }

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Compile all parameters into a dictionary
    parameters = {
        # Preprocessing parameters
        'dataset_name': dataset_name,
        'preproc_adata': preproc_adata,
        'smooth_k': smooth_k,
        'n_highly_var_genes': n_highly_var_genes,
        'cell_type_key': cell_type_key,
        'save_umap': save_umap,
        'show_umap': show_umap,
        'unspliced_key': unspliced_key,
        'spliced_key': spliced_key,
        'filter_on_r2': filter_on_r2,
        'knn_rep': knn_rep,
        'n_components': n_components,
        'n_knn_search': n_knn_search,
        'best_key': best_key,
        'K': K,
        've_layer': ve_layer,

        # Training parameters
        'model_hidden_dim': model_hidden_dim,
        'train_size': train_size,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'first_regime_end': first_regime_end,
        'kl_start': kl_start,
        'kl_weight_upper': kl_weight_upper,
        'base_lr': base_lr,
        'recon_loss_weight': recon_loss_weight,
        'empirical_loss_weight': empirical_loss_weight,
        'p_loss_weight': p_loss_weight,
        'split_data': split_data,
        'weight_decay': weight_decay,
        'load_last': load_last,

        # Optimizer learning rate adjustments
        'optimizer_lr_factors': optimizer_lr_factors
    }

    return parameters


def add_cell_types_to_adata(adata):
    cluster_to_cell_type = {
    '0': 'Radial Glia 1',
    '1': 'Radial Glia 2',
    '2': 'Neuroblast 1',
    '3': 'Neuroblast 2',
    '4': 'Immature Neuron 1',
    '5': 'Immature Neuron 2',
    '6': 'Neuron'
    }

    # Example of mapping clusters to cell types in your dataset
    adata.obs['Clusters'] = adata.obs['Clusters'].map(cluster_to_cell_type)

    return adata

def preprocess(
    dataset_name
    ):
        if dataset_name == "forebrain":
            adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/benchmark/imVelo/forebrain/forebrain/K11/adata/adata_K11_dt_ve.h5ad"
        elif dataset_name == "pancreas":
            adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/benchmark/imVelo/pancreas/pancreas/K11/adata/adata_K11_dt_ve.h5ad"
        elif dataset_name == "gastrulation_erythroid":
            adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/benchmark/imVelo/gastrulation_erythroid/gastrulation_erythroid/K11/adata/adata_K11_dt_ve.h5ad"
        elif dataset_name == "dentategyrus_lamanno_P5":
            adata_path = "/mnt/data2/home/leonardo/git/multilineage_velocity/benchmark/imVelo/dentategyrus_lamanno_P5/imVelo_dentategyrus_lamanno_P5.h5ad"

        adata = sc.read_h5ad(adata_path)

        print(f"number of gene programs: {len(adata.uns['terms'])}")

        return adata