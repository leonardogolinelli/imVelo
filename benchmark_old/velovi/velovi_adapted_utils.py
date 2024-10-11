import scanpy as sc
import torch
import os, sys
current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))
from model import VAE
from velovi import VELOVI
import numpy as np

def load_files(dataset, K):
    path = f"../../outputs/{dataset}/K{K}/"
    adata_path = path + "adata/adata.h5ad"
    model_path = path + "model/model.pth"
    adata = sc.read_h5ad(adata_path)
    model = VAE(adata, hidden_dim=512)
    model.load_state_dict(torch.load(model_path))

    return adata, model


def add_velovi_outputs_to_adata(adata, vae):
    latent_time = vae.get_latent_time(n_samples=25)
    velocities = vae.get_velocity(n_samples=25, velo_statistic="mean")
    velocities_u = vae.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="unspliced")

    t = latent_time
    scaling = 20 / t.max(0)

    adata.layers["velocity"] = velocities / scaling
    adata.layers["velocity_u"] = velocities_u / scaling
    adata.layers["latent_time"] = latent_time
    adata.obsm["z"] = vae.get_latent_representation(adata)

    adata.var["fit_alpha"] = vae.get_rates()["alpha"] / scaling
    adata.var["fit_beta"] = vae.get_rates()["beta"] / scaling
    adata.var["fit_gamma"] = vae.get_rates()["gamma"] / scaling
    adata.var["fit_t_"] = (
        torch.nn.functional.softplus(vae.module.switch_time_unconstr)
        .detach()
        .cpu()
        .numpy()
    ) * scaling
    scaling = np.array(scaling)
    adata.layers["fit_t"] = latent_time.values * scaling[np.newaxis, :]
    adata.var['fit_scaling'] = 1.0
    return adata




def return_gnames():
    gnames = [
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
        "Pyy"
    ]
    return gnames

def save_files(adata, vae, dataset):
    try:
        os.remove(f"{dataset}/velovi_{dataset}/model.pt")
    except:
        print()
    try:
        os.rmdir(f"{dataset}/velovi_{dataset}")
    except:
        print()

    try:
        os.remove(f"{dataset}/velovi_{dataset}.h5ad")
    except:
        print()

    adata.write_h5ad(os.path.expanduser(f"{dataset}/velovi_{dataset}.h5ad"))
    VELOVI.save(vae, os.path.expanduser(f"{dataset}/velovi_{dataset}"))


def get_velocity(adata, model, n_samples, full_data_loader, return_mean=True):

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    #model = model.to(device)

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