import scanpy as sc
import torch
import os, sys
current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))
from model import VAE

def load_files(dataset, K):
    path = f"../../outputs/{dataset}/K{K}/"
    adata_path = path + "adata/adata.h5ad"
    model_path = path + "model/model.pth"
    adata = sc.read_h5ad(adata_path)
    model = VAE(adata, hidden_dim=512)
    model.load_state_dict(torch.load(model_path))

    return adata, model


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