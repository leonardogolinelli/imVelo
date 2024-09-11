import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules import Encoder, MaskedLinearDecoder, KineticsDecoder

class VAE(nn.Module):
    def __init__(
        self,
        adata,
        hidden_dim=512,
        device ="cpu"
    ):
        super().__init__()
        mask = adata.uns["mask"]
        if "v0_u" in adata.layers:
            v0_u = torch.tensor(adata.layers["v0_u"], device=device)
            v0_s = torch.tensor(adata.layers["v0_s"], device=device)
        else:
            v0_u, v0_s = None, None

        input_dim = adata.shape[1]*2
        latent_dim = mask.shape[1]
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.linear_decoder = MaskedLinearDecoder(latent_dim, input_dim, mask)
        self.kinetics_decoder = KineticsDecoder(latent_dim, hidden_dim, input_dim, v0_u, v0_s)

    def forward(self, x, idx, learn_kinetics=False):
        z_mean, z_log_var, z = self.encoder(x)
        recons = self.linear_decoder(z)
        if learn_kinetics:
            prediction, pp, nn, pn, np = self.kinetics_decoder(z, x, idx)
        else:
            prediction, pp, nn, pn, np = 0, 0, 0, 0, 0
        
        #output contains values relevant in loss calculation
        outputs = {
            "prediction": prediction,
            "recons": recons,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
            "pp" : pp,
            "nn" : nn,
            "pn" : pn,
            "np" : np
        }
        return outputs



