import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(
        self, 
        input_dims, 
        hidden_dim, 
        latent_dim
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)   
        )
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.z = None
        self.z_mean = None
        self.z_log_var = None

    def forward(self, x):
        x = self.net(x)
        self.z_mean = self.fc_mean(x)
        self.z_log_var = self.fc_log_var(x)
        self.z = self._reparametrize(self.z_mean, self.z_log_var)
        return self.z_mean, self.z_log_var, self.z

    def _reparametrize(self, z_mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

class MaskedLinearDecoder(nn.Module):
    def __init__(
        self, 
        latent_dim, 
        input_dim, 
        mask
    ):
        super().__init__()
        self.linear = nn.Parameter(torch.randn(input_dim, latent_dim))
        self.bias = nn.Parameter(torch.randn(input_dim))
        self.mask = nn.Parameter(torch.tensor(mask, dtype=torch.float32), requires_grad=False)
        self.recons = None

    def forward(self, x):
        masked_gene_weights = self.linear * self.mask
        self.recons = F.linear(x, masked_gene_weights, self.bias)
        return self.recons

class DebugLayer(nn.Module):
    def forward(self, x):
        print("Pre-Softmax Values:", x)
        return x

class KineticsDecoder(nn.Module):
    def __init__(
        self, 
        latent_dim, 
        hidden_dim, 
        output_dim,
        v0_u,
        v0_s
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.out_dim_params = 3*output_dim//2
        self.out_dim_p = 4*output_dim//2

        self.kinetics = nn.Sequential(
            nn.Linear(hidden_dim, self.out_dim_params), #3 output kinetic parameters for each gene
            nn.Softplus()
        )

        self.p_sign_params = nn.Linear(hidden_dim, self.out_dim_p) #4 output probabilities for each gene

        self.u_rate = None
        self.s_rate = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.prediction = None
        self.pp = None
        self.nn = None
        self.pn = None
        self.np = None
        self.v0_u = v0_u
        self.v0_s = v0_s

    def forward(self, z, x, idx):
        net_out = self.net(z)
        kinetic_params = self.kinetics(net_out)
        p_sign = self.p_sign_params(net_out)
        p_sign = p_sign.view(-1, self.out_dim_p//4, 4)
        p_sign = F.softmax(p_sign, dim=-1)
        self.pp = p_sign[:,:,0]
        self.nn = p_sign[:,:,1]
        self.pn = p_sign[:,:,2]
        self.np = p_sign[:,:,3]
        self.alpha, self.beta, self.gamma = torch.split(kinetic_params, kinetic_params.size(1) // 3, dim=1)
        unspliced, spliced = torch.split(x, x.size(1) // 2, dim=1)

        u_rate_pos = self.alpha - self.beta * unspliced #the predicted variation in unspliced rna in unit time
        s_rate_pos = self.beta * unspliced - self.gamma * spliced #the predicted variation in spliced rna in unit time (i.e. "RNA velocity")

        #if self.v0_u:
        #u_rate_pos *= self.v0_u[idx]
        #s_rate_pos *= self.v0_s[idx]

        u_rate_neg = -1 * u_rate_pos
        s_rate_neg = -1 * s_rate_pos

        self.u_rate = u_rate_pos * self.pp + u_rate_neg * self.nn + u_rate_pos * self.pn + u_rate_neg * self.np
        self.s_rate = s_rate_pos * self.pp + s_rate_neg * self.nn + s_rate_neg * self.pn + s_rate_pos * self.np

        unspliced_pred = unspliced + self.u_rate
        spliced_pred = spliced + self.s_rate

        self.prediction = torch.cat([unspliced_pred, spliced_pred], dim=1)

        return self.prediction, self.pp, self.nn, self.pn, self.np