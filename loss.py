import torch
import torch.nn.functional as F
import scanpy as sc
import numpy as np

class CustomLoss:
    def __init__(
        self, 
        device, 
        recon_loss_weight, 
        kl_weight_upper, 
        empirical_loss_weight,
        p_loss_weight,
        kl_start, 
        annealing_epochs,
        write_losses = False,
    ):
        self.device = device
        self.recon_loss_weight = recon_loss_weight
        self.kl_weight_upper = kl_weight_upper
        self.empirical_loss_weight = empirical_loss_weight
        self.p_loss_weight = p_loss_weight
        self.kl_start = kl_start
        self.annealing_epochs = annealing_epochs
        self.write_losses = write_losses

    def heuristic_loss(
        self,
        adata, 
        x, 
        batch_indices, 
        prediction_nn, 
        device, 
        K):

        reference_data = x #fetch the GE data of the samples in the batch 
        neighbor_indices = adata.uns["indices"][batch_indices,1:K] #fetch the nearest neighbors   
        neighbor_data_u = torch.from_numpy(adata.layers["Mu"][neighbor_indices]).to(device) 
        neighbor_data_s = torch.from_numpy(adata.layers["Ms"][neighbor_indices]).to(device)
        neighbor_data = torch.cat([neighbor_data_u, neighbor_data_s], dim=2) #fetch the GE data of the neighbors for each sample in the batch

        model_prediction_vector = prediction_nn - reference_data #compute the difference vector of the model prediction vs the input samples
        neighbor_prediction_vectors = neighbor_data - reference_data.unsqueeze(1) #compute the difference vector of the neighbor data vs the input samples

        # Normalize the vectors cell-wise
        model_prediction_vector_normalized = F.normalize(model_prediction_vector, p=2, dim=1)
        neighbor_prediction_vectors_normalized = F.normalize(neighbor_prediction_vectors, p=2, dim=2)

        # Calculate the norms of the normalized vectors
        model_prediction_vector_norms = torch.norm(model_prediction_vector_normalized, p=2, dim=1)
        neighbor_prediction_vectors_norms = torch.norm(neighbor_prediction_vectors_normalized, p=2, dim=2)
        
        # Assertions to ensure each vector is a unit vector, considering a small tolerance
        tolerance = 1e-4  # Adjust the tolerance if needed
        #assert torch.allclose(model_prediction_vector_norms, torch.ones_like(model_prediction_vector_norms), atol=tolerance), "Model prediction vectors are not properly normalized"
        #assert torch.allclose(neighbor_prediction_vectors_norms, torch.ones_like(neighbor_prediction_vectors_norms), atol=tolerance), "Neighbor prediction vectors are not properly normalized"

        cos_sim = F.cosine_similarity(neighbor_prediction_vectors_normalized, model_prediction_vector_normalized.unsqueeze(1), dim=-1)

        aggr, _ = cos_sim.max(dim=1)
        cell_loss = 1 - aggr 
        batch_loss = torch.mean(cell_loss) # compute the batch loss

        return batch_loss

    def compute_kl_weight(self, current_epoch):
        # Linear annealing of KL weight
        #print(current_epoch/self.annealing_epochs)
        """print(f"current_epoch: {current_epoch}")
        print(f"annealing_epochs: {self.annealing_epochs}")"""
        return self.kl_start + (self.kl_weight_upper - self.kl_start) * min(1.0, (current_epoch) / self.annealing_epochs)

    def __call__(self, device, adata, learn_kinetics, K, x, batch_indices, model_out, current_epoch):
        # Compute reconstruction loss
        recon_loss = F.mse_loss(model_out["recons"], x)
        # Initializing the uniform probability loss
        uniform_p_loss_value = 0.0
        #print(f"kl weight upper { self.kl_weight_upper}")
        #print(f"kl weight lower { self.kl_start}")

        # Calculate uniform probability loss
        for p in ["pp", "nn", "pn", "np"]:
            uniform_p_loss_value += ((torch.tensor(0.25, device=device) - model_out[p])**2)
        uniform_p_loss_value = (uniform_p_loss_value/4).mean()

        # Compute KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + model_out["z_log_var"] - model_out["z_mean"].pow(2) - model_out["z_log_var"].exp())
        
        if learn_kinetics:
            # Regime 2: No KL annealing, use heuristic loss
            kl_weight = 0
            self.recon_loss_weight = 0
            heuristic_loss_value = self.heuristic_loss(adata, x, batch_indices, model_out["prediction"], device, K)
        else:
            # Regime 1: Linear KL annealing, no heuristic loss
            kl_weight = self.compute_kl_weight(current_epoch)
            heuristic_loss_value = 0
            self.uniform_p_loss = 0

        if self.write_losses:
            return {
                "recons_loss": recon_loss.cpu().numpy(),
                "kl_loss": kl_loss.cpu().numpy(),
                "heuristic_loss": heuristic_loss_value.cpu().numpy(),
                "uniform_p_loss" : uniform_p_loss_value.cpu().numpy()
            }

        def assert_requires_grad(tensor, name):
            assert tensor.requires_grad, f"{name} does not require gradients"
    
        # Apply weights to losses
        recon_loss = self.recon_loss_weight * recon_loss if not learn_kinetics else 0
        kl_loss = kl_weight * kl_loss if not learn_kinetics else 0
        heuristic_loss = self.empirical_loss_weight * heuristic_loss_value if learn_kinetics else 0
        uniform_p_loss = self.p_loss_weight * uniform_p_loss_value if learn_kinetics else 0 

        # Perform assertions
        if not learn_kinetics:
            assert_requires_grad(recon_loss, "recon_loss_value")
            assert_requires_grad(kl_loss, "kl_loss_value")
        else:
            assert_requires_grad(heuristic_loss, "heuristic_loss_value")
            assert_requires_grad(uniform_p_loss, "uniform_p_loss_value")

        # Compute total loss
        total_loss = recon_loss + kl_loss + heuristic_loss + uniform_p_loss

        #print(f"kl weight: {kl_weight}")
        

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "heuristic_loss": heuristic_loss,
            "uniform_p_loss" : uniform_p_loss,
            "kl_weight": kl_weight
        }