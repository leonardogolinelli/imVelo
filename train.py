import torch
import torch.optim as optim
import numpy as np
from dataloaders import setup_dataloaders
from loss import CustomLoss
import torch.optim as optim
from model import VAE
from utils import extract_outputs
import os
import pickle
import time

class Trainer:
    def __init__(self, adata, model_hidden_dim=512, K=30, train_size=0.8, batch_size=256, 
                 n_epochs=2000, first_regime_end=1000, kl_start=1e-4, base_lr=1e-5, 
                 recon_loss_weight=1, empirical_loss_weight=1, kl_weight_upper=1e-1, 
                 p_loss_weight=1e-2, split_data=True, n_samples=100, weight_decay = 1e-4,
                 dataset_name="pancreas", load_last=True):
        self.adata = adata
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VAE(adata, hidden_dim=model_hidden_dim, device=self.device).to(self.device)
        self.K = K
        self.train_size = train_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.first_regime_end = first_regime_end
        self.base_lr = base_lr
        self.recon_loss_weight = recon_loss_weight
        self.empirical_loss_weight = empirical_loss_weight
        self.kl_weight_upper = kl_weight_upper
        self.p_loss_weight = p_loss_weight
        self.kl_start = kl_start
        self.split_data = split_data
        self.n_samples = n_samples
        self.weight_decay = weight_decay,
        self.dataset_name = dataset_name
        self.load_last = load_last

        # Setup dataloaders
        self.train_dl, self.test_dl, self.full_data_loader = setup_dataloaders(self.adata, self.batch_size, self.train_size, self.split_data)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.base_lr, weight_decay=weight_decay)

        # Loss calculator
        self.loss_calculator = CustomLoss(
            device=self.device,
            recon_loss_weight=self.recon_loss_weight,
            kl_weight_upper=self.kl_weight_upper,
            empirical_loss_weight=self.empirical_loss_weight,
            p_loss_weight = self.p_loss_weight,
            kl_start=self.kl_start, 
            annealing_epochs=self.first_regime_end,
            write_losses=False
        )

        # Initialize loss register
        self.loss_register = {
            'training': {
                "total_loss": [],
                "recon_loss": [],
                "kl_loss": [],
                "heuristic_loss": [],
                "uniform_p_loss" : [],
                "kl_weight":[],
            },

            'evaluation': {
                "total_loss": [],
                "recon_loss": [],
                "kl_loss": [],
                "heuristic_loss": [],
                "uniform_p_loss" : [],
                "kl_weight":[],
            }
        }
    
        # Initialize model checkpoints dictionary
        self.model_checkpoints = {}
        self.losses = {}

    def write_loss_register(self, loss_dictionary, modality):
        for key, val in loss_dictionary.items():
            self.loss_register[modality][key].append(val)

    def write_final_losses(self):
        self.model.eval()
        with torch.no_grad():
            recons_loss = 0
            kl_loss = 0
            heuristic_loss = 0
            uniform_p_loss = 0
            _,_, full_dl = setup_dataloaders(self.adata, 256, split_data=False)
            for data, idx in full_dl:
                data = data.to(self.device)
                out_dic = self.model(data, idx)
                final_losses_calculator = CustomLoss(
                    device=self.device,
                    recon_loss_weight=self.recon_loss_weight,
                    kl_weight_upper=self.kl_weight_upper,
                    empirical_loss_weight=self.empirical_loss_weight,
                    p_loss_weight=self.p_loss_weight,
                    kl_start=self.kl_start,
                    annealing_epochs=self.first_regime_end,
                    write_losses=True
                )
                losses = final_losses_calculator(self.device, self.adata,
                                            learn_kinetics=True, K=self.K, x=data,
                                                batch_indices=idx, model_out=out_dic, current_epoch=self.first_regime_end)
            
                recons_loss += losses["recons_loss"] / len(full_dl)
                kl_loss += losses["kl_loss"] / len(full_dl)
                heuristic_loss += losses["heuristic_loss"] / len(full_dl)
                uniform_p_loss += losses["uniform_p_loss"] / len(full_dl)

            final_losses_dic = {}
            final_losses_dic["recons_loss"] = recons_loss
            final_losses_dic["kl_loss"] = kl_loss
            final_losses_dic["heuristic_loss"] = heuristic_loss
            final_losses_dic["uniform_p_loss"] = uniform_p_loss

            self.adata.uns["final_losses_dic"] = final_losses_dic

            return final_losses_dic
 
    def train_epoch(self, learn_kinetics, current_epoch):
        self.model.train()

        total_loss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        heuristic_loss = 0.0
        uniform_p_loss = 0.0

        for x_batch, idx_batch in self.train_dl:
            x_batch = x_batch.to(self.device)
            self.optimizer.zero_grad()
            model_outputs = self.model(x_batch, idx_batch, learn_kinetics=learn_kinetics)
            losses = self.loss_calculator(device=self.device, 
                                              adata=self.adata, 
                                              learn_kinetics=learn_kinetics, 
                                              K=self.K, 
                                              x=x_batch, 
                                              batch_indices=idx_batch, 
                                              model_out=model_outputs,
                                              current_epoch=current_epoch)
            losses["total_loss"].backward()
            self.optimizer.step()

            total_loss += losses["total_loss"].item() 
            recon_loss += losses["recon_loss"].item() if not learn_kinetics else 0
            kl_loss += losses["kl_loss"].item() if not learn_kinetics else 0
            heuristic_loss += losses["heuristic_loss"].item() if learn_kinetics else 0
            uniform_p_loss += losses["uniform_p_loss"].item() if learn_kinetics else 0
            kl_weight = losses["kl_weight"] if not learn_kinetics else 0
        
        self.loss_register["training"]["total_loss"].append(total_loss / len(self.train_dl))
        self.loss_register["training"]["recon_loss"].append(recon_loss / len(self.train_dl))
        self.loss_register["training"]["kl_loss"].append(kl_loss / len(self.train_dl))
        self.loss_register["training"]["heuristic_loss"].append(heuristic_loss / len(self.train_dl))
        self.loss_register["training"]["uniform_p_loss"].append(uniform_p_loss / len(self.train_dl))
        self.loss_register["training"]["kl_weight"].append(kl_weight)

        return losses["total_loss"]

    def eval_epoch(self, learn_kinetics, current_epoch):
        self.model.eval()
        total_loss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        heuristic_loss = 0.0
        uniform_p_loss = 0.0

        with torch.no_grad():
            for x_batch, idx_batch in self.test_dl:
                x_batch = x_batch.to(self.device)
                model_out = self.model(x_batch, learn_kinetics=learn_kinetics)
                losses = self.loss_calculator(self.device, self.adata, learn_kinetics, self.K, x_batch, idx_batch, model_out, current_epoch)
                total_loss += losses["total_loss"].item() 
                recon_loss += losses["recon_loss"].item() if not learn_kinetics else 0
                kl_loss += losses["kl_loss"].item() if not learn_kinetics else 0
                heuristic_loss += losses["heuristic_loss"].item() if learn_kinetics else 0
                uniform_p_loss += losses["uniform_p_loss"].item() if learn_kinetics else 0
                kl_weight = losses["kl_weight"].item() if not learn_kinetics else 0

            self.loss_register["evaluation"]["total_loss"].append(total_loss / len(self.train_dl))
            self.loss_register["evaluation"]["recon_loss"].append(recon_loss / len(self.train_dl))
            self.loss_register["evaluation"]["kl_loss"].append(kl_loss / len(self.train_dl))
            self.loss_register["evaluation"]["heuristic_loss"].append(heuristic_loss / len(self.train_dl)) 
            self.loss_register["evaluation"]["uniform_p_loss"].append(uniform_p_loss / len(self.train_dl)) 
            self.loss_register["evaluation"]["kl_weight"].append(kl_weight) 

        return losses["total_loss"]
    
    def check_params_frozen(self, learn_kinetics):
        for name, param in self.model.named_parameters():
            if 'encoder' in name or 'linear_decoder' in name:
                assert param.requires_grad == (not learn_kinetics), f"Parameter {name} is not correctly frozen."

    def adjust_learning_rates(self, learn_kinetics):
        """Adjust the learning rates for different parameter groups based on the current training regime."""
        for name, param in self.model.named_parameters():
            if 'kinetics_decoder' in name:
                param.requires_grad = learn_kinetics
            """elif 'linear_decoder' in name:
                param.requires_grad = not learn_kinetics
            #the next elif added on 11 sept 2024 to avoid losing GP identity via non linearity
            elif 'encoder' in name:
                param.requires_grad = not learn_kinetics"""
        
        """for name, param in self.model.named_parameters():
            if 'encoder' in name or 'linear_decoder' in name:
                param.requires_grad = not learn_kinetics
            elif 'kinetics_decoder' in name:
                param.requires_grad = learn_kinetics
            else:
                param.requires_grad = True  # Default for other parts"""

        for param_group in self.optimizer.param_groups:
            if 'tag' in param_group:
                if param_group['tag'] == 'encoder':
                    param_group['lr'] = self.base_lr  * (1e-1 if learn_kinetics else 1)
                elif param_group['tag'] == 'linear_decoder':
                    param_group['lr'] = self.base_lr  (1e-1 if learn_kinetics else 1)
                elif param_group['tag'] == 'kinetics_decoder':
                    param_group['lr'] = self.base_lr * (1e-1 if learn_kinetics else 0)
                else:
                    param_group['lr'] = self.base_lr  # Default scaling for other parts

    def train(self):
        print("")
        model_save_dir = f"outputs/{self.dataset_name}/model_checkpoints"
        losses_save_dir = f"outputs/{self.dataset_name}/losses_dic"
        time_save_dir = f"outputs/{self.dataset_name}/training_time"

        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(losses_save_dir, exist_ok=True)
        os.makedirs(time_save_dir, exist_ok=True)

        # Start timing the training process
        start_time = time.time()

        for epoch in range(self.n_epochs):
            learn_kinetics = epoch >= self.first_regime_end
            self.adjust_learning_rates(learn_kinetics)  # Adjust learning rates before each epoch
            loss_train = self.train_epoch(learn_kinetics, epoch)
            loss_eval = self.eval_epoch(learn_kinetics, epoch) if self.split_data else None

            if learn_kinetics or epoch == self.first_regime_end-1:
                if epoch % 25 == 0 or epoch == self.n_epochs - 1 or epoch == self.first_regime_end-1:
                    # Save the model
                    model_path = os.path.join(model_save_dir, f"model_epoch_{epoch}.pt")
                    retry = 3
                    for attempt in range(retry):
                        try:
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': loss_train,
                            }, model_path)
                            self.model_checkpoints[epoch] = model_path
                            self.losses[epoch] = loss_train
                        except RuntimeError as e:
                            print(f"Attempt {attempt + 1} failed with error: {e}")
                            if attempt < retry - 1:
                                print("Retrying...")
                            else:
                                raise

                    print(f"Epoch: {epoch}, train loss: {loss_train}, eval loss: {loss_eval}")

        # Save final losses dictionary
        final_losses_dic = self.write_final_losses()
        with open(os.path.join(losses_save_dir, "final_losses.pkl"), 'wb') as f:
            pickle.dump(final_losses_dic, f)

        # End timing the training process
        end_time = time.time()
        total_time = end_time - start_time

        # Save training time
        with open(os.path.join(time_save_dir, "training_time.txt"), 'w') as f:
            f.write(f"Total training time: {total_time:.2f} seconds")

        print(f"Total training time: {total_time:.2f} seconds")
        return self.model
    
    def load_second_regime_best_model(self):
        
        """
        Load the best or the last model from the second training regime.

        Parameters:
            load_best (bool): If True, load the model with the lowest loss in the second regime.
                            If False, load the last model saved in the second regime.

        Returns:
            model: The loaded model from the second regime.
        """
        # Filter epochs that correspond to the second training regime
        second_regime_epochs = [epoch for epoch in self.model_checkpoints if epoch >= self.first_regime_end]

        if not second_regime_epochs:
            print("No models saved during the second training regime.")
            return None

        if not self.load_last:
            # Load the model with the lowest loss during the second regime
            best_epoch = min(second_regime_epochs, key=lambda epoch: self.losses.get(epoch, float('inf')))
            model_path = self.model_checkpoints[best_epoch]
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded the best model from epoch {best_epoch} with loss {checkpoint['loss']}")
        else:
            # Load the last model saved during the second regime
            last_epoch = max(second_regime_epochs)
            model_path = self.model_checkpoints[last_epoch]
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded the last model from epoch {last_epoch} with loss {checkpoint['loss']}")

        return self.model
        
    def load_model(self, epoch):
        # Load the specific model checkpoint for the given epoch
        model_path = self.model_checkpoints[epoch]
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Loaded model from epoch {epoch} with loss {checkpoint['loss']}")

        return self.model

    def load_best_model(self):
        # Load the dictionary of losses
        with open(os.path.join(f"outputs/model_checkpoints", "losses.pkl"), 'rb') as f:
            self.losses = pickle.load(f)

        # Find the epoch with the lowest total loss
        #best_epoch = min(self.losses, key=self.losses.get)

        # Load the best model
        #best_model = self.load_model(best_epoch)
        best_model = self.load_second_regime_best_model()

        return best_model

    def self_extract_outputs(self):
        self.model.eval()  # Set model to evaluation mode
        #self.model = self.load_best_model()
        extract_outputs(self.adata, self.model, self.full_data_loader, self.device)