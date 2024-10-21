import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import scanpy as sc


class CustomDataset(Dataset):
    def __init__(self, adata):
        unspliced = torch.tensor(adata.layers["Mu"], dtype=torch.float32)
        spliced = torch.tensor(adata.layers["Ms"], dtype=torch.float32)
        self.x = torch.cat([unspliced, spliced], dim=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], idx
    

def setup_dataloaders(adata, batch_size, train_size=0.8, split_data=True):
    custom_dataset = CustomDataset(adata)
    if split_data:
        num_samples = len(custom_dataset)
        indices = np.random.permutation(num_samples)
        split = int(train_size * num_samples)
        train_indices, test_indices = indices[:split], indices[split:]
        adata.uns["train_indices"] = train_indices
        adata.uns["test_indices"] = test_indices

        print(f"number of training observations: {len(train_indices)}")
        print(f"number of test observations: {len(test_indices)}")
        
        train_subset = Subset(custom_dataset, train_indices)
        test_subset = Subset(custom_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    else:
        train_subset = custom_dataset
        test_loader = None

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    full_data_loader = DataLoader(custom_dataset, batch_size=256, shuffle=False)  # Simplified DataLoader

    return train_loader, test_loader, full_data_loader
