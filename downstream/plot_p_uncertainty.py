import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from plotting import compute_velocity_sign_uncertainty

# Assuming compute_velocity_sign_uncertainty is already imported

datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
gene_discrepancy = {}

# Create folder to save plots if not already present
import os
if not os.path.exists("plot_sign_uncertainty"):
    os.makedirs("plot_sign_uncertainty")

# Collect gene uncertainty data and generate UMAP plots for cell uncertainty
for dataset in datasets:
    adata = sc.read_h5ad(f"benchmark/imVelo/{dataset}/imVelo_{dataset}.h5ad")
    
    # Compute velocity sign uncertainty
    compute_velocity_sign_uncertainty(adata, aggregate_method='mean')
    adata.obs["sign_discrepancy_cell"] = adata.obs["p_cell_uncertainty"].copy()
    adata.var["sign_discrepancy_gene"] = adata.var["p_gene_uncertainty"].copy()
    # Generate UMAP plot for cell uncertainty
    sc.pl.umap(adata, color="sign_discrepancy_cell", title=f"Mean sign discrepancy of cells: {dataset}")
    plt.savefig(f"plot_sign_uncertainty/sign_uncertainty_{dataset}.png", bbox_inches="tight")
    
    # Store gene uncertainty for later use in violin plot
    gene_discrepancy[dataset] = adata.var["sign_discrepancy_gene"].copy()

# Combine data into a DataFrame for violin plotting
plot_data = pd.DataFrame({
    "dataset": [],
    "gene_discrepancy": []
})

for dataset in datasets:
    dataset_data = pd.DataFrame({
        "dataset": [dataset] * len(gene_discrepancy[dataset]),
        "sign_discrepancy_gene": gene_discrepancy[dataset]
    })
    plot_data = pd.concat([plot_data, dataset_data])

# Create violin plot for gene uncertainty across datasets
plt.figure(figsize=(10, 6))

# Violin plot
sns.violinplot(x="dataset", y="sign_discrepancy_gene", data=plot_data, inner=None)

# Add points for individual observations
sns.stripplot(x="dataset", y="sign_discrepancy_gene", data=plot_data, 
              color="black", jitter=True, size=3, alpha=0.6)

# Add titles and labels
plt.title("Mean sign discrepancy of genes")
plt.ylabel("Gene Sign Discrepancy")
plt.xlabel("Dataset")
plt.xticks(rotation=45)
plt.tight_layout()

# Save the figure
plt.savefig("plot_sign_uncertainty/gene_uncertainty_violin_plots_with_points.png", bbox_inches="tight")

# Display the violin plot with points
plt.show()
