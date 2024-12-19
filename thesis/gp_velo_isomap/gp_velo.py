import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import sys
import os
import seaborn as sns
import pandas as pd

# Add the parent directory to the system path
parent_directory = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(parent_directory)
from plotting import plot_phase_plane_gp, gp_phase_plane_no_velocity


pairs_pancreas = [
    ("LUMINAL_EPITHELIAL_CELLS", "DUCTAL_CELLS"),
    ("METABOLISM_OF_CARBOHYDRATES", "GLUCOSE_METABOLISM"),
    ("METABOLISM_OF_LIPIDS_AND_LIPOP", "METABOLISM_OF_CARBOHYDRATES"),
    ("DELTA_CELLS", "BETA_CELLS"),
    ("GLUCAGON_SIGNALING_IN_METABOLI", "G_ALPHA_I_SIGNALLING_EVENTS")
]

pairs_forebrain = [
    ("EMBRYONIC_STEM_CELLS", "NEURAL_STEM-PRECURSOR_CELLS"),
    ("ASTROCYTES", "LOSS_OF_NLP_FROM_MITOTIC_CENTR")
]

pairs_gastrulation_erythroid = [
    ("ERYTHROID-LIKE_AND_ERYTHROID_P", "IRON_UPTAKE_AND_TRANSPORT"),
    ("METABOLISM_OF_PROTEINS", "ERYTHROID-LIKE_AND_ERYTHROID_P")
]

pairs_dentategyrus_lamanno_P5 = [
    ("NEURONS", "OLIGODENDROCYTE_PROGENITOR_CEL"),
    ("ASTROCYTES", "OLIGODENDROCYTE_PROGENITOR_CEL"),
    ("NEURONS", "NEURONAL_SYSTEM"),
    ("NEURONAL_SYSTEM", "PYRAMIDAL_CELLS")
]

pairs_dic = {
    "forebrain" : pairs_forebrain,
    "pancreas" : pairs_pancreas,
    "gastrulation_erythroid" : pairs_gastrulation_erythroid,
    "dentategyrus_lamanno_P5" : pairs_dentategyrus_lamanno_P5
}

datasets = ["forebrain", "pancreas", "gastrulation_erythroid", "dentategyrus_lamanno_P5"]
cell_type_keys = ["Clusters", "clusters", "celltype"] #, "clusters"]
#model_names = ["imVelo", "ivelo", "ivelo_filtered"]
model_names = ["expimap"]
datasets = ["dentategyrus_lamanno_P5"]
cell_type_keys = ["clusters"]


def change_colors(adata):
    adata.obs[cell_type_key] = [str(cat) for cat in list(adata.obs[cell_type_key])]
    adata.obs[cell_type_key] = pd.Series(adata.obs[cell_type_key], dtype="category")
    unique_categories = adata.obs[cell_type_key].cat.categories
    rgb_colors = sns.color_palette("tab20", len(unique_categories))
    hex_colors = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b in rgb_colors]
    adata.uns[f"{cell_type_key}_colors"] = hex_colors
    return adata


for model_name in model_names:
    print(f"model name: {model_name}")
    for dataset, cell_type_key in zip(datasets, cell_type_keys):
        print(f"dataset: {dataset}")
        adata = sc.read_h5ad(f"../../benchmark/{model_name}/{dataset}/adata_gp.h5ad")
        adata = change_colors(adata)
        for gp_names in pairs_dic[dataset]:
            gp1, gp2 = gp_names
            print(f"gp1: {gp1}, gp2: {gp2}")
            if (gp1 in list(adata.var_names)) and (gp2 in list(adata.var_names)):
                os.makedirs(f"plots/gp_velo/{dataset}/{gp1}_vs_{gp2}/", exist_ok=True)
                os.makedirs(f"plots/gp_no_velo/{dataset}/{gp1}_vs_{gp2}/", exist_ok=True)
                if not model_name == "expimap":
                    plot_phase_plane_gp(adata, gp1, gp2, u_scale=.1, s_scale=.1, 
                        cell_type_key=cell_type_key, dataset=dataset, 
                        K=11, save_path= f"plots/gp_velo/{dataset}/{gp1}_vs_{gp2}/{model_name}.png", 
                        save_plot=True, scale_expression=1)
                gp_phase_plane_no_velocity(adata, gp1, gp2, dataset=dataset, K=11, show_plot=True, save_plot=True, 
                                        save_path=f"plots/gp_no_velo/{dataset}/{gp1}_vs_{gp2}/{model_name}.png", 
                                        cell_type_key=cell_type_key, scale_expression=1)
            else:
                print(f"skippin'")


