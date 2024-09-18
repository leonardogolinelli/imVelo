import pandas as pd
import scanpy as sc
import scvelo as scv

import scib_metrics
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection


import matplotlib.pyplot as plt
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
import os


filtered_adatas = {}
cell_type_key = "clusters"

path = "../../../outputs_old/final_anndatas/pancreas"
mivelo_path = f"{path}/mivelo.h5ad"
celldancer_path = f"{path}/celldancer.h5ad"
baseline_path = f"{path}/baseline.h5ad"
scvelo_path = f"{path}/scvelo.h5ad"
stochastic_path = f"{path}/stochastic.h5ad"
ivelo_noproc_path = f"{path}/ivelo_noproc.h5ad"
ivelo_proc_path = f"{path}/ivelo_proc.h5ad"
velovi_noproc_path = f"{path}/velovi_noproc.h5ad"
velovi_proc_path = f"{path}/velovi_proc.h5ad"
expimap_path = f"{path}/expimap.h5ad"
manifold_path = f"{path}/manifold.h5ad"

paths = [mivelo_path, scvelo_path, stochastic_path, celldancer_path, velovi_noproc_path, ivelo_noproc_path,
             velovi_proc_path, ivelo_proc_path, baseline_path, expimap_path, manifold_path]
names = ["mivelo", "scvelo", "steadystate_stochastic", "celldancer", "velovi_noproc", "ivelo_noproc", "velovi_proc", 
         "ivelo_proc", "baseline", "expimap", "dt"]

adata = sc.read_h5ad(paths[0])
adata.obsm["z_mivelo"] = adata.obsm["z"]
adata.obs["batch"] = 0

for i, path in enumerate(paths):
    if names[i] in ["velovi_noproc","ivelo_noproc", "expimap", "dt"]:
        query_adata = sc.read_h5ad(path)
        if names[i] == "dt":
            adata.obsm[f"z_{names[i]}"] = query_adata.obsm["z5"]
        else:
            adata.obsm[f"z_{names[i]}"] = query_adata.obsm["z"]


z_names = ["z_mivelo","z_velovi_noproc","z_ivelo_noproc", "z_expimap", "z_dt"]
metric_names = ["isolated_labels","nmi_ari_cluster_labels_kmeans", "silhouette_label", "klisi_knn"]


adata.write_h5ad("adata_latents.h5ad")
adata = sc.read_h5ad("adata_latents.h5ad")
bio_metrics = BioConservation()
batch_metrics = BatchCorrection(
    False, False, False, False, False
)

bm = Benchmarker(
    adata,
    batch_key="batch",
    label_key="clusters",
    embedding_obsm_keys=z_names,
    bio_conservation_metrics=bio_metrics,
    batch_correction_metrics=batch_metrics,
    n_jobs=1,
)
bm.benchmark()

#The below methods are adapted from scib_metrics
from sklearn.preprocessing import MinMaxScaler

def get_results( min_max_scale: bool = False, clean_names: bool = True) -> pd.DataFrame:
    """Return the benchmarking results.

    Parameters
    ----------
    min_max_scale
        Whether to min max scale the results.
    clean_names
        Whether to clean the metric names.

    Returns
    -------
    The benchmarking results.
    """
    _METRIC_TYPE = "Metric Type"
    # Mapping of metric fn names to clean DataFrame column names
    metric_name_cleaner = {
        "silhouette_label": "Silhouette label",
        "silhouette_batch": "Silhouette batch",
        "isolated_labels": "Isolated labels",
        "nmi_ari_cluster_labels_leiden_nmi": "Leiden NMI",
        "nmi_ari_cluster_labels_leiden_ari": "Leiden ARI",
        "nmi_ari_cluster_labels_kmeans_nmi": "KMeans NMI",
        "nmi_ari_cluster_labels_kmeans_ari": "KMeans ARI",
        "clisi_knn": "cLISI",
        "ilisi_knn": "iLISI",
        "kbet_per_label": "KBET",
        "graph_connectivity": "Graph connectivity",
        "pcr_comparison": "PCR comparison",
    }

    df = bm._results.transpose()
    df.index.name = "Embedding"
    
    df = df.loc[df.index != _METRIC_TYPE]
    if min_max_scale:
        # Use sklearn to min max scale
        df = pd.DataFrame(
            MinMaxScaler().fit_transform(df),
            columns=df.columns,
            index=df.index,
        )
    if clean_names:
        df = df.rename(columns=metric_name_cleaner)
    df = df.transpose()
    df[_METRIC_TYPE] = bm._results[_METRIC_TYPE].values
    df=df.transpose()

    return df

df = get_results()


def plot_results_table(df, show: bool = True, save_dir: str = None) -> Table:
    """Plot the benchmarking results.

    Parameters
    ----------
    show
        Whether to show the plot.
    save_dir
        The directory to save the plot to. If `None`, the plot is not saved.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np

    _METRIC_TYPE = "Metric Type"
    num_embeds = len(bm._embedding_obsm_keys)
    cmap_fn = lambda col_data: normed_cmap(col_data, cmap=matplotlib.cm.PRGn, num_stds=2.5)
    # Do not want to plot what kind of metric it is
    plot_df = df.iloc[:-1, :]
    # Sort by total score
    # plot_df = plot_df.sort_values(by="Total", ascending=False).astype(np.float64)
    plot_df["Method"] = plot_df.index

    # Split columns by metric type, using df as it doesn't have the new method col
    cols = df.columns
    column_definitions = [
        ColumnDefinition("Method", width=4, textprops={"ha": "left", "weight": "bold"}),
    ]
    # Circles for the metric values
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=1,
            textprops={
                "ha": "center",
                "bbox": {"boxstyle": "circle", "pad": 0.25},
            },
            cmap=cmap_fn(plot_df[col]),
            group=df.loc[_METRIC_TYPE, col],
            formatter="{:.2f}",
        )
        for i, col in enumerate(cols)
    ]

    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(len(df.columns) * 1.25, 3 + 0.3 * num_embeds))
        tab = Table(
            plot_df,
            cell_kw={
                "linewidth": 0,
                "edgecolor": "k",
            },
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 8, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(colnames=plot_df.columns)
    plt.show()
    fig.savefig("benchmark.png", facecolor=ax.get_facecolor(), dpi=300)

    return tab


plot_results_table(df)