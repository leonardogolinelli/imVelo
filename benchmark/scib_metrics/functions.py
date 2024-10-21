
#The below methods are adapted from scib_metrics
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
import pandas as pd

def get_results(bm, min_max_scale: bool = False, clean_names: bool = True) -> pd.DataFrame:
    '''Return the benchmarking results.

    Parameters
    ----------
    min_max_scale
        Whether to min max scale the results.
    clean_names
        Whether to clean the metric names.

    Returns
    -------
    The benchmarking results.
    '''
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

def plot_results_table(df, bm, save_path: str = None, show: bool = True, ) -> Table:
    '''Plot the benchmarking results.

    Parameters
    ----------
    show
        Whether to show the plot.
    save_dir
        The directory to save the plot to. If `None`, the plot is not saved.
    '''
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
    fig.savefig(save_path, facecolor=ax.get_facecolor(), dpi=300)

    return tab

