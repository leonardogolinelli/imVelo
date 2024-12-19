
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

def compute_cell_pops(adata, cell_type_key, batch_key, save_path):
    # Cross-tabulate counts of cell types across stages
    counts = pd.crosstab(adata.obs[cell_type_key], adata.obs[batch_key])

    # Calculate the total counts for each stage across all cell types
    total_counts_per_stage = counts.sum(axis=0)

    # Calculate the percentage of each cell type relative to all cell types for each stage
    celltype_percent = (counts / total_counts_per_stage) * 100

    # Create the stacked bar plot using matplotlib directly
    fig, ax = plt.subplots(figsize=(14, 10))  # Increased figure size for readability

    # Plot each cell type separately to form a stacked bar
    bottom = None
    for cell_type in celltype_percent.index:
        ax.bar(celltype_percent.columns, celltype_percent.loc[cell_type],
            label=cell_type, bottom=bottom)
        # Update bottom for stacking
        if bottom is None:
            bottom = celltype_percent.loc[cell_type]
        else:
            bottom += celltype_percent.loc[cell_type]

    # Customize the plot with increased font sizes
    ax.set_xlabel('Stage', fontsize=24)
    ax.set_title('Percentage of Each Cell Type Across Stages', fontsize=22)
    ax.set_xticks(range(len(celltype_percent.columns)))
    ax.set_xticklabels(celltype_percent.columns, rotation=45, fontsize=20)
    ax.set_ylim(0, 110)  # Extend y-axis to accommodate full 100%

    # Set y-ticks to integers
    ax.set_yticks(range(0, 111, 10))
    ax.set_yticklabels([str(i) for i in range(0, 111, 10)], fontsize=20)

    # Add the total number of cells on top of each bar
    for i, stage in enumerate(celltype_percent.columns):
        total_cells = total_counts_per_stage[stage]
        ax.text(i, 102, f'{total_cells}', ha='center', fontsize=18)  # Position above 100%

    # Add a legend entry with only a point or a number for total cells
    handles, labels = ax.get_legend_handles_labels()
    total_cells_overall = total_counts_per_stage.sum()  # Calculate the overall total
    # Add a single point or number for "Total Cells"
    handles.append(plt.Line2D([0], [0], color='None', label=f'Total Cells', markersize=10, linestyle='None'))
    ax.legend(handles=handles, title='Cell Type', fontsize=20, title_fontsize=20, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

def plot_cell_type_distribution(adata, cell_type_key, dataset, save_path):
    # Count the occurrences of each cell type
    cell_type_counts = adata.obs[cell_type_key].value_counts()

    # Convert counts to percentages
    cell_type_percentages = (cell_type_counts / cell_type_counts.sum()) * 100

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(cell_type_percentages.index, cell_type_percentages.values, 
                   color="skyblue", edgecolor="black")

    # Add absolute numbers on top of each bar
    for bar, absolute_value in zip(bars, cell_type_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                 f"{absolute_value}", ha='center', va='bottom', fontsize=10)

    # Customize the plot
    plt.title(f"Cell Type Distribution - {dataset}", fontsize=16)
    plt.xlabel("Cell Type", fontsize=14)
    plt.ylabel("Percentage", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    
    # Fix yticks to include both ticks and labels
    current_ticks = plt.gca().get_yticks()
    plt.yticks(ticks=current_ticks, labels=[f"{int(tick)}%" for tick in current_ticks], fontsize=12)

    plt.tight_layout()

    # Save and show the plot
    plt.savefig(save_path)
    plt.show()
