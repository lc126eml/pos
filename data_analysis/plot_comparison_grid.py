#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import os

#%%
# =============================================================================
# Step 3: Data Reading Function
# =============================================================================
def read_csv_data(csv_root, csv_files):
    """
    Reads data from multiple CSV files and returns a list of pandas DataFrames.

    Args:
        csv_root (str): The root directory where the CSV files are located.
        csv_files (list): A list of CSV file names to process.

    Returns:
        list: A list of tuples, where each tuple contains (base_name, DataFrame).
    """
    data_to_plot = []
    for file_name in csv_files:
        file_path = os.path.join(csv_root, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found at {file_path}. Skipping.")
            continue
            
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            required_cols = {'epoch', 'train_acc', 'valid_acc'}
            if not required_cols.issubset(df.columns):
                print(f"Warning: Skipping {file_name} as it's missing one of {required_cols}.")
                continue

            # Get the base name for the legend (e.g., 'small_pos' from 'small_pos.csv')
            base_name = os.path.splitext(file_name)[0]
            data_to_plot.append((base_name, df))

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    
    return data_to_plot

# --- Helper Function: Plots data onto a single given axis ---
def plot_on_ax(ax, data_to_plot, colors, show_train_acc=True, max_epoch=None):
    """
    Helper function to plot accuracy data on a single Matplotlib axis.
    
    Args:
        ax (matplotlib.axes.Axes): The axis object to plot on.
        data_to_plot (list): A list of tuples, each containing (base_name, DataFrame).
        colors (list): A list of colors to use for plotting.
        show_train_acc (bool): Whether to plot training accuracy.
        max_epoch (int, optional): The maximum epoch to display.
    """
    # Define line styles for clarity: solid for validation, dashed for training
    line_styles = {'valid_acc': '-', 'train_acc': '--'}

    for i, (base_name, df) in enumerate(data_to_plot):
        plot_df = df
        if max_epoch is not None:
            plot_df = df[df['epoch'] <= max_epoch].copy()
        
        if plot_df.empty:
            continue

        # Plot validation accuracy (solid line)
        ax.plot(
            plot_df['epoch'], 
            plot_df['valid_acc'], 
            label=f'{base_name} - Valid Acc', 
            color=colors[i], 
            linestyle=line_styles['valid_acc']
        )
        
        # Plot training accuracy (dashed line), if enabled
        if show_train_acc:
            ax.plot(
                plot_df['epoch'], 
                plot_df['train_acc'], 
                label=f'{base_name} - Train Acc', 
                color=colors[i], 
                linestyle=line_styles['train_acc']
            )
    
    # --- Auto-adjust y-axis ---
    # Find the minimum accuracy value across all plotted data to set a sensible lower limit.
    acc_values = []
    for _, df_orig in data_to_plot:
        df = df_orig
        if max_epoch is not None:
            df = df_orig[df_orig['epoch'] <= max_epoch]
        if df.empty:
            continue
        acc_values.append(df['valid_acc'].min())
        acc_values.append(df['valid_acc'].max())
        if show_train_acc:
            acc_values.append(df['train_acc'].min())
            acc_values.append(df['train_acc'].max())
    
    min_y = min(acc_values) if acc_values else 0
    max_y = max(acc_values) if acc_values else 1.0    
    # Apply grid styling to the individual axis
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylim(bottom=max(0, min_y - 0.0), top=min(1.0, max_y + 0.02)) 
    
    if max_epoch is not None:
        ax.set_xlim(left=0, right=max_epoch)
    else:
        ax.set_xlim(left=0)
#%%
# --- Main Function: Creates the 3x3 Facet Grid ---
def plot_comparison_grid(dataset_names, model_types, CSV_ROOT_DIRS, row_titles, col_titles, legend_labels=None,show_train_acc=True, max_epoch=None, output_dir=None):
    """
    Creates and saves a 3x3 grid of accuracy plots.

    Args:
        all_data (dict): A dictionary mapping a unique key (e.g., 'small_cifar') 
                         to the data for that plot (list of tuples).
        row_titles (list): A list of 3 strings for the row titles.
        col_titles (list): A list of 3 strings for the column titles.
        show_train_acc (bool): Whether to plot training accuracy.
        max_epoch (int, optional): The maximum epoch to display.
        output_dir (str, optional): Directory to save the final plot.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    # Create a 3x3 grid of subplots with shared X and Y axes sharey='row'
    fig, axes = plt.subplots(
        len(dataset_names),
        len(model_types),
        figsize=(6*len(model_types), 4*len(dataset_names)),
        sharex=True,
        sharey='row',  # share y-axis per row (both columns)
        squeeze=False  # keep axes 2D even if one dimension is 1
    )

    # Use a clear color palette
    colors = sns.color_palette('tab10', n_colors=10)
    
    # --- Determine Global Y-Axis Limits ---
    # Calculate min/max accuracy across ALL datasets for consistent axis scaling
    # all_acc_values = []
    # for data_list in all_data.values():
    #     for _, df_orig in data_list:
    #         df = df_orig
    #         if max_epoch:
    #             df = df_orig[df_orig['epoch'] <= max_epoch]
    #         if not df.empty:
    #             all_acc_values.extend(df['valid_acc'].values)
    #             if show_train_acc:
    #                 all_acc_values.extend(df['train_acc'].values)
    
    # min_y = min(all_acc_values) if all_acc_values else 0
    # max_y = max(all_acc_values) if all_acc_values else 1.0

    # --- Plot Data on Each Subplot ---
    # plot_map = [
    #     ['small_imagenet100', 'base_imagenet100', 'large_imagenet100'],
    #     ['small_imagenet10',  'base_imagenet10',  'large_imagenet10'],
    #     ['small_cifar',       'base_cifar',       'large_cifar']
    # ]

    for i, dataset_name in enumerate(dataset_names): # Rows
        all_acc_values = []
        for j, model_type in enumerate(model_types): # Columns
            ax = axes[i, j]
            data_key = f"{model_type}_{dataset_name}"
            data_to_plot = read_csv_data(CSV_ROOT_DIRS[dataset_name], get_csv_filenames(model_type, dataset_name))
            for _, df_orig in data_to_plot:
                df = df_orig
                if max_epoch:
                    df = df_orig[df_orig['epoch'] <= max_epoch]
                if not df.empty:
                    all_acc_values.extend(df['valid_acc'].values)
                    if show_train_acc:
                        all_acc_values.extend(df['train_acc'].values)
            plot_on_ax(ax, data_to_plot, colors, show_train_acc, max_epoch)

        # Align y-limits across the row so both columns share the same ticks
        min_y = min(all_acc_values) if all_acc_values else 0
        max_y = max(all_acc_values) if all_acc_values else 1.0
        axes[i, 0].set_ylim(bottom=max(0, min_y - 0.00), top=min(1.0, max_y + 0.02))

    # --- Formatting the Grid ---
    # Set column and row titles
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=16, pad=8)
    # for i, title in enumerate(row_titles):
    #     # Use the ylabel of the first column as the row title
    #     axes[i, 0].set_ylabel(title, fontsize=16, labelpad=8)

    # Set shared axis labels for the entire figure
    fig.supxlabel('Epoch', fontsize=20, y=0.0) # Adjust y to control vertical position
    fig.supylabel('Top-1 Acc. (%)', fontsize=20, x=0.02) # Adjust x to control horizontal position

    # Format shared axes
    # The ylim is now set per-row inside the loop, so this global setting is not needed.
    if max_epoch:
        plt.setp(axes, xlim=(0, max_epoch))
    
    # Apply formatters to the shared axes (only need to do it once)
    # axes[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    axes[len(dataset_names)-1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Create a Single, Shared Legend ---
    # Get handles and labels from one of the plots to create the legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if legend_labels is not None:
        labels = legend_labels

    # Place legend centrally above the plots
    # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize='large')
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.96, 0.2), fontsize='large')

    # Adjust layout to prevent titles/labels from overlapping
    fig.tight_layout(rect=[0, 0.01, 1, 1]) # rect leaves space at the top for the legend

    # --- Save or Show the Plot ---
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, 'model_comparison_grid.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Plot grid saved to {output_path}")
    else:
        plt.show()

def get_csv_filenames(MODEL_TYPE, DATASET):
    if MODEL_TYPE == 'small':
        CSV_FILE_NAMES = [
            "small_overlap_0_rc_False_alpha_600lr100.csv",
            "small_abs_pos_overlap_0_rc_False_alpha_600lr100.csv",
            "small_rot_pos_overlap_0_rc_False_alpha_20lr100.csv",
            "small_overlap_0_rc_True_alpha_300lr100.csv",
        ]
    else:
        CSV_FILE_NAMES = [
            "base_overlap_0_rc_False_alpha_600.0.csv",
            "base_abs_pos_overlap_0_rc_False_alpha_600.0.csv",
            "base_rot_pos_overlap_0_rc_False_alpha_600lr50.csv",
            "base_overlap_0_rc_True_alpha_600lr50.csv",
        ]
    return CSV_FILE_NAMES
CSV_ROOT_DIRS = {
    # 'cifar': r'D:\codes\working\pos\Draft\csv\cifar',
    # 'imagenet10': r'D:\codes\working\pos\Draft\csv\imagenet10',
    'imagenet100': r'D:\codes\working\pos\Draft\csv\redo20251215\dinov3b392s55of0dynamic_224_',
}

# --- 3. Define titles and plotting parameters ---, 'imagenet10', 'ImageNet-10'
model_types = ['small', 'base']
dataset_names = ['imagenet100']
row_titles = ['ImageNet-100']
col_titles = ['ViT-S (DINOv3)', 'ViT-B (DINOv3)']
# legend_labels = ['Position-Agnostic (No PE)', 'Absolute learned', 'RoPE', 'Ours (Guidance)']
legend_labels = ['No PE', 'AbsPE', 'RoPE', 'Ours (Guid.)']
show_train_acc = False # Set to False to only show validation accuracy
max_epoch_to_show = 130
output_directory = r'D:\codes\working\pos\Draft\plots' # Or set to None to display interactively

# --- 4. Call the main function to generate the plot ---
plot_comparison_grid(
    dataset_names=dataset_names,
    model_types=model_types,
    CSV_ROOT_DIRS=CSV_ROOT_DIRS, 
    row_titles=row_titles,
    col_titles=col_titles,
    legend_labels=legend_labels,
    show_train_acc=show_train_acc,
    max_epoch=max_epoch_to_show,
    output_dir=output_directory
)
# %%
