#%%
# =============================================================================
# Step 1: Import Libraries
# =============================================================================
import os
from re import L
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FuncFormatter



#%%
# =============================================================================
# Step 3: Data Reading Function
# =============================================================================
def read_csv_data(csv_root, csv_files, metric_name='acc'):
    """
    Reads data from multiple CSV files and returns a list of pandas DataFrames.

    Args:
        csv_root (str): The root directory where the CSV files are located.
        csv_files (list): A list of CSV file names to process.
        metric_name (str): The name of the metric to plot (e.g., 'acc', 'miou').

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
            train_col = f'train_{metric_name}'
            valid_col = f'valid_{metric_name}'
            required_cols = {'epoch', train_col, valid_col}
            if not required_cols.issubset(df.columns):
                print(f"Warning: Skipping {file_name} as it's missing one of {required_cols}.")
                continue

            # Get the base name for the legend (e.g., 'small_pos' from 'small_pos.csv')
            base_name = os.path.splitext(file_name)[0]
            data_to_plot.append((base_name, df))

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    
    return data_to_plot

#%%
def read_csv_folder(csv_root, metric_names, epoch, sort_by_metric=True):
    """
    Reads all CSV files in a folder and extracts a specific metric at a given epoch.
    The results are returned as a pandas DataFrame and saved to a new CSV file.

    Args:
        csv_root (str): The root directory where the CSV files are located.
        metric_names (list): A list of metric column names to extract (e.g., ['valid_acc', 'valid_miou']).
        epoch (int): The epoch number to look for.
        sort_by_metric (bool): If True, sorts the resulting DataFrame by the metric value.

    Returns:
        pd.DataFrame: A DataFrame with 'Model' and metric columns, or an empty DataFrame if no data is found.
    """
    results_data = []
    if not os.path.isdir(csv_root):
        print(f"Error: Directory not found at {csv_root}")
        return pd.DataFrame()

    # Generate a safe filename for the summary CSV
    summary_filename = f"summary_epoch_{epoch}_{metric_names[0]}.csv"

    for file_name in os.listdir(csv_root):
        if not file_name.endswith('.csv'):
            continue

        # Avoid reading the summary file this function might create
        if file_name == summary_filename:
            continue

        file_path = os.path.join(csv_root, file_name)
        try:
            df = pd.read_csv(file_path)
            if not 'epoch' in df.columns:
                # print(f"Warning: Skipping {file_name} as it's missing epoch.")
                continue

            epoch_row = df[df['epoch'] == epoch]

            if epoch_row.empty:
                # print(f"Warning: Epoch {epoch} not found in {file_name}. Skipping.")
                continue
            

            model_metrics = {'Model': os.path.splitext(file_name)[0]}
            metrics_found = False
            for metric in metric_names:
                if metric in df.columns:
                    model_metrics[metric] = epoch_row[metric].iloc[0]
                    metrics_found = True
                else:
                    model_metrics[metric] = None  # Or np.nan
            
            if metrics_found:
                results_data.append(model_metrics)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    if not results_data:
        return pd.DataFrame()

    results_df = pd.DataFrame(results_data)
    if sort_by_metric and metric_names:
        results_df = results_df.sort_values(by=metric_names[0], ascending=False).reset_index(drop=True)

    # Save the results to a new CSV file in the same directory
    output_path = os.path.join(csv_root, summary_filename)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    return results_df

#%%
# =============================================================================
# Step 4: Plotting Function
# =============================================================================
def plot_data(data_to_plot, show_train_acc=True, max_epoch=None, min_epoch=None, title=None, filname=None,  legend_labels=None, output_dir=None, figsize=(6, 4), metric_name='acc', y_formatter='{y:.0%}', y_label=None):
    """
    Plots training and validation accuracies from a list of DataFrames.

    Args:
        data_to_plot (list): A list of tuples, each containing (base_name, DataFrame).
        show_train_acc (bool): Whether to show training accuracy.
        max_epoch (int, optional): Maximum epoch to plot.
        min_epoch (int, optional): Minimum epoch to plot.
        title (str, optional): The title for the plot. Defaults to a generic title.
        output_dir (str, optional): Directory to save the plot.
        figsize (tuple, optional): Figure size for the plot.
        metric_name (str): The name of the metric to plot (e.g., 'acc', 'miou').
        y_formatter (str, optional): A format string for the y-axis ticks.
    """
    if not data_to_plot:
        print("No data to plot. Exiting.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    # Use a color palette designed for many categories to ensure colors are distinct.
    # 'husl' (Hue, Saturation, Lightness) provides evenly spaced, vibrant colors.
    # Other good options include 'hls' or a perceptually uniform one like 'viridis'.
    num_files = len(data_to_plot)
    # colors = sns.color_palette('husl', n_colors=num_files)
    colors = sns.color_palette('tab10', n_colors=max(num_files, 10))
    line_styles = ['--', '-']  # Solid for train_acc, dashed for valid_acc

    train_col = f'train_{metric_name}'
    valid_col = f'valid_{metric_name}'
    min_epoch_val = min_epoch or 0
    acc_values = []
    for i, (base_name, df) in enumerate(data_to_plot):
        plot_df = df
        # Filter the DataFrame based on min_epoch and max_epoch
        if min_epoch is not None:
            plot_df = plot_df[plot_df['epoch'] >= min_epoch]
        if max_epoch is not None:
            plot_df = plot_df[plot_df['epoch'] <= max_epoch]

        if plot_df.empty:
            continue

        if show_train_acc:
            # Plot training accuracy
            ax.plot(
                plot_df['epoch'], 
                plot_df[train_col], 
                label=f'{base_name} - Train {metric_name.capitalize()}', 
                color=colors[i], 
                linestyle=line_styles[0],
                # marker='o',
                markersize=4
            )
            acc_values.append(plot_df[train_col].min())
            acc_values.append(plot_df[train_col].max())
        
        # Plot validation accuracy
        ax.plot(
            plot_df['epoch'], 
            plot_df[valid_col], 
            label=f'{base_name} - Valid {metric_name.capitalize()}', 
            color=colors[i], 
            linestyle=line_styles[1],
            # marker='x',
            markersize=5
        )
        acc_values.append(plot_df[valid_col].min())
        acc_values.append(plot_df[valid_col].max())
    
    # --- Auto-adjust y-axis ---
    min_y = min(acc_values) if acc_values else 0
    max_y = max(acc_values) if acc_values else 1.0

    # if title is None:
    #     title = 'Training and Validation Accuracy Comparison'

    # --- Formatting the plot ---
    if title is not None:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=12)
    else:
        ax.set_ylabel(metric_name.capitalize(), fontsize=12)
    
    # Improve legend placement
    if legend_labels is not None:
        ax.legend(legend_labels, loc='lower right',  fontsize='medium')
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Set y-axis to percentage format
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: y_formatter.format(y=y)))
    # Set a dynamic y-limit, starting slightly below the minimum observed accuracy.
    ax.set_ylim(bottom=max(0, min_y - 0.0), top=min(1.0, max_y + 0.01)) 
    
    if max_epoch is not None:
        ax.set_xlim(left=min_epoch_val, right=max_epoch)
    else:
        ax.set_xlim(left=min_epoch_val)

    # Ensure x-axis shows integer epoch numbers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout() # Adjust layout

    if output_dir:
        safe_filename = "".join(c for c in filname if c.isalnum() or c in (' ', '_', '-')).rstrip()
        safe_filename = safe_filename.replace(' ', '_') + f'{'_train' if show_train_acc else ''}.png'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, safe_filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
# %%
