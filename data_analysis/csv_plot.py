#%%
# =============================================================================
# Step 1: Import Libraries
# =============================================================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FuncFormatter
from csv_utils import read_csv_data, plot_data
#%%
show_train_acc = False
max_epoch_to_show = 130 # Set to None to show all epochs
# Define the root directory for your CSV files
CSV_ROOT_DIR = r'D:\codes\working\pos\Draft\csv\dinov3imagenet100'
DATASET = "imagenet100"

metric_name = 'acc'
legend_labels = ['DINOv3 Baseline (with Rope)', 'Position-Agnostic (No PE)', 'Ours (Guidance Only)', 'Ours (Full Method)'] #, 
MODEL_TYPE = 'base'
y_label = "Accuracy"
# MODEL_TYPE = 'small'
# MODEL_TYPE = 'large'
CSV_FILE_NAMES = [
    f'base_pos_overlap_0_rc_False_classes_30.0.csv',
    f'base_overlap_0_rc_False_classes_30.0.csv',
    f'base_overlap_0_rc_True_classes_30.0.csv',
    f'base_overlap_1_rc_True_classes_30.0.csv',
]

output_dir = rf"D:\codes\working\pos\Draft\plots\{'show_train' if show_train_acc else 'no_train'}"
# print("Reading data...")
all_data = read_csv_data(CSV_ROOT_DIR, CSV_FILE_NAMES)
# print("Plotting data...")
plot_title = f'Accuracy Comparison for DINOv3 {MODEL_TYPE.capitalize()} Model on {DATASET.capitalize()}'
# plot_data(all_data, show_train_acc=show_train_acc, max_epoch=max_epoch_to_show, title=plot_title)

plot_data(all_data, show_train_acc=show_train_acc, max_epoch=max_epoch_to_show, title=None, filname=plot_title, legend_labels=legend_labels, output_dir=output_dir, metric_name=metric_name, y_label=y_label)

#%%
show_train_acc = False
max_epoch_to_show = 130 # Set to None to show all epochs
# Define the root directory for your CSV files
CSV_ROOT_DIR = r'D:\codes\working\pos\Draft\csv\seg2'
DATASET = "ADE20K"

legend_labels = ['Baseline (with PE)', 'Position-Agnostic (No PE)', 'Ours (Guidance Only)', 'Ours (Full Method)']
metric_name = 'acc'
y_label = "Accuracy"
MODEL_TYPE = 'base'
# MODEL_TYPE = 'small'
# MODEL_TYPE = 'large'
CSV_FILE_NAMES = [
    f'{MODEL_TYPE}_pos.csv',
    f'{MODEL_TYPE}.csv',
    f'{MODEL_TYPE}_colrow.csv',
    f'{MODEL_TYPE}_colrow_o2.csv',
]

# =============================================================================
# Step 5: Execute the analysis
# =============================================================================
output_dir = rf"D:\codes\working\pos\Draft\plots\{'show_train' if show_train_acc else 'no_train'}"
# print("Reading data...")
all_data = read_csv_data(CSV_ROOT_DIR, CSV_FILE_NAMES)
# print("Plotting data...")
plot_title = f'Accuracy Comparison for {MODEL_TYPE.capitalize()} Model on {DATASET.capitalize()}'
# plot_data(all_data, show_train_acc=show_train_acc, max_epoch=max_epoch_to_show, title=plot_title)

plot_data(all_data, show_train_acc=show_train_acc, max_epoch=max_epoch_to_show, title=None, filname=plot_title, legend_labels=legend_labels, output_dir=output_dir, metric_name=metric_name, y_label=y_label)
#%%
# =============================================================================
# Step 2: Configuration
# =============================================================================
show_train_acc = False
max_epoch_to_show = 130 # Set to None to show all epochs
# Define the root directory for your CSV files
CSV_ROOT_DIR = r'D:\codes\working\pos\Draft\csv\cifar'
DATASET = "cifar"

CSV_ROOT_DIR = r'D:\codes\working\pos\Draft\csv\imagenet10'
# CSV_ROOT_DIR = r'D:\codes\working\pos\Draft\csv\imagenet10bs120'
# CSV_ROOT_DIR = r'D:\codes\working\pos\Draft\csv\imagenet10bs128'
# CSV_ROOT_DIR = r'D:\codes\working\pos\Draft\csv\imagenet10bs136s56'
CSV_ROOT_DIR = r'D:\codes\working\pos\Draft\csv\imagenet10bs136s60'
# CSV_ROOT_DIR = r'D:\codes\working\pos\Draft\csv\bak'
DATASET = "imagenet10"
# CSV_ROOT_DIR = r'D:\codes\working\pos\Draft\csv\imagenet100'
# DATASET = "imagenet100"

MODEL_TYPE = 'base'
# MODEL_TYPE = 'small'
MODEL_TYPE = 'large'
CSV_FILE_NAMES = [
    f'{MODEL_TYPE}{"_cifar" if DATASET == "cifar" else ""}_pos.csv',
    f'{MODEL_TYPE}{"_cifar" if DATASET == "cifar" else ""}.csv',
    f'{MODEL_TYPE}{"_cifar" if DATASET == "cifar" else ""}_colrow.csv',
    f'{MODEL_TYPE}{"_cifar" if DATASET == "cifar" else ""}_colrow_o2.csv',
    # f'{MODEL_TYPE}{"_cifar" if DATASET == "cifar" else ""}_o2.csv',
    # f'{MODEL_TYPE}{"_cifar" if DATASET == "cifar" else ""}_patch.csv',
    # f'{MODEL_TYPE}{"_cifar" if DATASET == "cifar" else ""}_patch_o2.csv',
    # f'rope_{MODEL_TYPE}.csv',
    # f'relpos_{MODEL_TYPE}.csv',
    # f'rpe_{MODEL_TYPE}.csv',
    # f'sin_{MODEL_TYPE}.csv',
    # f'alibi_{MODEL_TYPE}.csv',
]
# =============================================================================
# Step 5: Execute the analysis
# =============================================================================
output_dir = rf"D:\codes\working\pos\Draft\plots\{'show_train' if show_train_acc else 'no_train'}"
# print("Reading data...")
all_data = read_csv_data(CSV_ROOT_DIR, CSV_FILE_NAMES)
# print("Plotting data...")
plot_title = f'Accuracy Comparison for {MODEL_TYPE.capitalize()} Model on {DATASET.capitalize()}'
# plot_data(all_data, show_train_acc=show_train_acc, max_epoch=max_epoch_to_show, title=plot_title)

plot_data(all_data, show_train_acc=show_train_acc, max_epoch=max_epoch_to_show, title=plot_title, output_dir=output_dir)
#%%
show_train_acc = False
max_epoch_to_show = 130 # Set to None to show all epochs
min_epoch_to_show = 20 # Set to None to show all epochs

# Define the root directory for your CSV files
CSV_ROOT_DIR = r'D:\codes\working\pos\Draft\csv\imagenet10baseb392s55'
DATASET = "imagenet10"

MODEL_TYPE = 'base'
# MODEL_TYPE = 'small'
# MODEL_TYPE = 'large'
CSV_FILE_NAMES = [
    f'{MODEL_TYPE}_pos.csv',
    f'{MODEL_TYPE}.csv',
    # f'{MODEL_TYPE}_colrow.csv',
    # f'{MODEL_TYPE}_colrow2.csv',
    f'{MODEL_TYPE}_colrow3.csv',
    # f'{MODEL_TYPE}_colrow376.csv',
    # f'{MODEL_TYPE}_colrow390.csv',
    # f'{MODEL_TYPE}_colrow_o1.csv',
    f'{MODEL_TYPE}_colrow_o2.csv',
    # f'{MODEL_TYPE}_colrow_o3.csv',
    # f'{MODEL_TYPE}_colrow_o4.csv',
    # f'{MODEL_TYPE}_colrow_o5.csv',
    # f'{MODEL_TYPE}_colrow_o6.csv',
    # f'{MODEL_TYPE}_patch.csv',
    # f'{MODEL_TYPE}_patch_o2.csv',
    # f'{MODEL_TYPE}_o1.csv',
    # f'{MODEL_TYPE}_o2.csv',
    # f'{MODEL_TYPE}_o3.csv',
    # f'{MODEL_TYPE}_o4.csv',
    # # f'{MODEL_TYPE}_o5.csv',
    # f'{MODEL_TYPE}_o6.csv',
    f'rope_{MODEL_TYPE}.csv',
    f'relpos_{MODEL_TYPE}.csv',
    f'rpe_{MODEL_TYPE}.csv',
    f'sin_{MODEL_TYPE}.csv',
    f'alibi_{MODEL_TYPE}.csv',
]
# =============================================================================
# Step 5: Execute the analysis
# =============================================================================
output_dir = rf"D:\codes\working\pos\Draft\plots\pos_type"
# print("Reading data...")
all_data = read_csv_data(CSV_ROOT_DIR, CSV_FILE_NAMES)
# print("Plotting data...")
plot_title = f'Accuracy Comparison for {MODEL_TYPE.capitalize()} Model on {DATASET.capitalize()}'
# plot_data(all_data, show_train_acc=show_train_acc, max_epoch=max_epoch_to_show, title=plot_title)

plot_data(all_data, show_train_acc=show_train_acc, max_epoch=max_epoch_to_show, min_epoch=min_epoch_to_show, title=plot_title, output_dir=output_dir, figsize=(8, 8))

#%%
show_train_acc = False
max_epoch_to_show = 100 # Set to None to show all epochs
min_epoch_to_show = 50
metric_name = 'abs_rel'
# metric_name = 'a1'
# metric_name = 'rmse'
# Define the root directory for your CSV files
CSV_ROOT_DIR = r'D:\codes\working\pos\Draft\csv\depth'
DATASET = "hypersim"

MODEL_TYPE = 'base'
# MODEL_TYPE = 'small'
# MODEL_TYPE = 'large'
CSV_FILE_NAMES = [
    f'{MODEL_TYPE}_pos.csv',
    f'{MODEL_TYPE}.csv',
    f'{MODEL_TYPE}_colrow.csv',
    f'{MODEL_TYPE}_colrow_o2.csv',
    # f'{MODEL_TYPE}_colrow_o21.csv',
]
# =============================================================================
# Step 5: Execute the analysis
# =============================================================================
output_dir = rf"D:\codes\working\pos\Draft\plots\depth"
# print("Reading data...")
all_data = read_csv_data(CSV_ROOT_DIR, CSV_FILE_NAMES, metric_name=metric_name)
# print("Plotting data...")
plot_title = f'Accuracy Comparison for {MODEL_TYPE.capitalize()} Model on {DATASET.capitalize()}'
# plot_data(all_data, show_train_acc=show_train_acc, max_epoch=max_epoch_to_show, title=plot_title)

plot_data(all_data, show_train_acc=show_train_acc, max_epoch=max_epoch_to_show, min_epoch=min_epoch_to_show, title=plot_title, output_dir=output_dir, metric_name=metric_name, y_formatter='{y:.3}')
#%%