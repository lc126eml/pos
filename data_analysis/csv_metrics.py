
#%%
# =============================================================================
# Step 1: Import Libraries
# =============================================================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FuncFormatter
from csv_utils import read_csv_folder
# %%
csv_root = r"D:\codes\working\pos\Draft\csv\imagenet10baseb392s55"
metrics = read_csv_folder(csv_root, ['valid_acc'], 130)
print(metrics)
# %%
csv_root = r"D:\codes\working\pos\Draft\csv\depth"
metrics = read_csv_folder(csv_root, ['valid_abs_rel'], 100)
print(metrics)
# %%
csv_root = r"D:\codes\working\pos\Draft\csv\seg"
metrics = read_csv_folder(csv_root, ['valid_acc'], 130)
print(metrics)
# %%
csv_root = r"D:\codes\working\pos\Draft\csv\seg"
metrics = read_csv_folder(csv_root, ['valid_miou'], 130)
print(metrics)
# %%
csv_root = r"D:\codes\working\pos\Draft\csv\imagenet100"
metrics = read_csv_folder(csv_root, ['valid_acc'], 130)
print(metrics)
# %%
csv_root = r"D:\codes\working\pos\Draft\csv\cifar"
metrics = read_csv_folder(csv_root, ['valid_acc'], 130)
print(metrics)
# %%
