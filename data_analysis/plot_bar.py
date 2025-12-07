#%%
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Data Input ---
# This section contains the data from Table 5.
# You can replace these values with your own data.
methods = [
    'Ours',
    'DINOv2 PE',
    'Relative Position',
    # '1D Patch Index',
    '1D Patch Index',
    'RoPE',
    'ALiBi',
    'iRPE',
    'Sinusoidal',
]
accuracy = [
    67.04,
    54.54,
    60.45,
    59.55,
    60.44,
    58.95,
    55.65,
    52.63,
]

output_dir = rf"D:\codes\working\pos\Draft\plots\bar"
# --- Data Processing ---
# Combine the two lists and sort them by accuracy for a cleaner visualization.
data = sorted(zip(methods, accuracy), key=lambda item: item[1])
sorted_methods, sorted_accuracy = zip(*data)

# Create a color list to highlight your primary method.
colors = ['skyblue' if method != methods[0] else 'salmon' for method in sorted_methods]

# --- Plot Generation ---
# Set up the figure and axes for the plot.
fig, ax = plt.subplots(figsize=(6, 4))

# Create the horizontal bars.
bars = ax.barh(np.arange(len(sorted_methods)), sorted_accuracy, color=colors)

# Add the exact accuracy values as labels on each bar for clarity.
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.2f}%',
            va='center', ha='left', fontsize=10)

# --- Styling and Labels ---
# Set the labels for the y-axis, x-axis, and the main title.
ax.set_yticks(np.arange(len(sorted_methods)))
ax.set_yticklabels(sorted_methods, fontsize=11)
ax.set_xlabel('Classification Accuracy (%)', fontsize=12)
# ax.set_title('Comparison of Positional Information Methods on ImageNet-100', fontsize=14, pad=20)

# Adjust x-axis limits to ensure space for the data labels.
ax.set_xlim(0, max(sorted_accuracy) + 3)
ax.set_ylim(-0.5, len(sorted_methods) - 0.6)

# Remove unnecessary chart borders (spines) for a modern look.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add a subtle grid to make the values easier to read.
ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
ax.set_axisbelow(True)

# Ensure everything fits without overlapping.
plt.tight_layout()

# --- Save the Output ---
# Save the generated plot to a high-resolution PNG file.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, 'table_5_comparison_graph.png')
plt.savefig(output_path, dpi=300)

print("Graph has been generated and saved as 'table_5_comparison_graph.png'")
# %%
