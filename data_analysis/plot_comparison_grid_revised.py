#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

# =============================================================================
# 1) CSV reading
# =============================================================================
def read_csv_data(
    csv_root: str,
    csv_files_with_labels: list[tuple[str, str]],
) -> list[tuple[str, str, pd.DataFrame]]:
    """
    Reads data from multiple CSV files and returns a list of (label, base_name, DataFrame).
    Required columns: epoch, train_acc, valid_acc
    """
    data_to_plot = []
    for file_name, label in csv_files_with_labels:
        file_path = os.path.join(csv_root, file_name)
        if not os.path.exists(file_path):
            print(f"[WARN] File not found: {file_path}. Skipping.")
            continue

        df = pd.read_csv(file_path)
        required_cols = {"epoch", "train_acc", "valid_acc"}
        if not required_cols.issubset(df.columns):
            print(f"[WARN] Skipping {file_name}: missing one of {required_cols}.")
            continue

        base_name = os.path.splitext(file_name)[0]
        data_to_plot.append((label, base_name, df))

    return data_to_plot


# =============================================================================
# 2) Plot helpers
# =============================================================================
def percent_formatter(v, _):
    return f"{v*100:.0f}%"


def plot_panel(
    ax,
    data_to_plot: list[tuple[str, str, pd.DataFrame]],
    *,
    lw_map: dict[str, float],
    max_epoch: int = 130,
):
    """
    Plot validation accuracy curves only (no endpoint labels).
    """

    desired_order = ["No PE", "AbsPE", "RoPE", "Guidance"]
    order_index = {name: i for i, name in enumerate(desired_order)}

    series = []
    for label, base_name, df in data_to_plot:
        series.append((order_index.get(label, 999), label, base_name, df))
    series.sort(key=lambda t: t[0])

    for _, label, base_name, df in series:
        plot_df = df[df["epoch"] <= max_epoch].copy()
        if plot_df.empty:
            continue

        x = plot_df["epoch"].to_numpy()
        y = plot_df["valid_acc"].to_numpy()

        lw = lw_map.get(label, lw_map.get(base_name, 1.8))
        ax.plot(x, y, label=label, linewidth=lw)

    # Styling
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_xlim(0, max_epoch)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))


def set_ylim_from_data(axes, data_lists, max_epoch: int, pad: float = 0.02):
    vals = []
    for data_to_plot in data_lists:
        for _, _, df in data_to_plot:
            d = df[df["epoch"] <= max_epoch]
            if not d.empty:
                vals.append(d["valid_acc"].min())
                vals.append(d["valid_acc"].max())

    if not vals:
        return

    lo = max(0.0, min(vals) - 0.00)
    hi = min(1.0, max(vals) + pad)
    for ax in axes:
        ax.set_ylim(lo, hi)


# =============================================================================
# 3) Figure 1 (two panels)
# =============================================================================
def plot_fig1_two_panel(
    csv_root_dir: str,
    *,
    small_files_with_labels: list[tuple[str, str]],
    base_files_with_labels: list[tuple[str, str]],
    max_epoch: int = 130,
    output_path: str | None = None,
):
    # Make Ours thicker
    lw_map = {
        # "Guidance": 2.0,
    }

    small_data = read_csv_data(csv_root_dir, small_files_with_labels)
    base_data = read_csv_data(csv_root_dir, base_files_with_labels)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(11.0, 3.6),
        # figsize=(8.0, 3.),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    axes[0].set_title("ViT-S (DINOv3)", fontsize=13, pad=6)
    axes[1].set_title("ViT-B (DINOv3)", fontsize=13, pad=6)

    plot_panel(axes[0], small_data, lw_map=lw_map, max_epoch=max_epoch)
    plot_panel(axes[1], base_data, lw_map=lw_map, max_epoch=max_epoch)

    set_ylim_from_data(axes, [small_data, base_data], max_epoch=max_epoch, pad=0.02)

    fig.supxlabel("Epoch", fontsize=12)
    fig.supylabel("Top-1 Acc. (%)", fontsize=12)

    # --- Legend inside right panel, lower-right blank area ---
    handles, labels = axes[1].get_legend_handles_labels()
    # Deduplicate and enforce order
    uniq = {}
    for h, l in zip(handles, labels):
        uniq[l] = h
    ordered = ["No PE", "AbsPE", "RoPE", "Guidance"]
    handles = [uniq[l] for l in ordered if l in uniq]
    labels  = [l for l in ordered if l in uniq]

    axes[1].legend(
        handles, labels,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.02),  # slight inset from corner
        frameon=False,
        fontsize=10,
        ncol=1
    )

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved: {output_path}")
    else:
        plt.show()


# =============================================================================
# 4) Run
# =============================================================================
if __name__ == "__main__":
    CSV_ROOT_DIR = r"D:\codes\working\pos\Draft\csv\redo20251215\dinov3b392s55of0dynamic_224_"
    OUT_PATH = r"D:\gdrive\pos encoding\TNNLS\fig\fig1_imagenet100_dinov3.png"

    SMALL_FILES_WITH_LABELS = [
        ("small_overlap_0_rc_False_alpha_600lr100.csv", "No PE"),
        ("small_abs_pos2_rc_False_lr100.csv", "AbsPE"),
        ("small_rot_pos_rc_False_lr100.csv", "RoPE"),
        ("small_overlap_0_rc_True_alpha_300lr100.csv", "Guidance"),
    ]

    BASE_FILES_WITH_LABELS = [
        ("base_overlap_0_rc_False_alpha_600.0.csv", "No PE"),
        ("base_abs_pos_overlap_0_rc_False_alpha_250lr50.csv", "AbsPE"),
        ("base_rot_pos_rc_False_lr50.csv", "RoPE"),
        ("base_overlap_0_rc_True_alpha_600lr50.csv", "Guidance"),
    ]

    plot_fig1_two_panel(
        csv_root_dir=CSV_ROOT_DIR,
        small_files_with_labels=SMALL_FILES_WITH_LABELS,
        base_files_with_labels=BASE_FILES_WITH_LABELS,
        max_epoch=130,
        output_path=OUT_PATH,  # set None to show interactively
    )

# %%
