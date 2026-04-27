import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
from pathlib import Path
import os
from IBM_functions_package import plotting_related_functions
from get_odor_recodting_data import load_trial, extract_odor_blocks

# Colormap function
def darken_colormap(cmap_name='turbo', scale=0.85):
    base = plt.colormaps.get_cmap(cmap_name)
    dark_colors = base(np.linspace(0, 1, 256))
    dark_colors[:, :3] *= scale
    return ListedColormap(dark_colors)

# Define datasets to process
datasets = ["forwardConditioned", "backwardConditioned", "Testgroup", "Controls"]

# Shared odor labels
odor_labels = ["MCH", "3-OCT", "1-OCT", "OIL"]

dark_turbo = darken_colormap()

for dataset in datasets:
    print(f"\n--- Processing dataset: {dataset} ---")

    fly_trials = load_trial(dataset)
    fly_data_dict = {}

    for fname, (pre_df, post_df) in fly_trials.items():
        fly_name = Path(fname).stem

        if dataset == "Testgroup" and (post_df is None or post_df.empty):
            print(f"⚠️ Skipping {fly_name}: Missing post condition sheet.")
            continue

        blocks_pre = extract_odor_blocks(pre_df, dataset, fname, sheet="pre")
        blocks_post = extract_odor_blocks(post_df, dataset, fname, sheet="post") if post_df is not None else {}

        fly_data_dict[fly_name] = {"pre": blocks_pre, "post": blocks_post}

    # Save directories
    base_path = f"/home/baxter/IBM/material/figures/bilz_visulaisation/{dataset}"
    png_path = os.path.join(base_path, "png")
    svg_path = os.path.join(base_path, "svg")
    os.makedirs(png_path, exist_ok=True)
    os.makedirs(svg_path, exist_ok=True)

    # Plot for each fly
    for fly_name, recordings in fly_data_dict.items():
        post_blocks = recordings.get("post", {})
        if dataset == "Testgroup" and (not post_blocks or any(
            not any(k.endswith(f"post_{odor.replace('-', '_')}") and not df.empty for k, df in post_blocks.items())
            for odor in odor_labels)):
            print(f"⚠️ Skipping {fly_name}: incomplete or missing POST data")
            continue

        # === Compute per-fly vmin/vmax ===
        fly_all_data = [
            df_block
            for phase in ["pre", "post"]
            for df_block in recordings[phase].values()
            if not df_block.empty
        ]
        if not fly_all_data:
            print(f"⚠️ Skipping {fly_name}: No data available for min/max calculation.")
            continue

        fly_min = min(df.min().min() for df in fly_all_data)
        fly_max = max(df.max().max() for df in fly_all_data)
        vmin = fly_min
        vmax = fly_max

        fig = plt.figure(figsize=(18, 10))
        fig.canvas.manager.set_window_title(f"{fly_name} - Pre & Post Odor Responses")
        gs = gridspec.GridSpec(nrows=2, ncols=5, width_ratios=[1, 1, 1, 1, 0.05], hspace=0.15, wspace=0.2)

        for i, odor in enumerate(odor_labels):
            for row_idx, phase in enumerate(["pre", "post"]):
                blocks = recordings[phase]
                try:
                    block_key = next(k for k in blocks if k.endswith(f"{phase}_{odor.replace('-', '_')}"))
                except StopIteration:
                    print(f"⚠️ {phase.upper()} block for {odor} not found in fly {fly_name}")
                    continue

                df_block = blocks[block_key]
                if df_block.empty:
                    print(f"⚠️ No data for {phase.upper()} {odor} in fly {fly_name}")
                    continue

                data_matrix = df_block.to_numpy()
                roi_labels = df_block.columns

                ax = plt.subplot(gs[row_idx, i])
                im = ax.imshow(data_matrix.T, aspect='auto', cmap=dark_turbo, vmin=vmin, vmax=vmax)

                ax.axvline(x=25, color='red', linestyle='--', linewidth=1)
                ax.axvline(x=35, color='red', linestyle='--', linewidth=1)

                # ROI label grouping
                group_positions = []
                group_labels = []
                prev_label = roi_labels[0]
                start_idx = 0
                for j, label in enumerate(roi_labels[1:], start=1):
                    if label != prev_label:
                        center = (start_idx + j - 1) / 2
                        group_positions.append(center)
                        group_labels.append(rf'$g_{{{prev_label[1:]}}}$')
                        start_idx = j
                        prev_label = label
                center = (start_idx + len(roi_labels) - 1) / 2
                group_positions.append(center)
                group_labels.append(rf'$g_{{{prev_label[1:]}}}$')

                for y in group_positions[:-1]:
                    ax.axhline(y + (group_positions[1] - group_positions[0]) / 2, color='white', linewidth=1)

                ax.set_yticks(group_positions)
                ax.set_yticklabels(group_labels if i == 0 else [])

                xticks = np.arange(0, data_matrix.shape[0], step=16)
                xtick_labels = (xticks * 0.25).astype(int)
                ax.set_xticks(xticks)
                if row_idx == 1:
                    ax.set_xticklabels(xtick_labels)
                else:
                    ax.set_xticklabels([])

                ax.set_title(f'{odor}-Pre' if row_idx == 0 else f'{odor}-Post')

        # Shared colorbar
        cax = plt.subplot(gs[:, 4])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r'$\Delta F/F_0$ [%]')  # First line
        cbar.ax.text(0.5, -0.1, f'Min={vmin:.1f}, Max={vmax:.1f}',
             transform=cbar.ax.transAxes,
             ha='center', va='top',
             fontsize='small')

        # Global time axis label
        fig.text(0.467, 0.02, 't [s]', ha='center')

        plt.subplots_adjust(
            top=0.945,
            bottom=0.11,
            left=0.05,
            right=0.900,
            hspace=0.2,
            wspace=0.2
        )

        fig.savefig(os.path.join(png_path, f"{fly_name}.png"), dpi=300)
        fig.savefig(os.path.join(svg_path, f"{fly_name}.svg"))

        plt.close(fig)
