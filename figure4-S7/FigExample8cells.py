#!/usr/bin/env python3
import numpy as np
import pandas as pd
import nrrd, h5py, os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

_cache = {"path": None}

from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.weight'] = 'normal'
rcParams['font.size'] = 6




# ===================================================
# USER CONFIGURATION
# ===================================================
output_paths = [
    r'X:\Bahl lab member directories\Katja\paper_data\figure_4\20250331',
    r'X:\Bahl lab member directories\Katja\paper_data\figure_4\20241112',
    r'X:\Bahl lab member directories\Katja\paper_data\figure_4\20250120',
    r'X:\Bahl lab member directories\Katja\paper_data\figure_4\20250217',
    r'X:\Bahl lab member directories\Katja\paper_data\figure_4\20250408'
]

file_bases = [
    '2025-03-31_11-46-00_fish000_setup0_arena0_functional',
    '2024-11-12_11-40-51_fish000_SA',
    '2025-01-20_10-47-06_fish000_setup0_arena0_SA',
    '2025-02-17_18-40-48_fish001_setup1_arena0_functional',
    '2025-04-08_10-20-26_fish000_setup0_arena0_functional'
]

fish_labels = [f"fish {i}" for i in range(1, len(output_paths) + 1)]
save_base = r'Y:\M11 2P mircroscopes\Sophie\ExpWithKatja\Random8cells_gadlabels_dashed'

# ===================================================
# HELPERS
# ===================================================
def get_pixel_spacing(header):
    if "space directions" in header:
        return np.linalg.norm(header["space directions"], axis=0)
    elif "spacings" in header:
        return np.array(header["spacings"])
    raise ValueError("Pixel spacing info missing in NRRD header.")


def load_volumes_if_needed(path):
    global _cache
    if _cache.get("path") == path:
        return _cache
    print(f"Loading volumes from: {path}")
    gad, _ = nrrd.read(fr"{path}\gadregistered2volume.nrrd", index_order="C")
    vglut, _ = nrrd.read(fr"{path}\vglutregistered2volume.nrrd", index_order="C")
    ref1, ref1_header = nrrd.read(fr"{path}\ref1.nrrd", index_order="C")
    ref2, volume_header = nrrd.read(fr"{path}\volume.nrrd", index_order="C")
    scalev = get_pixel_spacing(volume_header)[0]
    scalev1 = get_pixel_spacing(ref1_header)[0]
    _cache = {"path": path, "gad": gad, "vglut": vglut,
              "ref1": ref1, "ref2": ref2, "scalev": scalev, "scalev1": scalev1}
    return _cache

# ===================================================
# LOAD AND CLEAN SELECTED CELLS
# ===================================================
all_dfs = []
for idx, (output_path, file_base, fish_label) in enumerate(zip(output_paths, file_bases, fish_labels), start=1):
    try:
        noi_df = pd.read_csv(fr"{output_path}\noi_df.csv")
        manual_df = pd.read_csv(fr"{output_path}\manual_entries.csv")

        # --- Normalize and clean descriptions ---
        manual_df["description"] = (
            manual_df["description"]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({
                "gaba weak": "gad",
                "gaba": "gad",
                "vglut weak": "vglut",
                "bad alignment": "bad alignement"
            })
        )

        # --- Filter out unwanted entries ---
        manual_df = manual_df[
            ~manual_df["description"].isin(["bad alignement", "unclear"])
        ]

        noi_df["cell_z_index"] = noi_df["cell_name"].astype(str) + "_" + noi_df["z_plane"].astype(str)
        manual_df["cell_z_index"] = manual_df["cell_name"].astype(str) + "_" + manual_df["zplane"].astype(str)

        merged = pd.merge(noi_df, manual_df, on="cell_z_index", how="inner")

        merged["output_path"] = output_path
        merged["file_base_name"] = file_base
        merged["fish_label"] = fish_label
        all_dfs.append(merged)
        print(f"✅ {fish_label} ({file_base}): {len(merged)} clean annotated cells.")
    except Exception as e:
        print(f"⚠️ Skipping {file_base}: {e}")

all_cells = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal clean annotated cells across all fish: {len(all_cells)}")

# ===================================================
# RANDOM SAMPLE (8 CELLS)
# ===================================================
sampled = all_cells.sample(8, random_state=np.random.randint(0, 9999)).reset_index(drop=True)

# ===================================================
# FIGURE GENERATION
# ===================================================
n_rows, n_cols = 4, 2
width_cm  = 14.5
height_cm = 2.5 * n_rows

fig, axes = plt.subplots(n_rows, n_cols, figsize = (width_cm/2.54, height_cm/2.54))
axes = np.array(axes).reshape(n_rows, n_cols)
# ---- Panel label “A” in top-left of the whole figure ----

k=0
for ax, (_, entry) in zip(axes.flat, sampled.iterrows()):
    path = entry["output_path"]
    file_base = entry["file_base_name"]
    fish_label = entry["fish_label"]
    description = str(entry.get("description", "unknown"))
    cache = load_volumes_if_needed(path)
    gad, vglut, ref1, ref2 = cache["gad"], cache["vglut"], cache["ref1"], cache["ref2"]
    scalev, scalev1 = cache["scalev"], cache["scalev1"]

    cell_name = str(int(entry["cell_name_x"]))
    z_plane   = int(entry["z_plane"])
    z_plane_v = int(entry["z_plane_v"])
    tile      = str(int(np.floor(entry["tile_x"]))).zfill(3)
    cell_x_v, cell_y_v = int(entry["cell_x_v"]), int(entry["cell_y_v"])
    cell_x, cell_y     = int(entry["cell_x"]), int(entry["cell_y"])

    crop_size = 80
    x_min_v = int(max(cell_x_v / scalev - crop_size / 2, 0))
    x_max_v = int(min(cell_x_v / scalev + crop_size / 2, gad.shape[2]))
    y_min_v = int(max(cell_y_v / scalev - crop_size / 2, 0))
    y_max_v = int(min(cell_y_v / scalev + crop_size / 2, gad.shape[1]))
    x_min = int(max(cell_x - crop_size * scalev / (2 * scalev1), 0))
    x_max = int(min(cell_x + crop_size * scalev / (2 * scalev1), ref1.shape[2]))
    y_min = int(max(cell_y - crop_size * scalev / (2 * scalev1), 0))
    y_max = int(min(cell_y + crop_size * scalev / (2 * scalev1), ref1.shape[1]))

    z = str(z_plane).zfill(3)
    try:
        with h5py.File(fr"{path}\{file_base}_preprocessed_data.h5", "r") as f:
            contour = f[f"repeat00_tile{tile}_z{z}_950nm/preprocessed_data/fish00/cellpose_segmentation/unit_contours/{cell_name}"][:]
            contourv = np.floor(
                f[f"repeat00_tile{tile}_z{z}_950nm/preprocessed_data/fish00/cellpose_segmentation/unit_contours_ants_volume_registered/{cell_name}"][:] / scalev
            ).astype(int)
    except Exception as e:
        print(f"⚠️ Skipping {file_base} cell {cell_name}: {e}")
        ax.axis("off")
        continue

    adjusted_contour = contour - np.array([x_min, y_min])
    adjusted_contourv = contourv[:, :2] - np.array([x_min_v, y_min_v])

    gad_slice = gad[int(z_plane_v / 2), y_min_v:y_max_v, x_min_v:x_max_v]
    vglut_slice = vglut[int(z_plane_v / 2), y_min_v:y_max_v, x_min_v:x_max_v]
    ref1_slice = ref1[int(z_plane), y_min:y_max, x_min:x_max]
    ref2_slice = ref2[int(z_plane_v / 2), y_min_v:y_max_v, x_min_v:x_max_v]

    gad_norm = (gad_slice - np.nanmin(gad_slice)) / (np.nanmax(gad_slice) - np.nanmin(gad_slice) + 1e-8)
    vglut_norm = (vglut_slice - np.nanmin(vglut_slice)) / (np.nanmax(vglut_slice) - np.nanmin(vglut_slice) + 1e-8)
    rgb = np.zeros((*gad_slice.shape, 3))
    rgb[..., 0] = gad_norm
    rgb[..., 2] = gad_norm     # magenta (red + blue)
    rgb[..., 1] = vglut_norm   # green

    # --- 3 subpanels per grid cell ---
    ax.axis("off")
    titles = ["gad/vglut", "fixed", "functional"]
    imgs = [rgb, ref2_slice, ref1_slice]
    contours = [adjusted_contourv, adjusted_contourv, adjusted_contour]

    for i in range(3):
        subax = inset_axes(ax, width="30%", height="92%", loc="center",
                           bbox_to_anchor=(i*0.33-0.33, 0, 1, 1),
                           bbox_transform=ax.transAxes, borderpad=0)
        subax.imshow(imgs[i], cmap="gray", origin="lower")

        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

        # Scale bar only in left-most subpanel
        if i == 0 and k==0:
            scalebar_um = 5
            scalebar_px = scalebar_um / scalev
            k=k+1
            scalebar = AnchoredSizeBar(
                subax.transData,
                scalebar_px,
                None,  # ✅ no label text
                loc='lower left',
                pad=0.1,
                color='white',
                frameon=False,
                size_vertical=2,
                bbox_to_anchor=(0.06, 0.01),  # ✅ control horizontal & vertical independently
                bbox_transform=subax.transAxes
            )
            subax.add_artist(scalebar)
            # --- Add magenta/green legend ---

            # Coordinates are relative to axis: (x, y) in [0,1]


        # Dashed contour for first panel only
        if i == 0:
            subax.plot(contours[i][:, 0], contours[i][:, 1],
                       color="w", lw=0.8, ls="--")  # dashed red
        else:
            subax.plot(contours[i][:, 0], contours[i][:, 1],
                       color="r", lw=0.8, ls="-")   # solid red

        subax.axis("off")
        # --- Titles only for first row ---
        if ax in axes[0, :]:  # ✅ only top row gets titles
            if i == 0:
                # Colored gad/vglut title
                subax.text(0.5, 1.30, "gad ",
                           color='magenta', fontweight='bold',
                           ha='right', va='bottom',
                           transform=subax.transAxes, fontfamily='Arial', fontsize=6)
                subax.text(0.5, 1.30, " / ",
                           color='black',
                           ha='center', va='bottom',
                           transform=subax.transAxes)
                subax.text(0.5, 1.30, " vglut",
                           color='lime', fontweight='bold',
                           ha='left', va='bottom',
                           transform=subax.transAxes)
            else:
                subax.set_title(titles[i],pad=14)

    ax.set_title(f"{fish_label} | cell {str(int(cell_name)-10000)} | {description}",pad=-2)

for ax in axes.flat[len(sampled):]:
    ax.axis("off")


# plt.tight_layout(h_pad=0.4, w_pad=0.5)

# Preserve text as text (not paths)
rcParams['svg.fonttype'] = 'none'
fig.savefig(f"{save_base}.svg", dpi=300, bbox_inches="tight")
fig.savefig(f"{save_base}.png", dpi=300, bbox_inches="tight")
print(f"\n✅ Saved random 8-cell figure (dashed red contour in first panel):\n{save_base}.svg\n{save_base}.png")
plt.show()
plt.close(fig)
