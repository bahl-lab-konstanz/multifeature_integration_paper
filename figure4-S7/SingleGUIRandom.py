#!/usr/bin/env python3
import numpy as np
import pandas as pd
import nrrd, h5py, os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
save_base = r'Y:\M11 2P mircroscopes\Sophie\ExpWithKatja\RandomCell_GUIFull_clean'

# ===================================================
# HELPERS
# ===================================================
def get_pixel_spacing(header):
    if "space directions" in header:
        return np.linalg.norm(header["space directions"], axis=0)
    elif "spacings" in header:
        return np.array(header["spacings"])
    raise ValueError("Pixel spacing info missing in NRRD header.")

_cache = {"path": None}
def load_volumes_if_needed(path):
    global _cache
    if _cache.get("path") == path:
        return _cache
    print(f"Loading volumes from: {path}")
    gad, _ = nrrd.read(fr"{path}\gadregistered2volume.nrrd", index_order="C")
    gadref, _ = nrrd.read(fr"{path}\gadGcampregistered2volume.nrrd", index_order="C")
    vglut, _ = nrrd.read(fr"{path}\vglutregistered2volume.nrrd", index_order="C")
    vglutref, _ = nrrd.read(fr"{path}\vglutGcampregistered2volume.nrrd", index_order="C")
    ref2, volume_header = nrrd.read(fr"{path}\volume.nrrd", index_order="C")
    ref1, ref1_header = nrrd.read(fr"{path}\ref1.nrrd", index_order="C")
    scalev = get_pixel_spacing(volume_header)[0]
    scalev1 = get_pixel_spacing(ref1_header)[0]
    _cache = {
        "path": path, "gad": gad, "gadref": gadref,
        "vglut": vglut, "vglutref": vglutref,
        "ref1": ref1, "ref2": ref2,
        "scalev": scalev, "scalev1": scalev1
    }
    return _cache

# ===================================================
# LOAD & FILTER MANUAL ENTRIES
# ===================================================
all_dfs = []
for output_path, file_base in zip(output_paths, file_bases):
    try:
        noi_df = pd.read_csv(fr"{output_path}\noi_df.csv")
        manual_df = pd.read_csv(fr"{output_path}\manual_entries.csv")

        # Clean and normalize manual labels
        manual_df["description"] = (
            manual_df["description"].astype(str).str.lower().str.strip()
            .replace({
                "gaba": "gad", "gaba weak": "gad",
                "vglut weak": "vglut", "bad alignment": "bad alignement"
            })
        )
        # keep only gad or vglut
        manual_df = manual_df[manual_df["description"].isin(["gad", "vglut"])]

        noi_df["cell_z_index"] = noi_df["cell_name"].astype(str) + "_" + noi_df["z_plane"].astype(str)
        manual_df["cell_z_index"] = manual_df["cell_name"].astype(str) + "_" + manual_df["zplane"].astype(str)
        merged = pd.merge(noi_df, manual_df, on="cell_z_index", how="inner")

        merged["output_path"] = output_path
        merged["file_base_name"] = file_base
        all_dfs.append(merged)
        print(f"✅ {file_base}: {len(merged)} gad/vglut cells loaded.")
    except Exception as e:
        print(f"⚠️ Skipping {file_base}: {e}")

all_cells = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal gad/vglut cells: {len(all_cells)}")

# ===================================================
# PICK ONE RANDOM CELL
# ===================================================
entry = all_cells.sample(1, random_state=np.random.randint(0, 9999)).iloc[0]
desc = entry["description"]
print(f"🎯 Random cell: {entry['file_base_name']} | cell {entry['cell_name_x']} | {desc}")

# ===================================================
# PLOT GUI-STYLE PANEL
# ===================================================
def plot_full_gui(entry):
    path = entry["output_path"]
    file_base = entry["file_base_name"]
    desc = entry["description"]
    cache = load_volumes_if_needed(path)
    gad, gadref, vglut, vglutref = cache["gad"], cache["gadref"], cache["vglut"], cache["vglutref"]
    ref1, ref2 = cache["ref1"], cache["ref2"]
    scalev, scalev1 = cache["scalev"], cache["scalev1"]

    cell_name = str(int(entry["cell_name_x"]))
    z_plane   = int(entry["z_plane"])
    z_plane_v = int(entry["z_plane_v"])
    tile      = str(int(np.floor(entry["tile_x"]))).zfill(3)
    cell_x_v, cell_y_v = int(entry["cell_x_v"]), int(entry["cell_y_v"])
    cell_x, cell_y     = int(entry["cell_x"]), int(entry["cell_y"])
    crop_size = 100

    x_min_v = int(max(cell_x_v / scalev - crop_size // 2, 0))
    x_max_v = int(min(cell_x_v / scalev + crop_size // 2, gad.shape[2]))
    y_min_v = int(max(cell_y_v / scalev - crop_size // 2, 0))
    y_max_v = int(min(cell_y_v / scalev + crop_size // 2, gad.shape[1]))
    x_min = int(max(cell_x - crop_size * scalev / (2 * scalev1), 0))
    x_max = int(min(cell_x + crop_size * scalev / (2 * scalev1), ref1.shape[2]))
    y_min = int(max(cell_y - crop_size * scalev / (2 * scalev1), 0))
    y_max = int(min(cell_y + crop_size * scalev / (2 * scalev1), ref1.shape[1]))

    z = str(z_plane).zfill(3)
    with h5py.File(fr"{path}\{file_base}_preprocessed_data.h5", "r") as f:
        contour = f[f"repeat00_tile{tile}_z{z}_950nm/preprocessed_data/fish00/cellpose_segmentation/unit_contours/{cell_name}"][:]
        contourv = np.floor(
            f[f"repeat00_tile{tile}_z{z}_950nm/preprocessed_data/fish00/cellpose_segmentation/unit_contours_ants_volume_registered/{cell_name}"][:] / scalev
        ).astype(int)

    adjusted_contour = contour - np.array([x_min, y_min])
    adjusted_contourv = contourv[:, :2] - np.array([x_min_v, y_min_v])

    gad_slice = gad[int(z_plane_v/2), y_min_v:y_max_v, x_min_v:x_max_v]
    vglut_slice = vglut[int(z_plane_v/2), y_min_v:y_max_v, x_min_v:x_max_v]
    gadref_slice = gadref[int(z_plane_v/2), y_min_v:y_max_v, x_min_v:x_max_v]
    vglutref_slice = vglutref[int(z_plane_v/2), y_min_v:y_max_v, x_min_v:x_max_v]
    ref1_slice = ref1[int(z_plane), y_min:y_max, x_min:x_max]
    ref2_slice = ref2[int(z_plane_v/2), y_min_v:y_max_v, x_min_v:x_max_v]

    # --- consistent magenta/green overlay ---
    gad_norm = (gad_slice - np.nanmin(gad_slice)) / (np.nanmax(gad_slice) - np.nanmin(gad_slice) + 1e-8)
    vglut_norm = (vglut_slice - np.nanmin(vglut_slice)) / (np.nanmax(vglut_slice) - np.nanmin(vglut_slice) + 1e-8)
    rgb = np.zeros((*gad_slice.shape, 3))
    rgb[..., 0] = gad_norm        # red
    rgb[..., 2] = gad_norm        # blue  → magenta
    rgb[..., 1] = vglut_norm      # green

    n_rows, n_cols = 3, 4
    width_cm = 7.5
    height_cm = (2.5* n_rows)

    fig, axes = plt.subplots(3, 4,  figsize = (width_cm/2.54, height_cm/2.54))
    axes = axes.flatten()

    # ---- 1st row ----
    axes[4].imshow(rgb, origin='lower')
    axes[4].plot(adjusted_contourv[:, 0], adjusted_contourv[:, 1],
                 color='w', lw=1, ls='--')  # dashed red
    axes[4].set_title("gad/vglut overlay")
    axes[4].axis("off")

    axes[0].imshow(gad_slice, cmap='gray', origin='lower')
    axes[0].set_title("gad")
    axes[0].axis("off")


    # --- Add scale bar (5 µm) to the first panel ---
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm

    scalebar_um = 5  # length in microns
    scalebar_px = scalebar_um / scalev  # convert to pixel units based on volume scale

    scalebar = AnchoredSizeBar(
        axes[0].transData,
        scalebar_px,
        None,  # ✅ no label text
        loc='lower left',
        pad=0.05,
        color='white',
        frameon=False,
        size_vertical=2,
        bbox_to_anchor=(0.06, 0.01),  # ✅ control horizontal & vertical independently
        bbox_transform=axes[0].transAxes
    )
    axes[0].add_artist(scalebar)

    axes[1].imshow(gad_slice, cmap='gray', origin='lower')
    axes[1].plot(adjusted_contourv[:, 0], adjusted_contourv[:, 1], 'r', lw=1)
    axes[1].set_title("gad")
    axes[1].axis("off")

    axes[2].imshow(vglut_slice, cmap='gray', origin='lower')
    axes[2].set_title("vglut")
    axes[2].axis("off")

    axes[3].imshow(vglut_slice, cmap='gray', origin='lower')
    axes[3].plot(adjusted_contourv[:, 0], adjusted_contourv[:, 1], 'r', lw=1)
    axes[3].set_title("vglut")
    axes[3].axis("off")

    # ---- 2nd & 3rd rows ----
    axes[5].imshow(gadref_slice, cmap='gray', origin='lower')
    axes[5].plot(adjusted_contourv[:, 0], adjusted_contourv[:, 1], 'r', lw=1)
    axes[5].set_title("gad vol ref")
    axes[5].axis("off")

    axes[6].imshow(vglutref_slice, cmap='gray', origin='lower')
    axes[6].set_title("vglut vol ref")
    axes[6].axis("off")

    axes[7].imshow(vglutref_slice, cmap='gray', origin='lower')
    axes[7].plot(adjusted_contourv[:, 0], adjusted_contourv[:, 1], 'r', lw=1)
    axes[7].set_title("vglut vol ref")
    axes[7].axis("off")

    axes[8].imshow(ref2_slice, cmap='gray', origin='lower')
    axes[8].set_title("ref2")
    axes[8].axis("off")

    axes[9].imshow(ref2_slice, cmap='gray', origin='lower')
    axes[9].plot(adjusted_contourv[:, 0], adjusted_contourv[:, 1], 'r', lw=1)
    axes[9].set_title("ref2")
    axes[9].axis("off")

    axes[10].imshow(ref1_slice, cmap='gray', origin='lower')
    axes[10].set_title("ref1")
    axes[10].axis("off")

    axes[11].imshow(ref1_slice, cmap='gray', origin='lower')
    axes[11].plot(adjusted_contour[:, 0], adjusted_contour[:, 1], 'r', lw=1)
    axes[11].set_title("ref1")
    axes[11].axis("off")

    folder_parts = os.path.normpath(path).split(os.sep)
    folder_name = folder_parts[-2] if folder_parts[-1].lower() == "left" else folder_parts[-1]
    # Determine fish label based on configuration order
    fish_idx = file_bases.index(file_base)
    fish_label = fish_labels[fish_idx]

    fig.suptitle(f"{fish_label} | cell {str(int(cell_name)-10000)} | {desc}", y=0.94)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ===================================================
# RUN
# ===================================================
fig = plot_full_gui(entry)
fig.savefig(f"{save_base}.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{save_base}.svg", dpi=300, bbox_inches="tight")
print(f"\n✅ Saved single random GUI-style figure:\n{save_base}.png\n{save_base}.svg")
plt.show()

