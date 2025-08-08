import h5py
import numpy as np
import navis
import os
from pathlib import Path
import matplotlib.pyplot as plt
from multifeature_integration_paper.figure_helper import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
import nrrd
from scipy.ndimage import binary_erosion
import imageio
from PIL import Image
from multifeature_integration_paper.useful_small_funcs import create_combined_region_npy_mask


def sub_plot_method_outline_PA(path_to_example_data_func, path_to_example_data_HD, path_to_example_data_volume, path_to_example_swc,
                               subfig_zoomout, subfig_zoominpre, subfig_zoominpost, subfig_traces, subfigs_tracings,
                               subfig_neuron_xy, subfig_neuron_yz, subfig_brain_xy, subfig_brain_yz, cell_id, xs, ys, zs, masks_path):
    preproc_hdf5 = h5py.File(path_to_example_data_func, "r")

    all_dynamics = np.array(preproc_hdf5[f'repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/dLpN_dRpR_dNpL/F'])

    avg_im = np.array(preproc_hdf5['repeat00_tile000_z000_950nm']['preprocessed_data']['fish00']['imaging_data_channel0_time_averaged']).reshape(799, 799)
    subfig_zoomout.draw_image(np.clip(avg_im, np.nanpercentile(avg_im, 5), np.nanpercentile(avg_im, 98)), colormap='gray',
                       extent=(0, 800, 800, 0), image_origin='upper')
    subfig_zoominpre.draw_image(np.clip(avg_im, np.nanpercentile(avg_im, 5), np.nanpercentile(avg_im, 98)), colormap='gray',
                                extent=(0, 800, 800, 0), image_origin='upper')

    unit_contour = np.array(preproc_hdf5['repeat00_tile000_z000_950nm']['preprocessed_data']['fish00']['cellpose_segmentation']['unit_contours'][f'{10000 + cell_id}'])
    subfig_zoomout.draw_line(unit_contour[:, 0], unit_contour[:, 1], lc='#D55E00', lw=0.4)
    subfig_zoomout.draw_line([539-220/2, 539-220/2, 539+220/2, 539+220/2, 539-220/2, ], [355+220/2, 355-220/2, 355-220/2, 355+220/2, 355+220/2,], lc='w', lw=0.5)

    subfig_zoominpre.draw_scatter([539+8.8, ], [355-8.8, ], pt=MarkerStyle('^', 'left', Affine2D().rotate_deg(135)), pc='#D55E00', ec=None, ps=4)

    preproc_hdf5.close()

    preproc_hdf5 = h5py.File(path_to_example_data_HD, "r")
    avg_im = np.array(preproc_hdf5['average_stack_repeat00_tile000_950nm_channel0'])[23, :, :].reshape(799, 799)

    subfig_zoominpost.draw_image(np.clip(avg_im, np.nanpercentile(avg_im, 5), np.nanpercentile(avg_im, 95)),
                                colormap='gray',
                                extent=(0, 800, 800, 0), image_origin='upper')

    subfig_zoominpost.draw_scatter([437+21.92, ], [274-21.92, ], pt=MarkerStyle('^', 'left', Affine2D().rotate_deg(135)), pc='#D55E00', ec=None, ps=4)

    preproc_hdf5.close()

    cell_dynamics = all_dynamics[:, cell_id, :]
    df_f0 = (cell_dynamics - np.nanmean(cell_dynamics[:, :10], axis=1)[:, None]) / np.nanmean(cell_dynamics[:, :10], axis=1)[:, None]

    for trial in range(df_f0.shape[0]):
        subfig_traces.draw_line(np.arange(0, 120), df_f0[trial, :120], lc='#EEAE7C', lw=0.5)
        subfig_traces.draw_line(np.arange(130, 250), df_f0[trial, 120:240], lc='#EEAE7C', lw=0.5)
        subfig_traces.draw_line(np.arange(260, 380), df_f0[trial, 240:], lc='#EEAE7C', lw=0.5)

    subfig_traces.draw_line(np.arange(0, 120), np.nanmean(df_f0, axis=0)[:120], lc='#D55E00', lw=1)
    subfig_traces.draw_line(np.arange(130, 250), np.nanmean(df_f0, axis=0)[120:240], lc='#D55E00', lw=1)
    subfig_traces.draw_line(np.arange(260, 380), np.nanmean(df_f0, axis=0)[240:], lc='#D55E00', lw=1)
    subfig_traces.draw_line([-2, -2], [0, 1], lc='k')
    subfig_traces.draw_text(-40, 0.5, '1 dF/F\u2080', textlabel_rotation=90, textlabel_ha='left')
    subfig_traces.draw_line([355, 375], [-0.95, -0.95], lc='k')
    subfig_traces.draw_text(360, -3, '10s', textlabel_ha='center')

    preproc_hdf5 = h5py.File(path_to_example_data_volume, "r")
    avg_im = np.array(preproc_hdf5['average_stack_repeat00_tile000_950nm_channel0'])
    preproc_hdf5.close()

    for subfig, z, x, y, contrast in zip(subfigs_tracings, zs, xs, ys, [99, 97, 90, 85]):
        subfig.draw_image(np.clip(avg_im[z, :, :], np.nanpercentile(avg_im, 5), np.nanpercentile(avg_im, contrast)),
                                     colormap='gray',
                                     extent=(0, 800, 800, 0), image_origin='upper')
        subfig.draw_text(x+40, y, f'z={2*(z-zs[0]):.0f}\u00b5m')

    total_brain_regions = [navis.read_mesh(fr'{masks_path}\prosencephalon_(forebrain).obj', units='microns', output='volume'),
                 navis.read_mesh(fr'{masks_path}\mesencephalon_(midbrain).obj', units='microns', output='volume'),
                 navis.read_mesh(fr'{masks_path}\rhombencephalon_(hindbrain).obj', units='microns', output='volume')]
    brain_regions = [navis.read_mesh(fr'{masks_path}\superior_ventral_medulla_oblongata_(entire).obj', units='microns', output='volume'),
                     navis.read_mesh(fr'{masks_path}\mesencephalon_(midbrain).obj', units='microns', output='volume'),
                     navis.read_mesh(fr'{masks_path}\cerebellum.obj', units='microns', output='volume')]

    neuron = navis.read_swc(path_to_example_swc)
    neuron.soma = 1
    neuron.nodes.iloc[0, 5] = 2
    neuron.soma_radius = 4
    cell_swc = navis.smooth_skeleton(neuron)
    subfig_neuron_xy.draw_navis_neuron(cell_swc, brain_regions, navis_view=('x', '-y'), lc='#D55E00', lw=0.5, rasterized=True)
    subfig_neuron_xy.draw_line([311 * 0.798, 311 * 0.798], [350 * 0.798, 750 * 0.798], lc='w')
    subfig_neuron_xy.draw_line([10, 110], [600, 600], lc='k')
    subfig_neuron_xy.draw_line([10, 10], [600, 500], lc='k')
    subfig_neuron_xy.draw_text(130, 600, 'm')
    subfig_neuron_xy.draw_text(10, 480, 'a')

    subfig_neuron_yz.draw_navis_neuron(cell_swc, brain_regions, navis_view=('z', '-y'), lc='#D55E00', lw=0.5, rasterized=True)
    subfig_neuron_yz.draw_line([180, 280], [600, 600], lc='k')
    subfig_neuron_yz.draw_line([280, 280], [600, 500], lc='k')
    subfig_neuron_yz.draw_text(160, 600, 'v')
    subfig_neuron_yz.draw_text(280, 480, 'a')

    subfig_brain_xy.draw_navis_neuron(None, total_brain_regions, navis_view=('x', '-y'), lw=0.5, rasterized=True)
    subfig_brain_xy.draw_navis_neuron(None, brain_regions, navis_color='gray', navis_view=('x', '-y'), lw=0.5, rasterized=True)
    subfig_brain_yz.draw_navis_neuron(None, total_brain_regions, navis_view=('z', '-y'), lw=0.5, rasterized=True)
    subfig_brain_yz.draw_navis_neuron(None, brain_regions, navis_color='gray', navis_view=('z', '-y'), lw=0.5, rasterized=True)

    return

def subfig_PA_functional_loc(subfig, subfig_xy, subfig_yz, PA_data_path, file_base_names, volume_file_base_names, cell_folders, cell_mask_IDs, swc_ids, masks_path, z_planes=None, plot_color='k', plot_color_2='g', timescalebar=False):
    preproc_paths = []
    for file_base_name, cell_folder in zip(file_base_names, cell_folders):
        preproc_paths.append(fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}_preprocessed_data.h5')

    first_fish = True
    if z_planes is None:
        z_planes = np.zeros(len(preproc_paths))
    for preproc_path, cell_mask_ID, z_plane in zip(preproc_paths, cell_mask_IDs, z_planes):

        print(preproc_path, cell_mask_ID, z_plane)
        preproc_hdf5 = h5py.File(preproc_path, 'r')
        if '2025' in preproc_path or '20240115-1' in preproc_path or '20240524-1' in preproc_path:
            all_dynamics = np.array(preproc_hdf5[f'repeat00_tile000_z{z_plane:03d}_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/dLpN_dRpR_dNpL/F'])
        elif '2023' in preproc_path:
            all_dynamics = np.concatenate(
                [np.array(preproc_hdf5[f'z_plane{z_plane:04d}/cellpose_segmentation/stimulus_aligned_dynamics/stimulus0000/F']),
                 np.array(
                     preproc_hdf5[f'z_plane{z_plane:04d}/cellpose_segmentation/stimulus_aligned_dynamics/stimulus0000/F'][:2, :,
                     :]),
                 np.array(
                     preproc_hdf5[f'z_plane{z_plane:04d}/cellpose_segmentation/stimulus_aligned_dynamics/stimulus0000/F'][:2, :,
                     :])], axis=2)
        else:
            all_dynamics = np.array(preproc_hdf5[f'repeat00_tile000_z{z_plane:03d}_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/stimulus0000/F'])
        preproc_hdf5.close()
        print(all_dynamics.shape)
        cell_dynamics = all_dynamics[:, cell_mask_ID, :]

        if first_fish:
            df_f0 = (cell_dynamics - np.nanmean(cell_dynamics[:, :10], axis=1)[:, None]) /  np.nanmean(cell_dynamics[:, :10], axis=1)[:, None]
            df_f0_avg = np.nanmean((cell_dynamics - np.nanmean(cell_dynamics[:, :10], axis=1)[:, None]) /  np.nanmean(cell_dynamics[:, :10], axis=1)[:, None], axis=0)
            first_fish = False
        else:
            df_f0 = np.vstack([df_f0, (cell_dynamics - np.nanmean(cell_dynamics[:, :10], axis=1)[:, None]) /  np.nanmean(cell_dynamics[:, :10], axis=1)[:, None]])
            df_f0_avg = np.vstack([df_f0_avg, np.nanmean((cell_dynamics - np.nanmean(cell_dynamics[:, :10], axis=1)[:, None]) /  np.nanmean(cell_dynamics[:, :10], axis=1)[:, None], axis=0)])

    for trial in range(df_f0_avg.shape[0]):
        subfig.draw_line(np.arange(0, 120), df_f0_avg[trial, :120], lc=plot_color_2, lw=0.5)
        subfig.draw_line(np.arange(130, 250), df_f0_avg[trial, 120:240], lc=plot_color_2, lw=0.5)
        subfig.draw_line(np.arange(260, 380), df_f0_avg[trial, 240:], lc=plot_color_2, lw=0.5)

    subfig.draw_line(np.arange(0, 120), np.nanmean(df_f0_avg[:, :120], axis=0), lc=plot_color, lw=1)
    subfig.draw_line(np.arange(130, 250), np.nanmean(df_f0_avg[:, 120:240], axis=0), lc=plot_color, lw=1)
    subfig.draw_line(np.arange(260, 380), np.nanmean(df_f0_avg[:, 240:], axis=0), lc=plot_color, lw=1)

    subfig.draw_line([-2, -2], [0, 1], lc='k')
    subfig.draw_text(-40, 0.5, '1 dF/F\u2080', textlabel_rotation=90, textlabel_ha='left')
    if timescalebar:
        subfig.draw_line([355, 375], [-0.8, -0.8], lc='k')
        subfig.draw_text(360, -2.2, '10s', textlabel_ha='center')

    brain_regions = [navis.read_mesh(fr'{masks_path}\superior_ventral_medulla_oblongata_(entire).obj', units='microns', output='volume'),
                     navis.read_mesh(fr'{masks_path}\mesencephalon_(midbrain).obj', units='microns', output='volume'),
                     navis.read_mesh(fr'{masks_path}\cerebellum.obj', units='microns', output='volume')]

    swc_cells = []
    for cell_folder, file_base_name, swc_id in zip(cell_folders, volume_file_base_names, swc_ids):
        if os.path.exists(fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'):
            swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'
        elif os.path.exists(fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'):
            swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'
        else:
            print(fr'WARNING: {PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc does not exist. Skipping this neuron. ')
            continue

        neuron = navis.read_swc(swc_file)
        neuron.soma = 1
        neuron.nodes.iloc[0, 5] = 2
        neuron.soma_radius = 4
        swc_cells.append(navis.smooth_skeleton(neuron))

    subfig_xy.draw_navis_neuron(swc_cells, brain_regions, navis_view=('x', '-y'), lc=plot_color, lw=0.25, rasterized=True)
    subfig_xy.draw_line([311 * 0.798, 311 * 0.798], [350 * 0.798, 750 * 0.798], lc='w')

    subfig_yz.draw_navis_neuron(swc_cells, brain_regions, navis_view=('z', '-y'), lc=plot_color, lw=0.25, rasterized=True)

    return

def subfig_all_PA_neurons_loc(subfig_xy, subfig_yz, subfigs_region_counts, PA_data_path, all_volume_file_base_names, all_cell_folders, all_swc_ids, all_plot_colors, detailed_brain_regions, masks_path, regions_path, video_path=None):
    brain_regions = [navis.read_mesh(fr'{masks_path}\superior_ventral_medulla_oblongata_(entire).obj', units='microns', output='volume'),
                     navis.read_mesh(fr'{masks_path}\mesencephalon_(midbrain).obj', units='microns', output='volume'),
                     navis.read_mesh(fr'{masks_path}\cerebellum.obj', units='microns', output='volume')]

    nodes_per_region = [[]] * 6
    for type in range(6):
        nodes_per_region[type] = np.zeros((len(detailed_brain_regions), len(all_volume_file_base_names[type])))

    region_masks = create_combined_region_npy_mask(regions_path, regions=detailed_brain_regions)
    tectum_mask = create_combined_region_npy_mask(regions_path, regions=['tectum'])

    first_set = True
    all_swc_cells = []
    all_colors = []
    for type, (volume_file_base_names, cell_folders, swc_ids, plot_color) in enumerate(zip(all_volume_file_base_names, all_cell_folders, all_swc_ids, all_plot_colors)):
        swc_cells = []
        for neuron_id, (cell_folder, file_base_name, swc_id) in enumerate(zip(cell_folders, volume_file_base_names, swc_ids)):
            if os.path.exists(fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'):
                swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'
            elif os.path.exists(fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'):
                swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'
            else:
                print(fr'WARNING: {PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc does not exist. Skipping this neuron. ')
                continue

            neuron = navis.read_swc(swc_file)
            neuron.soma = 1
            neuron.nodes.iloc[0, 5] = 2
            neuron.soma_radius = 4
            swc_cells.append(navis.smooth_skeleton(neuron))
            all_swc_cells.append(navis.smooth_skeleton(neuron))
            all_colors.append(plot_color)
            print(f'NEURON {type} in {detailed_brain_regions[np.array(region_masks[(neuron.soma_pos[0][0] / 0.798).astype(int), 
                                                                            (neuron.soma_pos[0][1] / 0.798).astype(int), 
                                                                            (neuron.soma_pos[0][2] / 2).astype(int)]).astype(int) - 1]}')

            mask_neuron = np.zeros((621, 1406, 138))
            mask_neuron[(neuron.nodes['x'] / 0.798).astype(int), (neuron.nodes['y'] / 0.798).astype(int), (neuron.nodes['z'] / 2).astype(int)] = 1
            region_values, region_counts = np.unique_counts(region_masks[mask_neuron.astype(bool)])
            if 0 in region_values:
                print('Warning found nodes outside regions. ')
                neuron_regions = np.array(detailed_brain_regions)[region_values[1:].astype(int) - 1]
                nodes_per_region[type][region_values.astype(int)[1:] - 1, neuron_id] = region_counts[1:]
            else:
                neuron_regions = np.array(detailed_brain_regions)[region_values.astype(int) - 1]
                nodes_per_region[type][region_values.astype(int) - 1, neuron_id] = region_counts
            print(neuron_regions, region_counts)

        outside_tectum = np.logical_not(np.array([np.array(
            tectum_mask[(np.array(swc_cells[i].nodes['x']) / 0.798).astype(int),
                         (np.array(swc_cells[i].nodes['y']) / 0.798).astype(int),
                         (np.array(swc_cells[i].nodes['z']) / 2).astype(int)]).astype(int).min()
                         for i in range(len(swc_cells))]).astype(bool))

        print('Found neurons outside the tectum: ')
        print(cell_folders, outside_tectum)

        if first_set:
            subfig_xy.draw_navis_neuron(swc_cells, brain_regions, navis_view=('x', '-y'), lc=plot_color, lw=0.5, alpha=0.5, rasterized=True)
            subfig_xy.draw_line([311 * 0.798, 311 * 0.798], [350 * 0.798, 750 * 0.798], lc='w')

            subfig_yz.draw_navis_neuron(swc_cells, brain_regions, navis_view=('z', '-y'), lc=plot_color, lw=0.5, alpha=0.5, rasterized=True)
            first_set=False
        else:
            subfig_xy.draw_navis_neuron(swc_cells, [], navis_view=('x', '-y'), lc=plot_color, lw=0.5, alpha=0.5, rasterized=True)
            subfig_xy.draw_line([311 * 0.798, 311 * 0.798], [350 * 0.798, 750 * 0.798], lc='w')

            subfig_yz.draw_navis_neuron(swc_cells, [], navis_view=('z', '-y'), lc=plot_color, lw=0.5, alpha=0.5, rasterized=True)

    for type, color, subfig_index in zip(range(6), ['#359B73', '#2271B2', '#D55E00', '#E69F00', '#9F0162', '#F748A5', ], [1, 0, 4, 5, 3, 2]):
        node_counts = np.array(nodes_per_region[type] / np.nansum(nodes_per_region[type], axis=0)).flatten()
        x_ticks = np.repeat(np.arange(len(detailed_brain_regions)), nodes_per_region[type].shape[1])[::-1]
        subfigs_region_counts[subfig_index].draw_scatter(x_ticks[node_counts > 0], node_counts[node_counts>0], pc=color, ec=None)

    if video_path is not None:
        for i in range(len(all_swc_cells)):
            all_swc_cells[i].nodes['x'] = 621 * 0.798 - all_swc_cells[i].nodes['x']

        brain_regions = [
            navis.read_mesh(fr'{masks_path}\rhombencephalon_(hindbrain).obj', units='microns', output='volume'),
            navis.read_mesh(fr'{masks_path}\mesencephalon_(midbrain).obj', units='microns', output='volume'),
            navis.read_mesh(fr'{masks_path}\prosencephalon_(forebrain).obj', units='microns', output='volume')]

        dpi = 300
        force_new = True
        fig, ax = navis.plot2d(all_swc_cells + brain_regions, linewidth=0.5, method='3d_complex', color=all_colors, figsize=(3, 5), autoscale=True, view=('x', '-y'))
        fig.set_dpi(dpi)
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 1122)
        ax.set_zlim(0, 276)
        ax.set_axis_off()
        ax.set_box_aspect([276, 496, 1122])
        fig.set_layout_engine("none")
        ax.set_position([-0.5, -0.6, 2., 2.])
        ax.set_facecolor("none")

        frames = []
        frames_filenames = []
        for i in range(0, 360, 2):
            frame_filename = rf"{video_path}\temp_img\frame_{i}_PAed_neurons.jpg"
            frames_filenames.append(frame_filename)
            if force_new or not Path(frame_filename).exists():
                ax.view_init(0, i, 180, vertical_axis='y')
                ax.dist = 2.5
                plt.savefig(frame_filename, dpi=dpi)
                if i == 0:
                    plt.savefig(rf"{video_path}\temp_img\frame_{i}_PAed_neurons.pdf", dpi=600)
                print("loading", frame_filename)
            temp_image = np.array(Image.open(frame_filename))
            frames.append(temp_image)
        imageio.mimsave(f"{video_path}/spinning_brain/PAed_neurons.mp4", frames, fps=30, codec="libx264", output_params=["-crf", "20"])

    return

def subfig_group_of_neurons(anterior_xy_plot, anterior_yz_plot, contralateral_xy_plot, contralateral_yz_plot, local_xy_plot, local_yz_plot,
                         ant_contr_plot, ant_contr_x_loc, PA_data_path, file_base_names, cell_folders, swc_ids, type, plot_color, masks_path):
    brain_regions = [navis.read_mesh(fr'{masks_path}\superior_ventral_medulla_oblongata_(entire).obj', units='microns', output='volume'),
                     navis.read_mesh(fr'{masks_path}\mesencephalon_(midbrain).obj', units='microns', output='volume'),
                     navis.read_mesh(fr'{masks_path}\cerebellum.obj', units='microns', output='volume')]

    for t, xy_plot, yz_plot in zip(range(3), [local_xy_plot, contralateral_xy_plot, anterior_xy_plot],
                                   [local_yz_plot, contralateral_yz_plot, anterior_yz_plot]):
        swc_cells = []
        for cell_folder, file_base_name, swc_id in zip(np.array(cell_folders)[np.array(type) == t],
                                                       np.array(file_base_names)[np.array(type) == t],
                                                       np.array(swc_ids)[np.array(type) == t]):
            if os.path.exists(fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'):
                swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'
            elif os.path.exists(fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'):
                swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'
            else:
                print(fr'WARNING: {PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc does not exist. Skipping this neuron. ')
                continue

            neuron = navis.read_swc(swc_file)
            neuron.soma = 1
            neuron.nodes.iloc[0, 5] = 2
            neuron.soma_radius = 4
            swc_cells.append(navis.smooth_skeleton(neuron))

        xy_plot.draw_navis_neuron(swc_cells, brain_regions, navis_view=('x', '-y'), lc=plot_color, lw=0.5, rasterized=True)
        xy_plot.draw_line([311 * 0.798, 311 * 0.798], [350 * 0.798, 750 * 0.798], lc='w')

        yz_plot.draw_navis_neuron(swc_cells, brain_regions, navis_view=('z', '-y'), lc=plot_color, lw=0.5, rasterized=True)

    n_local = np.sum(np.array(type) == 0)
    n_contra = np.sum(np.array(type) == 1)
    n_anterior = np.sum(np.array(type) == 2)
    n_tot = len(type)
    if ant_contr_x_loc == 1:
        plot_colors = ['#93BADA', '#2271B2', '#C0DFF8']
    else:
        plot_colors = ['#8DCDB4', '#359B73', '#C7F2E1', ]
    ant_contr_plot.draw_vertical_bars([ant_contr_x_loc, ant_contr_x_loc, ant_contr_x_loc],
                                      [n_local/n_tot, n_contra/n_tot, n_anterior/n_tot], vertical_bar_bottom=[0, n_local/n_tot, (n_local+n_contra)/n_tot],
                                      lc=plot_colors)
    if ant_contr_x_loc == 1:
        ant_contr_plot.draw_vertical_bars([4, 4, 4], [1/8, 1/8, 1/8], vertical_bar_bottom=[2/8, 4/8, 6/8], lc=['#808080', '#414141', '#C0C0C0'])
        ant_contr_plot.draw_text(4.6, 2.5/8, 'local', textlabel_ha='left')
        ant_contr_plot.draw_text(4.6, 4.5/8, 'contralateral', textlabel_ha='left')
        ant_contr_plot.draw_text(4.6, 6.5/8, 'anterior', textlabel_ha='left')
        ant_contr_plot.draw_text(3.7, 1, 'projection type:', textlabel_ha='left')

    return

def subfig_zoomin_neurons(drive_change_neurons_xy_plot, drive_change_neurons_yz_plot, drive_lumi_neurons_xy_plot, drive_lumi_neurons_yz_plot, PA_data_path,
                          volume_file_base_names, cell_folders, swc_ids, cell_types, plot_colors):
    drive_change_swc_cells = []
    drive_lumi_swc_cells = []
    drive_lumi_plot_colors = []
    drive_change_plot_colors = []
    for cell_folder, file_base_name, swc_id in zip(np.array(cell_folders)[np.array(cell_types) == 0],
                                                   np.array(volume_file_base_names)[np.array(cell_types) == 0],
                                                   np.array(swc_ids)[np.array(cell_types) == 0]):
        if os.path.exists(fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'):
            swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'
        elif os.path.exists(fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'):
            swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'
        else:
            print(fr'WARNING: {PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc does not exist. Skipping this neuron. ')
            continue

        neuron = navis.read_swc(swc_file)
        neuron.soma = 1
        neuron.nodes.iloc[0, 5] = 2
        neuron.soma_radius = 4
        drive_change_swc_cells.append(navis.smooth_skeleton(neuron))
        drive_lumi_swc_cells.append(navis.smooth_skeleton(neuron))
        drive_lumi_plot_colors.append('#2271B2')
        drive_change_plot_colors.append('#2271B2')

    for cell_folder, file_base_name, swc_id in zip(np.array(cell_folders)[np.array(cell_types) == 1],
                                                   np.array(volume_file_base_names)[np.array(cell_types) == 1],
                                                   np.array(swc_ids)[np.array(cell_types) == 1]):
        if os.path.exists(fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'):
            swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'
        elif os.path.exists(fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'):
            swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'
        else:
            print(fr'WARNING: {PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc does not exist. Skipping this neuron. ')
            continue

        neuron = navis.read_swc(swc_file)
        neuron.soma = 1
        neuron.nodes.iloc[0, 5] = 2
        neuron.soma_radius = 4
        drive_change_swc_cells.append(navis.smooth_skeleton(neuron))
        drive_change_plot_colors.append('#D55E00')

    for cell_folder, file_base_name, swc_id in zip(np.array(cell_folders)[np.array(cell_types) == 2],
                                                   np.array(volume_file_base_names)[np.array(cell_types) == 2],
                                                   np.array(swc_ids)[np.array(cell_types) == 2]):
        if os.path.exists(fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'):
            swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'
        elif os.path.exists(fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'):
            swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'
        else:
            print(fr'WARNING: {PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc does not exist. Skipping this neuron. ')
            continue

        neuron = navis.read_swc(swc_file)
        neuron.soma = 1
        neuron.nodes.iloc[0, 5] = 2
        neuron.soma_radius = 4
        drive_lumi_swc_cells.append(navis.smooth_skeleton(neuron))
        drive_lumi_plot_colors.append('#E69F00')

    drive_change_neurons_xy_plot.draw_navis_neuron(drive_change_swc_cells, [], navis_view=('x', '-y'), lc=drive_change_plot_colors, lw=0.5, rasterized=True)
    drive_change_neurons_xy_plot.draw_scatter([134, ], [340, ], pt=MarkerStyle('^', 'left', Affine2D().rotate_deg(225)), pc='k', ec=None, ps=3)
    drive_change_neurons_xy_plot.draw_line([70, 120], [550, 550], lc='k')
    drive_change_neurons_xy_plot.draw_line([70, 70], [500, 550], lc='k')
    drive_change_neurons_xy_plot.draw_text(130, 550, 'm')
    drive_change_neurons_xy_plot.draw_text(70, 490, 'a')

    drive_change_neurons_yz_plot.draw_navis_neuron(drive_change_swc_cells, [], navis_view=('z', '-y'), lc=drive_change_plot_colors, lw=0.5, rasterized=True)
    drive_change_neurons_yz_plot.draw_scatter([100, ], [360, ], pt=MarkerStyle('^', 'left', Affine2D().rotate_deg(225)), pc='k', ec=None, ps=3)
    drive_change_neurons_yz_plot.draw_line([150, 200], [550, 550], lc='k')
    drive_change_neurons_yz_plot.draw_line([200, 200], [500, 550], lc='k')
    drive_change_neurons_yz_plot.draw_text(140, 550, 'v')
    drive_change_neurons_yz_plot.draw_text(200, 490, 'a')

    drive_lumi_neurons_xy_plot.draw_navis_neuron(drive_lumi_swc_cells, [], navis_view=('x', '-y'), lc=drive_lumi_plot_colors, lw=0.5, rasterized=True)
    drive_lumi_neurons_xy_plot.draw_scatter([143, ], [535, ], pt=MarkerStyle('^', 'left', Affine2D().rotate_deg(-35)), pc='k', ec=None, ps=3)
    drive_lumi_neurons_xy_plot.draw_line([70, 120], [550, 550], lc='k')
    drive_lumi_neurons_xy_plot.draw_line([70, 70], [500, 550], lc='k')
    drive_lumi_neurons_xy_plot.draw_text(130, 550, 'm')
    drive_lumi_neurons_xy_plot.draw_text(70, 490, 'a')

    drive_lumi_neurons_yz_plot.draw_navis_neuron(drive_lumi_swc_cells, [], navis_view=('z', '-y'), lc=drive_lumi_plot_colors, lw=0.5, rasterized=True)
    drive_lumi_neurons_yz_plot.draw_scatter([52, ], [535, ], pt=MarkerStyle('^', 'left', Affine2D().rotate_deg(-15)), pc='k', ec=None, ps=3)
    drive_lumi_neurons_yz_plot.draw_line([150, 200], [550, 550], lc='k')
    drive_lumi_neurons_yz_plot.draw_line([200, 200], [500, 550], lc='k')
    drive_lumi_neurons_yz_plot.draw_text(140, 550, 'v')
    drive_lumi_neurons_yz_plot.draw_text(200, 490, 'a')
    drive_lumi_neurons_yz_plot.draw_text(175, 570, '50 \u00b5m')
    return

def mapzebrain_neuron_analysis(path_to_swc_folder_ahb, path_to_swc_folder_tectum, mapzebrain_nrrd_paths, nrrd_plot,
                               all_neurons_xy, all_neurons_yz, xy_plots, yz_plots, masks_path, regions_path, do_video=False, movie_path=None):
    ahb = np.logical_or(nrrd.read(mapzebrain_nrrd_paths[0])[0], nrrd.read(mapzebrain_nrrd_paths[1])[0])
    tectum = np.logical_or(
        np.logical_or(np.logical_or(nrrd.read(mapzebrain_nrrd_paths[2])[0], nrrd.read(mapzebrain_nrrd_paths[3])[0]),
                      nrrd.read(mapzebrain_nrrd_paths[4])[0]), nrrd.read(mapzebrain_nrrd_paths[5])[0])

    rgb_xy = np.ones((621, 1406, 3))
    rgb_yz = np.ones((1406, 138, 3))
    ahb_xy = np.nanmean(ahb, axis=2).astype(bool)
    rgb_xy[ahb_xy == True, 0] = 0.6
    rgb_xy[ahb_xy == True, 1] = 0.6
    rgb_xy[ahb_xy == True, 2] = 0.6
    ahb_yz = np.nanmean(ahb, axis=0).astype(bool)
    rgb_yz[ahb_yz == True, 0] = 0.6
    rgb_yz[ahb_yz == True, 1] = 0.6
    rgb_yz[ahb_yz == True, 2] = 0.6

    tectum_xy = np.nanmean(tectum, axis=2).astype(bool)
    rgb_xy[tectum_xy == True, 0] -= 1
    rgb_xy[tectum_xy == True, 1] -= 1
    rgb_xy[tectum_xy == True, 2] -= 1
    tectum_yz = np.nanmean(tectum, axis=0).astype(bool)
    rgb_yz[tectum_yz == True, 0] -= 1
    rgb_yz[tectum_yz == True, 1] -= 1
    rgb_yz[tectum_yz == True, 2] -= 1

    rgb_xy[rgb_xy == -0.4] = 0.3
    rgb_yz[rgb_yz == -0.4] = 0.3

    nrrd_plot.draw_image(np.swapaxes(rgb_xy, 0, 1), image_origin='lower', extent=(0, 621 * 0.798, 0, 1406 * 0.798))
    nrrd_plot.draw_image(rgb_yz, image_origin='lower', extent=(515, 515 + 138 * 2, 0, 1406 * 0.798))
    nrrd_plot.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
    nrrd_plot.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')

    brain_regions = [navis.read_mesh(fr'{masks_path}\prosencephalon_(forebrain).obj', units='microns', output='volume'),
                     navis.read_mesh(fr'{masks_path}\mesencephalon_(midbrain).obj', units='microns', output='volume'),
                     navis.read_mesh(fr'{masks_path}\rhombencephalon_(hindbrain).obj', units='microns', output='volume')]

    swc_cells = []
    soma_x = []
    soma_y = []
    soma_z = []
    for swc_file in os.listdir(path_to_swc_folder_ahb):
        if not 'to_ZBRAIN' in swc_file:
            continue
        neuron = navis.read_swc(f'{path_to_swc_folder_ahb}/{swc_file}')
        neuron.soma = 1
        neuron.nodes.iloc[0, 5] = 2
        neuron.soma_radius = 5
        swc_cells.append(navis.smooth_skeleton(neuron))
        soma_x.append(neuron.soma_pos[0][0])
        soma_y.append(neuron.soma_pos[0][1])
        soma_z.append(neuron.soma_pos[0][2])

    regions = ['superior_medulla_oblongata']

    region_mask = create_combined_region_npy_mask(regions_path, regions=regions)
    region_mask = binary_erosion(region_mask, iterations=3)

    region_ids = region_mask[
        (np.array(soma_x) / 0.798).astype(int), (np.array(soma_y) / 0.789).astype(int), (np.array(soma_z) / 2).astype(
            int)]

    good_neurons = np.logical_and(region_ids == 1, np.array(soma_x) < 311 * 0.798)

    good_swc_cells_ahb = np.array(swc_cells)[good_neurons].tolist()

    hemisphere_crossing_neurons = np.array(
        [np.sum(good_swc_cells_ahb[i].nodes['x'] > 311 * 0.798) for i in range(len(good_swc_cells_ahb))]).astype(bool)

    dorsal_ahb = 90
    ventral_ahb = 30
    anterior_ahb = 500

    anterior = np.array([np.sum(good_swc_cells_ahb[i].nodes['y'] < anterior_ahb * 0.798) for i in
                         range(len(good_swc_cells_ahb))]).astype(bool)
    ventral = np.array(
        [np.sum(good_swc_cells_ahb[i].nodes['z'] < ventral_ahb * 2) for i in range(len(good_swc_cells_ahb))]).astype(
        bool)
    posterior = np.array(
        [np.sum(good_swc_cells_ahb[i].nodes['y'] > 780 * 0.798) for i in range(len(good_swc_cells_ahb))]).astype(bool)
    dorsal = np.array(
        [np.sum(good_swc_cells_ahb[i].nodes['z'] > dorsal_ahb * 2) for i in range(len(good_swc_cells_ahb))]).astype(
        bool)
    lateral = np.array(
        [np.sum(good_swc_cells_ahb[i].nodes['x'] > 400 * 0.798) for i in range(len(good_swc_cells_ahb))]).astype(bool)

    ant_ipsi = np.logical_and(
        np.logical_and(np.logical_and(np.logical_not(hemisphere_crossing_neurons), anterior), np.logical_not(ventral)),
        np.logical_not(dorsal))
    ant_contra = np.logical_and(np.logical_and(hemisphere_crossing_neurons, anterior), lateral)
    local_ipsi = np.logical_and(
        np.logical_and(np.logical_and(np.logical_not(hemisphere_crossing_neurons), np.logical_not(anterior)),
                       np.logical_not(posterior)), np.logical_not(dorsal))

    swc_cells = []
    soma_x = []
    soma_y = []
    soma_z = []
    for swc_file in os.listdir(path_to_swc_folder_tectum):
        if not 'to_ZBRAIN' in swc_file:
            continue
        neuron = navis.read_swc(f'{path_to_swc_folder_tectum}/{swc_file}')
        neuron.soma = 1
        neuron.nodes.iloc[0, 5] = 2
        neuron.soma_radius = 5
        swc_cells.append(navis.smooth_skeleton(neuron))
        soma_x.append(neuron.soma_pos[0][0])
        soma_y.append(neuron.soma_pos[0][1])
        soma_z.append(neuron.soma_pos[0][2])

    regions = ['tectum']

    region_mask = create_combined_region_npy_mask(regions_path, regions=regions)
    region_mask = binary_erosion(region_mask, iterations=3)

    region_ids = region_mask[
        (np.array(soma_x) / 0.798).astype(int), (np.array(soma_y) / 0.789).astype(int), (np.array(soma_z) / 2).astype(
            int)]

    good_neurons = np.logical_and(region_ids == 1, np.array(soma_x) < 311 * 0.798)

    good_swc_cells_tectum = np.array(swc_cells)[good_neurons].tolist()

    all_neurons_xy.draw_navis_neuron(np.array(good_swc_cells_ahb).tolist(), brain_regions, navis_view=('x', '-y'),
                                     lc='gray', lw=0.25, rasterized=True)
    all_neurons_yz.draw_navis_neuron(np.array(good_swc_cells_ahb).tolist(), brain_regions, navis_view=('z', '-y'),
                                     lc='gray', lw=0.25, rasterized=True)
    all_neurons_xy.draw_navis_neuron(np.array(good_swc_cells_tectum).tolist(), [], navis_view=('x', '-y'), lc='k',
                                     lw=0.25, rasterized=True)
    all_neurons_yz.draw_navis_neuron(np.array(good_swc_cells_tectum).tolist(), [], navis_view=('z', '-y'), lc='k',
                                     lw=0.25, rasterized=True)

    region_of_interest_cells = np.array([np.array(region_mask[
                                                      (np.array(good_swc_cells_tectum[i].nodes['x']) / 0.798).astype(
                                                          int), (np.array(
                                                          good_swc_cells_tectum[i].nodes['y']) / 0.798).astype(int), (
                                                                  np.array(
                                                                      good_swc_cells_tectum[i].nodes['z']) / 2).astype(
                                                          int)]).astype(int).min() for i in
                                         range(len(good_swc_cells_tectum))]).astype(bool)
    outside_region_of_interest_cells = np.logical_not(region_of_interest_cells)

    hemisphere_crossing_neurons = np.array(
        [np.sum(good_swc_cells_tectum[i].nodes['x'] > 311 * 0.798) for i in range(len(good_swc_cells_tectum))]).astype(
        bool)

    outside_and_crossing = np.logical_and(outside_region_of_interest_cells, hemisphere_crossing_neurons)
    outside_not_crossing = np.logical_and(outside_region_of_interest_cells, np.logical_not(hemisphere_crossing_neurons))

    dorsal_tectum = 120
    ventral_tectum = 50
    anterior_tectum = 400

    anterior = np.array([np.sum(good_swc_cells_tectum[i].nodes['y'] < anterior_tectum * 0.798) for i in
                         range(len(good_swc_cells_tectum))]).astype(bool)
    ventral = np.array([np.sum(good_swc_cells_tectum[i].nodes['z'] < ventral_tectum * 2) for i in
                        range(len(good_swc_cells_tectum))]).astype(bool)
    posterior = np.array(
        [np.sum(good_swc_cells_tectum[i].nodes['y'] > 780 * 0.798) for i in range(len(good_swc_cells_tectum))]).astype(
        bool)
    dorsal = np.array([np.sum(good_swc_cells_tectum[i].nodes['z'] > dorsal_tectum * 2) for i in
                       range(len(good_swc_cells_tectum))]).astype(bool)
    front_pathway_neurons = np.logical_and(np.logical_and(outside_not_crossing, anterior), np.logical_not(posterior))
    lateral_pathway_neurons = np.logical_and(
        np.logical_and(np.logical_and(outside_not_crossing, np.logical_not(anterior)), ventral),
        np.logical_not(posterior))

    lateral_cross_neurons = np.logical_and(np.logical_and(np.logical_and(outside_and_crossing, ventral), dorsal),
                                           np.logical_not(posterior))

    for ahb_neurons, tectum_neurons, xy_plot, yz_plot, name, flip in zip([local_ipsi, ant_ipsi,
                                                                          ant_contra, local_ipsi],
                                                                         [lateral_pathway_neurons,
                                                                          front_pathway_neurons,
                                                                          lateral_pathway_neurons,
                                                                          lateral_cross_neurons, ],
                                                                         xy_plots, yz_plots,
                                                                         [f'lateral ipsi', 'anterior ipsi',
                                                                          f'one cross lateral', 'one cross 1', ],
                                                                         [False, False, True, True]):
        print(f'Analysing {name}...')
        subset_swc_cells_ahb = np.array(good_swc_cells_ahb)[ahb_neurons].tolist()
        subset_swc_cells_tectum = np.array(good_swc_cells_tectum)[tectum_neurons].tolist()
        if flip:
            for i in range(len(subset_swc_cells_ahb)):
                subset_swc_cells_ahb[i].nodes['x'] = 621 * 0.798 - subset_swc_cells_ahb[i].nodes['x']
        xy_plot.draw_navis_neuron(subset_swc_cells_ahb, brain_regions, navis_view=('x', '-y'), lc='gray', lw=0.25, rasterized=True)
        yz_plot.draw_navis_neuron(subset_swc_cells_ahb, brain_regions, navis_view=('z', '-y'), lc='gray', lw=0.25, rasterized=True)
        if flip:
            for i in range(len(subset_swc_cells_ahb)):
                subset_swc_cells_ahb[i].nodes['x'] = 621 * 0.798 - subset_swc_cells_ahb[i].nodes['x']
        xy_plot.draw_navis_neuron(subset_swc_cells_tectum, [], navis_view=('x', '-y'), lc='k', lw=0.25, rasterized=True)
        yz_plot.draw_navis_neuron(subset_swc_cells_tectum, [], navis_view=('z', '-y'), lc='k', lw=0.25, rasterized=True)

        xy_plot.draw_line([311 * 0.798, 311 * 0.798], [100 * 0.798, 1006 * 0.798], lc='w')
        xy_plot.draw_text(0, -10, name)

        if do_video:
            print('Creating movie...')
            if flip:
                for i in range(len(subset_swc_cells_ahb)):
                    subset_swc_cells_ahb[i].nodes['x'] = 621 * 0.798 - subset_swc_cells_ahb[i].nodes['x']

            dpi = 300
            force_new = True

            fig, ax = navis.plot2d(subset_swc_cells_ahb + subset_swc_cells_tectum + brain_regions, linewidth=0.5,
                                   method='3d_complex',
                                   color=np.concatenate([np.repeat('gray', len(subset_swc_cells_ahb)),
                                                         np.repeat('k', len(subset_swc_cells_tectum))]),
                                   figsize=(3, 5), autoscale=True)
            fig.set_dpi(dpi)
            ax.set_xlim(0, 500)
            ax.set_ylim(0, 1122)
            ax.set_zlim(0, 276)
            ax.set_axis_off()
            ax.set_box_aspect([276, 496, 1122])
            fig.set_layout_engine("none")
            ax.set_position([-0.5, -0.6, 2., 2.])
            ax.set_facecolor("none")

            frames = []
            frames_filenames = []
            for i in range(0, 360, 2):
                frame_filename = rf"{movie_path}\temp_img\frame_{i}_{name}.jpg"
                frames_filenames.append(frame_filename)
                if force_new or not Path(frame_filename).exists():
                    ax.view_init(0, i, 180, vertical_axis='y')
                    ax.dist = 2.5
                    plt.savefig(frame_filename, dpi=dpi)
                    if i == 0:
                        plt.savefig(rf"{movie_path}\temp_img\frame_{i}_{name}.pdf",
                                    dpi=600)
                    print("loading", frame_filename)
                temp_image = np.array(Image.open(frame_filename))
                frames.append(temp_image)
            imageio.mimsave(f"{movie_path}/spinning_brain/{name}.mp4", frames, fps=30,
                            codec="libx264", output_params=["-crf", "20"])

            print('Done. ')
            if flip:
                for i in range(len(subset_swc_cells_ahb)):
                    subset_swc_cells_ahb[i].nodes['x'] = 621 * 0.798 - subset_swc_cells_ahb[i].nodes['x']
    return


if __name__ == '__main__':
    fig = Figure(fig_width=18, fig_height=17, dpi=900)
    supfig = Figure(fig_width=9, fig_height=17)
    mapzebrain_nrrd_paths = [r'C:/Users/katja/Desktop/region_cut_motion left.nrrd',
                             r'C:/Users/katja/Desktop/region_cut_drive left.nrrd',
                             r'C:/Users/katja/Desktop/region_cut_lumi left.nrrd',
                             r'C:/Users/katja/Desktop/region_cut_diff left.nrrd',
                             r'C:/Users/katja/Desktop/region_cut_bright left.nrrd',
                             r'C:/Users/katja/Desktop/region_cut_dark left.nrrd']

    path_to_swc_folder_ahb = r'C:\Users\Katja\Downloads\Soma_in_mapzebrain_ahb'
    path_to_swc_folder_tectum = r'C:\Users\Katja\Downloads\Soma_in_mapzebrain_tectum'
    masks_path = r'Z:\Zebrafish atlases\z_brain_atlas\region_masks_mapzebrain_atlas2024'
    regions_path = r'Z:\Zebrafish atlases\z_brain_atlas\region_masks_mapzebrain_atlas2024\all_masks_indexed.hdf5'
    video_path_PA = fr'C:\Users\Katja\Desktop\rotating_brain'
    video_path_PA = None
    video_path_MZB = r'C:\Users\Katja\Desktop\zbrain_mesh_mapzebrain'

    nrrd_plot = fig.create_plot(xpos=0.1, ypos=14.25, plot_height=2.3, plot_width=2.3, axis_off=True, xmin=30, xmax=800, ymin=850, ymax=80)

    all_neurons_xy = fig.create_plot(xpos=1.4, ypos=14.25, plot_height=2.3, plot_width=2.3 * 1.5525, axis_off=True)
    all_neurons_yz = fig.create_plot(xpos=3.2, ypos=14.25, plot_height=2.3, plot_width=2.3 / 1.1565, axis_off=True)
    xy_plots = [[]] * 4
    yz_plots = [[]] * 4
    for i in range(4):
        xy_plots[i] = fig.create_plot(xpos=3.75 + i*3.25, ypos=14.25, plot_height=2.3, plot_width=2.3*1.5525, axis_off=True)
        yz_plots[i] = fig.create_plot(xpos=5.55 + i*3.25, ypos=14.25, plot_height=2.3, plot_width=2.3/1.1565, axis_off=True)

    mapzebrain_neuron_analysis(path_to_swc_folder_ahb, path_to_swc_folder_tectum, mapzebrain_nrrd_paths, nrrd_plot, all_neurons_xy, all_neurons_yz, xy_plots, yz_plots, masks_path, regions_path, do_video=False, movie_path=video_path_MZB)

    PA_data_path = r'Y:\M11 2P microscopes\Katja\PA'
    example_func_path = fr'{PA_data_path}\20250128-2\2025-01-28_13-21-48_fish002_setup0_arena0_KS\2025-01-28_13-21-48_fish002_setup0_arena0_KS_preprocessed_data.h5'
    example_HD_path = fr'{PA_data_path}\20250128-2\2025-01-28_14-22-35_fish002_setup0_arena0_KS\2025-01-28_14-22-35_fish002_setup0_arena0_KS_preprocessed_data.h5'
    example_volume_path = fr'{PA_data_path}\20250128-2\2025-01-28_14-58-07_fish002_setup0_arena0_KS\2025-01-28_14-58-07_fish002_setup0_arena0_KS_preprocessed_data.h5'
    example_swc_path = fr'{PA_data_path}\20250128-2\2025-01-28_14-58-07_fish002_setup0_arena0_KS\2025-01-28_14-58-07_fish002_setup0_arena0_KS-000_to_ZBRAIN.swc'
    example_cell_id = 160

    detailed_brain_regions = ['superior_ventral_medulla_oblongata_(entire)', 'superior_raphe',
                              'interpeduncular_nucleus', 'nucleus_isthmi', 'superior_dorsal_medulla_oblongata_stripe_1_(entire)',
                              'superior_dorsal_medulla_oblongata_stripe_2&3', 'superior_dorsal_medulla_oblongata_stripe_4',
                              'cerebellum', 'mesencephalon_(midbrain)', 'tegmentum',
                              'stratum_marginale', 'stratum_opticum',
                              'stratum_fibrosum_et_griseum_superficiale', 'sfgs__sgc',
                              'stratum_griseum_centrale', 'stratum_album_centrale', 'sac__spv',
                              'periventricular_layer',
                              'pretectum', 'dorsal_thalamus_proper', 'intermediate_hypothalamus_(entire)', 'caudal_hypothalamus',
                              'posterior_tuberculum_(basal_part_of_prethalamus_and_thalamus)']
    short_names_brain_regions = ['other sup. vMO', 'sup. Raphe',
                                 'interpeduncular nuc.', 'nuc. isthmi', 'sup. dMO stripe 1', 'sup. dMO stripe 2&3', 'sup. dMO stripe 4',
                                 'cerebellum', 'other midbrain', 'tegmentum',
                                 'SM', 'SO', 'SFGS', 'SFGS-SGC', 'SGC', 'SAC', 'SAC-SPV',
                                 'periventricular layer', 'pretectum', 'dThalamus', 'int. hypothalamus', 'cau. hypothalamus',
                                 'posterior tuberculum']

    example_loc_plot_zoomout = fig.create_plot(xpos=3.5, ypos=11, plot_height=2., plot_width=2., axis_off=True,
                                               xmin=0, xmax=800, ymin=800, ymax=0,)
    example_loc_plot_zoominpre = fig.create_plot(xpos=5.75, ypos=11.55, plot_height=1.45, plot_width=1.45, axis_off=True,
                                                 xmin=539-220/2, xmax=539+220/2, ymin=355+220/2, ymax=355-220/2,)
    example_loc_plot_zoominpost = fig.create_plot(xpos=5.75, ypos=10, plot_height=1.45, plot_width=1.45, axis_off=True,
                                                  xmin=437-548/2, xmax=437+548/2, ymin=274+548/2, ymax=274-548/2,)
    example_traces_plot = fig.create_plot(xpos=3.5, ypos=10, plot_height=0.5, plot_width=2,
                                          xmin=-5, xmax=385, ymin=-1, ymax=5, vspans=[[20, 80, 'lightgray', 1.0], [150, 210, 'lightgray', 1.0], [280, 340, 'lightgray', 1.0]])
    example_tracings_plots = [[]] * 3
    for i, (x, y) in enumerate(zip([338, 290, 254], [465, 415, 364])): #243, 264
        example_tracings_plots[i] = fig.create_plot(xpos=7.4, ypos=12.1 - i * 1.05, plot_height=0.9, plot_width=0.9, axis_off=True,
                                                    xmin=x-20, xmax=x+20, ymin=y+20, ymax=y-20)
    example_neuron_xy_plot = fig.create_plot(xpos=9.1, ypos=11.5, plot_height=1.5, plot_width=1.5*1.5525, axis_off=True)
    example_neuron_yz_plot = fig.create_plot(xpos=11.2, ypos=11.5, plot_height=1.5, plot_width=1.5/1.1565, axis_off=True)
    example_brain_xy_overview_plot = fig.create_plot(xpos=10.8, ypos=9.6, plot_height=2, plot_width=2/2.274, axis_off=True)
    example_brain_yz_overview_plot = fig.create_plot(xpos=11.8, ypos=9.6, plot_height=2, plot_width=2/4.395, axis_off=True)

    sub_plot_method_outline_PA(example_func_path, example_HD_path, example_volume_path, example_swc_path,
                               example_loc_plot_zoomout, example_loc_plot_zoominpre, example_loc_plot_zoominpost,
                               example_traces_plot, example_tracings_plots, example_neuron_xy_plot, example_neuron_yz_plot,
                               example_brain_xy_overview_plot, example_brain_yz_overview_plot,
                               cell_id=example_cell_id, xs=[338, 290, 254, 243], ys=[465, 415, 364, 264], zs=[28, 26, 25, 23],
                               masks_path=masks_path)

    mot_functional_plot = fig.create_plot(xpos=0.5, ypos=6.5, plot_height=1, plot_width=3,
                                          xmin=-5, xmax=385, ymin=-1, ymax=12, vspans=[[20, 80, 'lightgray', 1.0], [150, 210, 'lightgray', 1.0], [280, 340, 'lightgray', 1.0]])
    drive_functional_plot = fig.create_plot(xpos=0.5, ypos=8, plot_height=1, plot_width=3,
                                          xmin=-5, xmax=385, ymin=-1, ymax=6, vspans=[[20, 80, 'lightgray', 1.0], [150, 210, 'lightgray', 1.0], [280, 340, 'lightgray', 1.0]])
    lumi_functional_plot = fig.create_plot(xpos=0.5, ypos=0.5, plot_height=1, plot_width=3,
                                          xmin=-5, xmax=385, ymin=-1, ymax=4, vspans=[[20, 80, 'lightgray', 1.0], [150, 210, 'lightgray', 1.0], [280, 340, 'lightgray', 1.0]])
    diff_functional_plot = fig.create_plot(xpos=0.5, ypos=2, plot_height=1, plot_width=3,
                                          xmin=-5, xmax=385, ymin=-1, ymax=4, vspans=[[20, 80, 'lightgray', 1.0], [150, 210, 'lightgray', 1.0], [280, 340, 'lightgray', 1.0]])
    bright_functional_plot = fig.create_plot(xpos=0.5, ypos=5, plot_height=1, plot_width=3,
                                          xmin=-5, xmax=385, ymin=-1, ymax=3, vspans=[[20, 80, 'lightgray', 1.0], [150, 210, 'lightgray', 1.0], [280, 340, 'lightgray', 1.0]])
    dark_functional_plot = fig.create_plot(xpos=0.5, ypos=3.5, plot_height=1, plot_width=3,
                                          xmin=-5, xmax=385, ymin=-1, ymax=6, vspans=[[20, 80, 'lightgray', 1.0], [150, 210, 'lightgray', 1.0], [280, 340, 'lightgray', 1.0]])
    mot_xy_plot = fig.create_plot(xpos=3.75, ypos=6.5, plot_height=1., plot_width=1.5525, axis_off=True)
    mot_yz_plot = fig.create_plot(xpos=5.25, ypos=6.5, plot_height=1., plot_width=1/1.1565, axis_off=True)
    drive_xy_plot = fig.create_plot(xpos=3.75, ypos=8, plot_height=1., plot_width=1.5525, axis_off=True)
    drive_yz_plot = fig.create_plot(xpos=5.25, ypos=8, plot_height=1., plot_width=1/1.1565, axis_off=True)
    lumi_xy_plot = fig.create_plot(xpos=3.75, ypos=0.5, plot_height=1., plot_width=1.5525, axis_off=True)
    lumi_yz_plot = fig.create_plot(xpos=5.25, ypos=0.5, plot_height=1., plot_width=1/1.1565, axis_off=True)
    diff_xy_plot = fig.create_plot(xpos=3.75, ypos=2, plot_height=1., plot_width=1.5525, axis_off=True)
    diff_yz_plot = fig.create_plot(xpos=5.25, ypos=2, plot_height=1., plot_width=1/1.1565, axis_off=True)
    bright_xy_plot = fig.create_plot(xpos=3.75, ypos=5, plot_height=1., plot_width=1.5525, axis_off=True)
    bright_yz_plot = fig.create_plot(xpos=5.25, ypos=5, plot_height=1., plot_width=1/1.1565, axis_off=True)
    dark_xy_plot = fig.create_plot(xpos=3.75, ypos=3.5, plot_height=1., plot_width=1.5525, axis_off=True)
    dark_yz_plot = fig.create_plot(xpos=5.25, ypos=3.5, plot_height=1., plot_width=1/1.1565, axis_off=True)

    all_neurons_xy_plot = fig.create_plot(xpos=12.5, ypos=10, plot_height=2.5, plot_width=2.5*1.5525, axis_off=True)
    all_neurons_yz_plot = fig.create_plot(xpos=15.75, ypos=10, plot_height=2.5, plot_width=2.5/1.1565, axis_off=True)

    brain_region_plots = [[]] * 6
    for type in range(6):
        if type == 0:
            brain_region_plots[type] = supfig.create_plot(xpos=1.25, ypos=13.5-type*2, plot_height=1.5, plot_width=4, yticks=[0.0, 0.5, 1.0],
                                                          yl='nodes in region (ratio per neuron)', xmin=-1, xmax=len(detailed_brain_regions), ymin=-0.05, ymax=1.05,
                                                          vspans=[[4.5, 14.5, 'lightgray', 1.0], ])
        elif type == 5:
            brain_region_plots[type] = supfig.create_plot(xpos=1.25, ypos=13.5 - type * 2, plot_height=1.5,
                                                          plot_width=4, xticks=np.arange(len(detailed_brain_regions)),
                                                          xticklabels=short_names_brain_regions[::-1],
                                                          yticks=[0.0, 0.5, 1.0], xticklabels_rotation=90,
                                                          xmin=-1, xmax=len(detailed_brain_regions), ymin=-0.05,
                                                          ymax=1.05,
                                                          vspans=[[4.5, 14.5, 'lightgray', 1.0], ])

        else:
            brain_region_plots[type] = supfig.create_plot(xpos=1.25, ypos=13.5-type*2, plot_height=1.5, plot_width=4, yticks=[0.0, 0.5, 1.0],
                                                          xmin=-1, xmax=len(detailed_brain_regions), ymin=-0.05, ymax=1.05,
                                                          vspans=[[4.5, 14.5, 'lightgray', 1.0], ])

    anterior_xy_plot = fig.create_plot(xpos=8.8, ypos=7.5, plot_height=1.5, plot_width=1.5*1.5525, axis_off=True)
    anterior_yz_plot = fig.create_plot(xpos=10.8, ypos=7.5, plot_height=1.5, plot_width=1.5/1.1565, axis_off=True)
    contralateral_xy_plot = fig.create_plot(xpos=11.8, ypos=7.5, plot_height=1.5, plot_width=1.5*1.5525, axis_off=True)
    contralateral_yz_plot = fig.create_plot(xpos=13.8, ypos=7.5, plot_height=1.5, plot_width=1.5/1.1565, axis_off=True)
    local_xy_plot = fig.create_plot(xpos=14.8, ypos=7.5, plot_height=1.5, plot_width=1.5*1.5525, axis_off=True)
    local_yz_plot = fig.create_plot(xpos=16.8, ypos=7.5, plot_height=1.5, plot_width=1.5/1.1565, axis_off=True)
    anterior_xy_mot_plot = fig.create_plot(xpos=8.8, ypos=6, plot_height=1.5, plot_width=1.5*1.5525, axis_off=True)
    anterior_yz_mot_plot = fig.create_plot(xpos=10.8, ypos=6, plot_height=1.5, plot_width=1.5/1.1565, axis_off=True)
    contralateral_xy_mot_plot = fig.create_plot(xpos=11.8, ypos=6, plot_height=1.5, plot_width=1.5*1.5525, axis_off=True)
    contralateral_yz_mot_plot = fig.create_plot(xpos=13.8, ypos=6, plot_height=1.5, plot_width=1.5/1.1565, axis_off=True)
    local_xy_mot_plot = fig.create_plot(xpos=14.8, ypos=6, plot_height=1.5, plot_width=1.5*1.5525, axis_off=True)
    local_yz_mot_plot = fig.create_plot(xpos=16.8, ypos=6, plot_height=1.5, plot_width=1.5/1.1565, axis_off=True)
    anterior_vs_contra_count_plot = fig.create_plot(xpos=6.5, ypos=6, plot_height=3, plot_width=1.25, axis_off=True,
                                                    xmin=0, xmax=5, ymin=0, ymax=1, yl='proportion neurons')

    drive_change_neurons_xy_plot = fig.create_plot(xpos=6.4, ypos=3.5, plot_height=2., plot_width=2, axis_off=True)
    drive_change_neurons_yz_plot = fig.create_plot(xpos=8.5, ypos=3.5, plot_height=2, plot_width=2, axis_off=True)
    drive_lumi_neurons_xy_plot = fig.create_plot(xpos=6.4, ypos=1.5, plot_height=2., plot_width=2, axis_off=True)
    drive_lumi_neurons_yz_plot = fig.create_plot(xpos=8.5, ypos=1.5, plot_height=2, plot_width=2, axis_off=True)

    mot_file_base_names = ['2024-01-15_14-32-16_fish000_KS', '2024-01-16_10-23-16_fish000_KS', '2024-01-29_13-28-48_fish000_KS', '2024-01-30_14-29-11_fish002_KS',
                           '2024-02-26_15-33-10_fish002_KS', '2024-02-27_10-10-39_fish000_KS', '2024-04-08_10-57-50_fish001_KS', '2024-04-09_14-45-31_fish002_KS',
                           '2025-04-03_09-40-14_fish000_setup0_arena0_HN', '2025-04-03_09-40-14_fish000_setup0_arena0_HN', '2025-04-03_09-40-14_fish000_setup0_arena0_HN', '2025-04-03_09-40-14_fish000_setup0_arena0_HN']
    mot_volume_file_base_names = ['2024-01-15_15-27-36_fish000_KS', '2024-01-16_11-28-32_fish000_KS', '2024-01-29_14-27-53_fish000_KS', '2024-01-30_15-15-23_fish002_KS',
                                  '2024-02-26_16-38-23_fish002_KS', '2024-02-27_11-19-48_fish000_KS', '2024-04-08_12-06-09_fish001_KS', '2024-04-09_15-39-07_fish002_KS',
                                  '2025-04-03_17-38-56_fish000_setup0_arena0_HN', '2025-04-03_17-38-56_fish000_setup0_arena0_HN', '2025-04-03_17-38-56_fish000_setup0_arena0_HN', '2025-04-03_17-38-56_fish000_setup0_arena0_HN']
    mot_cell_folders = ['20240115-1', '20240116-1', '20240129-1', '20240130-3',
                        '20240226-3', '20240227-1', '20240408-2', '20240409-3',
                        '20250403-0', '20250403-0', '20250403-0', '20250403-0']
    mot_cell_mask_IDs = [439, 210, 358, 550, 255, 231, 838, 187, 307, 433, 637, 464]
    mot_swc_ids = [0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 2, 3]
    mot_z_planes = [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0]
    mot_type = [0, 0, 0, 1, 0, 2, 1, 0, 0, 0, 0, 0]

    drive_file_base_names = ['2023-12-12_12-14-52', '2024-01-16_14-10-16_fish001_KS', '2024-01-22_13-58-17_fish001_KS', '2024-02-26_11-03-33_fish000_KS',
                             '2024-02-27_14-48-03_fish002_KS', '2024-04-09_14-46-28_fish003_KS', '2025-04-03_09-40-14_fish000_setup0_arena0_HN',
                             '2025-02-17_09-29-26_fish001_setup0_arena0_KS', '2025-02-17_09-29-26_fish001_setup0_arena0_KS', '2025-02-17_09-29-26_fish001_setup0_arena0_KS',
                             '2025-02-17_09-29-26_fish001_setup0_arena0_KS', '2025-02-17_09-29-26_fish001_setup0_arena0_KS']
    drive_volume_file_base_names = ['2023-12-12_14-06-43', '2024-01-16_15-07-59_fish001_KS', '2024-01-22_15-05-28_fish001_KS', '2024-02-26_12-26-15_fish000_KS',
                                    '2024-02-27_15-44-31_fish002_KS', '2024-04-09_15-40-32_fish003_KS', '2025-04-03_17-38-56_fish000_setup0_arena0_HN',
                                    '2025-02-17_15-27-37_fish001_setup0_arena0_KS', '2025-02-17_15-27-37_fish001_setup0_arena0_KS', '2025-02-17_15-27-37_fish001_setup0_arena0_KS',
                                    '2025-02-17_15-27-37_fish001_setup0_arena0_KS', '2025-02-17_15-27-37_fish001_setup0_arena0_KS']
    drive_cell_folders = ['20231212-1', '20240116-2', '20240122-2', '20240226-1',
                          '20240227-3', '20240409-4', '20250403-0',
                          '20250217-1', '20250217-1', '20250217-1',
                          '20250217-1', '20250217-1']
    drive_cell_mask_IDs = [87, 215, 617, 558, 134, 458, 361, 498, 517, 561, 530, 585]
    drive_swc_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4]
    drive_z_planes = [0, 0, 0, 0, 0, 0, 3, 0, 3, 1, 0, 2]
    drive_type = [1, 1, 2, 2, 2, 0, 0, 0, 2, 1, 0, 0,]

    lumi_file_base_names = ['2025-01-13_10-46-04_fish000_setup0_arena0_KS', '2025-04-15_09-55-21_fish000_setup0_arena0_HN', '2025-04-15_09-55-21_fish000_setup0_arena0_HN', '2025-04-15_09-55-21_fish000_setup0_arena0_HN',
                            '2025-05-06_08-44-53_fish000_setup0_arena0_HN', '2025-05-06_08-44-53_fish000_setup0_arena0_HN', '2025-05-06_08-44-53_fish000_setup0_arena0_HN', '2025-05-06_08-44-53_fish000_setup0_arena0_HN',
                            '2025-05-06_08-44-53_fish000_setup0_arena0_HN', '2025-05-06_08-44-53_fish000_setup0_arena0_HN']
    lumi_volume_file_base_names = ['2025-01-13_12-00-41_fish000_setup0_arena0_KS', '2025-04-15_12-22-11_fish000_setup0_arena0_HN', '2025-04-15_16-53-32_fish000_setup0_arena0_HN', '2025-04-15_16-53-32_fish000_setup0_arena0_HN',
                                   '2025-05-06_10-44-27_fish000_setup0_arena0_HN', '2025-05-06_17-48-53_fish000_setup0_arena0_HN', '2025-05-06_17-48-53_fish000_setup0_arena0_HN', '2025-05-06_17-48-53_fish000_setup0_arena0_HN',
                                   '2025-05-06_17-48-53_fish000_setup0_arena0_HN', '2025-05-06_17-48-53_fish000_setup0_arena0_HN']
    lumi_cell_folders = ['20250113-0', '20250415-0', '20250415-0', '20250415-0',
                         '20250506-0', '20250506-0', '20250506-0', '20250506-0',
                         '20250506-0', '20250506-0']
    lumi_cell_mask_IDs = [347, 388, 269, 388, 381, 100, 428, 14, 420, 350]
    lumi_swc_ids = [0, 0, 1, 2, 0, 1, 4, 5, 2, 3]
    lumi_z_planes = [0, 3, 2, 1, 3, 2, 1, 1, 0, 0]

    diff_file_base_names = ['2024-04-09_10-00-45_fish001_KS', '2024-05-24_10-57-08_fish001_KS', '2025-01-27_11-37-08_fish001_setup1_arena0_KS', '2025-01-28_13-21-48_fish002_setup0_arena0_KS',
                            '2025-02-25_09-36-39_fish000_setup0_arena0_HN', '2025-03-04_09-31-15_fish000_setup0_arena0_HN', '2025-03-04_09-31-15_fish000_setup0_arena0_HN', '2025-03-04_09-31-15_fish000_setup0_arena0_HN',
                            '2025-03-04_09-31-15_fish000_setup0_arena0_HN', '2025-03-04_09-31-15_fish000_setup0_arena0_HN', '2025-03-04_09-31-15_fish000_setup0_arena0_HN', '2025-03-04_09-31-15_fish000_setup0_arena0_HN',
                            '2025-03-04_09-31-15_fish000_setup0_arena0_HN', '2025-03-04_09-31-15_fish000_setup0_arena0_HN', '2025-03-18_08-56-10_fish000_setup0_arena0_HN', '2025-03-18_08-56-10_fish000_setup0_arena0_HN',
                            '2025-03-18_08-56-10_fish000_setup0_arena0_HN', '2025-03-18_08-56-10_fish000_setup0_arena0_HN']
    diff_volume_file_base_names = ['2024-04-09_10-56-23_fish001_KS', '2024-05-24_11-37-43_fish001_KS', '2025-01-27_12-35-14_fish001_setup1_arena0_KS', '2025-01-28_14-58-07_fish002_setup0_arena0_KS',
                                   '2025-02-25_11-07-34_fish000_setup0_arena0_HN', '2025-03-04_18-17-30_fish000_setup0_arena0_HN', '2025-03-04_18-17-30_fish000_setup0_arena0_HN', '2025-03-04_18-17-30_fish000_setup0_arena0_HN',
                                   '2025-03-04_18-17-30_fish000_setup0_arena0_HN', '2025-03-04_18-17-30_fish000_setup0_arena0_HN', '2025-03-04_18-17-30_fish000_setup0_arena0_HN', '2025-03-04_18-17-30_fish000_setup0_arena0_HN',
                                   '2025-03-04_18-17-30_fish000_setup0_arena0_HN', '2025-03-04_18-17-30_fish000_setup0_arena0_HN', '2025-03-18_10-44-23_fish000_setup0_arena0_HN', '2025-03-18_15-07-25_fish000_setup0_arena0_HN',
                                   '2025-03-18_15-07-25_fish000_setup0_arena0_HN', '2025-03-18_15-07-25_fish000_setup0_arena0_HN']
    diff_cell_folders = ['20240409-2', '20240524-1', '20250127-1', '20250128-2',
                         '20250225-0', '20250304-0', '20250304-0', '20250304-0',
                         '20250304-0', '20250304-0', '20250304-0', '20250304-0',
                         '20250304-0', '20250304-0', '20250318-0', '20250318-0',
                         '20250318-0', '20250318-0']
    diff_cell_mask_IDs = [219, 162, 417, 160, 358, 295, 109, 323, 119, 138, 174, 230, 245, 273, 351, 445, 486, 345]
    diff_swc_ids = [0, 0, 0, 0, 0, 0, 8, 7, 1, 2, 3, 6, 5, 4, 0, 2, 1, 3]
    diff_z_planes = [0, 0, 0, 0, 3, 3, 0, 0, 1, 1, 1, 3, 3, 3, 3, 0, 0, 1]

    bright_file_base_names = ['2025-02-16_09-32-03_fish001_setup0_arena0_KS', '2025-04-15_09-55-21_fish000_setup0_arena0_HN', '2025-04-15_09-55-21_fish000_setup0_arena0_HN', '2025-04-15_09-55-21_fish000_setup0_arena0_HN']
    bright_volume_file_base_names = ['2025-02-16_11-05-39_fish001_setup0_arena0_KS', '2025-04-15_16-53-32_fish000_setup0_arena0_HN', '2025-04-15_16-53-32_fish000_setup0_arena0_HN', '2025-04-15_16-53-32_fish000_setup0_arena0_HN']
    bright_cell_folders = ['20250216-1', '20250415-0', '20250415-0', '20250415-0']
    bright_cell_mask_IDs = [332, 49, 390, 275]
    bright_swc_ids = [0, 3, 1, 4]
    bright_z_planes = [0, 1, 1, 0]

    dark_file_base_names = ['2025-02-16_09-29-42_fish000_setup1_arena0_KS', '2025-02-16_09-29-42_fish000_setup1_arena0_KS', '2025-02-16_09-29-42_fish000_setup1_arena0_KS', '2025-02-16_09-29-42_fish000_setup1_arena0_KS',
                            '2025-02-16_09-29-42_fish000_setup1_arena0_KS', '2025-02-16_09-29-42_fish000_setup1_arena0_KS', '2025-02-16_09-29-42_fish000_setup1_arena0_KS', '2025-02-16_09-29-42_fish000_setup1_arena0_KS',
                            '2025-02-16_09-29-42_fish000_setup1_arena0_KS']
    dark_volume_file_base_names = ['2025-02-16_15-12-29_fish000_setup1_arena0_KS', '2025-02-16_15-12-29_fish000_setup1_arena0_KS', '2025-02-16_15-12-29_fish000_setup1_arena0_KS', '2025-02-16_15-12-29_fish000_setup1_arena0_KS',
                                   '2025-02-16_15-12-29_fish000_setup1_arena0_KS', '2025-02-16_15-12-29_fish000_setup1_arena0_KS', '2025-02-16_15-12-29_fish000_setup1_arena0_KS', '2025-02-16_15-12-29_fish000_setup1_arena0_KS',
                                   '2025-02-16_15-12-29_fish000_setup1_arena0_KS']
    dark_cell_folders = ['20250216-0', '20250216-0', '20250216-0', '20250216-0',
                         '20250216-0', '20250216-0', '20250216-0', '20250216-0',
                         '20250216-0', ]
    dark_cell_mask_IDs = [199, 217, 385, 183, 168, 348, 341]
    dark_swc_ids = [0, 6, 5, 4, 3, 1, 2]
    dark_z_planes = [0, 2, 2, 1, 1, 0, 0]

    zoomin_volume_file_base_names = ['2024-01-22_15-05-28_fish001_KS', '2024-02-26_12-26-15_fish000_KS', '2024-02-27_15-44-31_fish002_KS',
                                     '2024-04-09_10-56-23_fish001_KS', '2025-01-28_14-58-07_fish002_setup0_arena0_KS', '2025-02-25_11-07-34_fish000_setup0_arena0_HN',
                                     '2025-03-04_18-17-30_fish000_setup0_arena0_HN', '2025-03-04_18-17-30_fish000_setup0_arena0_HN', '2025-03-04_18-17-30_fish000_setup0_arena0_HN',
                                     '2025-04-15_12-22-11_fish000_setup0_arena0_HN', '2025-04-15_16-53-32_fish000_setup0_arena0_HN', '2025-05-06_17-48-53_fish000_setup0_arena0_HN',
                                     '2025-02-17_15-27-37_fish001_setup0_arena0_KS']
    zoomin_cell_folders = ['20240122-2', '20240226-1', '20240227-3',
                           '20240409-2', '20250128-2', '20250225-0',
                           '20250304-0', '20250304-0', '20250304-0',
                           '20250415-0', '20250415-0', '20250506-0',
                           '20250217-1']
    zoomin_swc_ids = [0, 0, 0,
                      0, 0, 0,
                      8, 1, 2,
                      0, 2, 5,
                      1]
    zoomin_types = [0, 0, 0,
                    1, 1, 1,
                    1, 1, 1,
                    2, 2, 2,
                    0]
    zoomin_plot_colors = ['#2271B2', '#2271B2', '#2271B2',
                          '#D55E00',
                          '#D55E00', '#D55E00', '#D55E00',
                          '#E69F00', '#E69F00', '#E69F00',
                          '#2271B2',]

    all_volume_file_base_names = [mot_volume_file_base_names, drive_volume_file_base_names, diff_volume_file_base_names, lumi_volume_file_base_names, dark_volume_file_base_names, bright_volume_file_base_names,  ]
    all_cell_folders = [mot_cell_folders, drive_cell_folders, diff_cell_folders, lumi_cell_folders, dark_cell_folders, bright_cell_folders, ]
    all_swc_ids = [mot_swc_ids, drive_swc_ids, diff_swc_ids, lumi_swc_ids, dark_swc_ids, bright_swc_ids, ]
    all_plot_colors = ['#359B73', '#2271B2',   '#D55E00', '#E69F00', '#9F0162', '#F748A5', ]

    subfig_PA_functional_loc(mot_functional_plot, mot_xy_plot, mot_yz_plot, PA_data_path, mot_file_base_names, mot_volume_file_base_names, mot_cell_folders, mot_cell_mask_IDs, mot_swc_ids, masks_path, z_planes=mot_z_planes, plot_color='#359B73', plot_color_2='#8DCDB4')
    subfig_PA_functional_loc(drive_functional_plot, drive_xy_plot, drive_yz_plot,  PA_data_path, drive_file_base_names, drive_volume_file_base_names, drive_cell_folders, drive_cell_mask_IDs, drive_swc_ids, masks_path, z_planes=drive_z_planes, plot_color='#2271B2', plot_color_2='#93BADA')
    subfig_PA_functional_loc(lumi_functional_plot, lumi_xy_plot, lumi_yz_plot,  PA_data_path, lumi_file_base_names, lumi_volume_file_base_names, lumi_cell_folders, lumi_cell_mask_IDs, lumi_swc_ids, masks_path, z_planes=lumi_z_planes, plot_color='#E69F00', plot_color_2='#F7D280')
    subfig_PA_functional_loc(diff_functional_plot, diff_xy_plot, diff_yz_plot,  PA_data_path, diff_file_base_names, diff_volume_file_base_names, diff_cell_folders, diff_cell_mask_IDs, diff_swc_ids, masks_path, z_planes=diff_z_planes, plot_color='#D55E00', plot_color_2='#EEAE7C')
    subfig_PA_functional_loc(bright_functional_plot, bright_xy_plot, bright_yz_plot,  PA_data_path, bright_file_base_names, bright_volume_file_base_names, bright_cell_folders, bright_cell_mask_IDs, bright_swc_ids, masks_path, z_planes=bright_z_planes, plot_color='#F748A5', plot_color_2='#F7A4D0')
    subfig_PA_functional_loc(dark_functional_plot, dark_xy_plot, dark_yz_plot,  PA_data_path, dark_file_base_names, dark_volume_file_base_names, dark_cell_folders, dark_cell_mask_IDs, dark_swc_ids, masks_path, z_planes=dark_z_planes, plot_color='#9F0162', plot_color_2='#CC7CAD', timescalebar=True)

    subfig_all_PA_neurons_loc(all_neurons_xy_plot, all_neurons_yz_plot, brain_region_plots, PA_data_path, all_volume_file_base_names, all_cell_folders, all_swc_ids, all_plot_colors, detailed_brain_regions, masks_path, regions_path, video_path=video_path_PA)

    subfig_group_of_neurons(anterior_xy_mot_plot, anterior_yz_mot_plot, contralateral_xy_mot_plot, contralateral_yz_mot_plot, local_xy_mot_plot, local_yz_mot_plot,
                         anterior_vs_contra_count_plot, 2, PA_data_path, mot_volume_file_base_names, mot_cell_folders, mot_swc_ids, mot_type, '#359B73', masks_path)

    subfig_group_of_neurons(anterior_xy_plot, anterior_yz_plot, contralateral_xy_plot, contralateral_yz_plot, local_xy_plot, local_yz_plot,
                         anterior_vs_contra_count_plot, 1, PA_data_path, drive_volume_file_base_names, drive_cell_folders, drive_swc_ids, drive_type, '#2271B2', masks_path)

    subfig_zoomin_neurons(drive_change_neurons_xy_plot, drive_change_neurons_yz_plot, drive_lumi_neurons_xy_plot, drive_lumi_neurons_yz_plot, PA_data_path,
                          zoomin_volume_file_base_names, zoomin_cell_folders, zoomin_swc_ids, zoomin_types, zoomin_plot_colors)

    fig.save('C:/users/katja/Desktop/fig5.pdf')
    supfig.save('C:/users/katja/Desktop/sup_figS6.pdf')