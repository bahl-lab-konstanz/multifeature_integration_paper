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

def sub_plot_method_outline_PA(path_to_example_data_func, path_to_example_data_HD, path_to_example_data_volume,
                               path_to_example_swc,
                               subfig_zoomout, subfig_zoominpre, subfig_zoominpost, subfig_traces, subfigs_tracings,
                               subfig_neuron_xy, subfig_neuron_yz, subfig_brain_xy, subfig_brain_yz, cell_id, xs, ys,
                               zs, masks_path):
    '''
    This function plots the method outline of the PA procedure using one example neuron.
    :param path_to_example_data_func: Path to the funcitonal data of the example neuron.
    :param path_to_example_data_HD: Path to the close-up stack of the example neuron.
    :param path_to_example_data_volume: Path to the volume data of the example neuron.
    :param path_to_example_swc: Path to the swc file of the example neuron.
    :param subfig_zoomout: Subfigure showing the functional imaging overview.
    :param subfig_zoominpre: Subfigure showing a zoom-in of the functional imaging pre-PA
    :param subfig_zoominpost: Subfigure showing a zoom-in of the close-up stack post-PA.
    :param subfig_traces: Subfigure showing the funcitonal traces of the example neuorn.
    :param subfigs_tracings: List of subfigures to show the zoom in of the volume post-PA, stepping through the neuron tracing.
    :param subfig_neuron_xy: Subfigure showing the neuron morphology in xy view
    :param subfig_neuron_yz: Subfigure showing the neuron morphology in yz view
    :param subfig_brain_xy: Subfigure showing the full brain reference in xy view
    :param subfig_brain_yz: Subfigure showing the full brain reference in yz view.
    :param cell_id: Segmentation ID of the example neuron.
    :param xs: List of x positions to center the subfigs_tracings on
    :param ys: List of y positions to center the subfigs_tracings on
    :param zs: List of z positions to center the subfigs_tracings on
    :param masks_path: Path to the folder containing obj files of the major brain regions.
    '''

    # Load the functional data of the example neuron (both the average image as the activity traces).
    preproc_hdf5 = h5py.File(path_to_example_data_func, "r")

    all_dynamics = np.array(preproc_hdf5[
                                f'repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/dLpN_dRpR_dNpL/F'])

    avg_im = np.array(preproc_hdf5['repeat00_tile000_z000_950nm']['preprocessed_data']['fish00'][
                          'imaging_data_channel0_time_averaged']).reshape(799, 799)

    # Draw the average image of the functional data in the zoom-out and zoom-in panels.
    subfig_zoomout.draw_image(np.clip(avg_im, np.nanpercentile(avg_im, 5), np.nanpercentile(avg_im, 98)),
                              colormap='gray',
                              extent=(0, 800, 800, 0), image_origin='upper')
    subfig_zoominpre.draw_image(np.clip(avg_im, np.nanpercentile(avg_im, 5), np.nanpercentile(avg_im, 98)),
                                colormap='gray',
                                extent=(0, 800, 800, 0), image_origin='upper')

    # Load the contour of the example neuron and add it to the zoom out image.
    unit_contour = np.array(
        preproc_hdf5['repeat00_tile000_z000_950nm']['preprocessed_data']['fish00']['cellpose_segmentation'][
            'unit_contours'][f'{10000 + cell_id}'])
    subfig_zoomout.draw_line(unit_contour[:, 0], unit_contour[:, 1], lc='#D55E00', lw=0.4)

    # Plot the outline of the zoom-in image in the zoom-out image
    subfig_zoomout.draw_line([539 - 220 / 2, 539 - 220 / 2, 539 + 220 / 2, 539 + 220 / 2, 539 - 220 / 2, ],
                             [355 + 220 / 2, 355 - 220 / 2, 355 - 220 / 2, 355 + 220 / 2, 355 + 220 / 2, ], lc='w',
                             lw=0.5)

    # mark the neuron of interest.
    subfig_zoominpre.draw_scatter([539 + 8.8, ], [355 - 8.8, ], pt=MarkerStyle('^', 'left', Affine2D().rotate_deg(135)),
                                  pc='#D55E00', ec=None, ps=4)

    preproc_hdf5.close()

    # Load the close-up stack post PA and plot the average image.
    preproc_hdf5 = h5py.File(path_to_example_data_HD, "r")
    avg_im = np.array(preproc_hdf5['average_stack_repeat00_tile000_950nm_channel0'])[23, :, :].reshape(799, 799)

    subfig_zoominpost.draw_image(np.clip(avg_im, np.nanpercentile(avg_im, 5), np.nanpercentile(avg_im, 95)),
                                 colormap='gray',
                                 extent=(0, 800, 800, 0), image_origin='upper')

    # Mark the neuron of interest.
    subfig_zoominpost.draw_scatter([437 + 21.92, ], [274 - 21.92, ],
                                   pt=MarkerStyle('^', 'left', Affine2D().rotate_deg(135)), pc='#D55E00', ec=None, ps=4)

    preproc_hdf5.close()

    # load the functional actvity of our example cell, calculate df/f0.
    cell_dynamics = all_dynamics[:, cell_id, :]
    df_f0 = (cell_dynamics - np.nanmean(cell_dynamics[:, :10], axis=1)[:, None]) / np.nanmean(cell_dynamics[:, :10],
                                                                                              axis=1)[:, None]

    # plot the traces for the single trials
    for trial in range(df_f0.shape[0]):
        subfig_traces.draw_line(np.arange(0, 120), df_f0[trial, :120], lc='#EEAE7C', lw=0.5)
        subfig_traces.draw_line(np.arange(130, 250), df_f0[trial, 120:240], lc='#EEAE7C', lw=0.5)
        subfig_traces.draw_line(np.arange(260, 380), df_f0[trial, 240:], lc='#EEAE7C', lw=0.5)
    # plot the mean trace and scale bars
    subfig_traces.draw_line(np.arange(0, 120), np.nanmean(df_f0, axis=0)[:120], lc='#D55E00', lw=1)
    subfig_traces.draw_line(np.arange(130, 250), np.nanmean(df_f0, axis=0)[120:240], lc='#D55E00', lw=1)
    subfig_traces.draw_line(np.arange(260, 380), np.nanmean(df_f0, axis=0)[240:], lc='#D55E00', lw=1)
    subfig_traces.draw_line([-2, -2], [0, 1], lc='k')
    subfig_traces.draw_text(-40, 0.5, '1 dF/F\u2080', textlabel_rotation=90, textlabel_ha='left')
    subfig_traces.draw_line([355, 375], [-0.95, -0.95], lc='k')
    subfig_traces.draw_text(360, -3, '10s', textlabel_ha='center')

    # Load the volume data post PA
    preproc_hdf5 = h5py.File(path_to_example_data_volume, "r")
    avg_im = np.array(preproc_hdf5['average_stack_repeat00_tile000_950nm_channel0'])
    preproc_hdf5.close()

    # Loop over the zoom-in tracing plots to track the neuron through the volume.
    for subfig, z, x, y, contrast in zip(subfigs_tracings, zs, xs, ys, [99, 97, 90, 85]):
        subfig.draw_image(np.clip(avg_im[z, :, :], np.nanpercentile(avg_im, 5), np.nanpercentile(avg_im, contrast)),
                          colormap='gray',
                          extent=(0, 800, 800, 0), image_origin='upper')
        subfig.draw_text(x + 40, y, f'z={2 * (z - zs[0]):.0f}\u00b5m')

    # Load the total brain region and reference regions used throughout figure 5.
    total_brain_regions = [
        navis.read_mesh(fr'{masks_path}\prosencephalon_(forebrain).obj', units='microns', output='volume'),
        navis.read_mesh(fr'{masks_path}\mesencephalon_(midbrain).obj', units='microns', output='volume'),
        navis.read_mesh(fr'{masks_path}\rhombencephalon_(hindbrain).obj', units='microns', output='volume')]
    brain_regions = [navis.read_mesh(fr'{masks_path}\superior_ventral_medulla_oblongata_(entire).obj', units='microns',
                                     output='volume'),
                     navis.read_mesh(fr'{masks_path}\mesencephalon_(midbrain).obj', units='microns', output='volume'),
                     navis.read_mesh(fr'{masks_path}\cerebellum.obj', units='microns', output='volume')]

    # Load the example neuron, add the soma and smoothen the skeleton.
    neuron = navis.read_swc(path_to_example_swc)
    neuron.soma = 1
    neuron.nodes.iloc[0, 5] = 2
    neuron.soma_radius = 4
    cell_swc = navis.smooth_skeleton(neuron)

    # Plot the neuron in the reference regions in xy view
    subfig_neuron_xy.draw_navis_neuron(cell_swc, brain_regions, navis_view=('x', '-y'), lc='#D55E00', lw=0.5,
                                       rasterized=True)
    subfig_neuron_xy.draw_line([311 * 0.798, 311 * 0.798], [350 * 0.798, 750 * 0.798], lc='w')
    subfig_neuron_xy.draw_line([10, 110], [600, 600], lc='k')
    subfig_neuron_xy.draw_line([10, 10], [600, 500], lc='k')
    subfig_neuron_xy.draw_text(130, 600, 'm')
    subfig_neuron_xy.draw_text(10, 480, 'a')

    # Plot the neuron in the reference regions in yz view
    subfig_neuron_yz.draw_navis_neuron(cell_swc, brain_regions, navis_view=('z', '-y'), lc='#D55E00', lw=0.5,
                                       rasterized=True)
    subfig_neuron_yz.draw_line([180, 280], [600, 600], lc='k')
    subfig_neuron_yz.draw_line([280, 280], [600, 500], lc='k')
    subfig_neuron_yz.draw_text(160, 600, 'v')
    subfig_neuron_yz.draw_text(280, 480, 'a')

    # Plot the full brain with highlighted reference brain regions as guide.
    subfig_brain_xy.draw_navis_neuron(None, total_brain_regions, navis_view=('x', '-y'), lw=0.5, rasterized=True)
    subfig_brain_xy.draw_navis_neuron(None, brain_regions, navis_color='gray', navis_view=('x', '-y'), lw=0.5,
                                      rasterized=True)
    subfig_brain_yz.draw_navis_neuron(None, total_brain_regions, navis_view=('z', '-y'), lw=0.5, rasterized=True)
    subfig_brain_yz.draw_navis_neuron(None, brain_regions, navis_color='gray', navis_view=('z', '-y'), lw=0.5,
                                      rasterized=True)

    return


def subfig_PA_functional_loc(subfig, subfig_xy, subfig_yz, PA_data_path, file_base_names, volume_file_base_names,
                             cell_folders, cell_mask_IDs, swc_ids, masks_path, z_planes=None, plot_color='k',
                             plot_color_2='g', timescalebar=False):
    '''
    This function plots all neurons and their functional activity for one functional type.
    This is related to Fig. 5e
    :param subfig: Subfigure showing the activity traces.
    :param subfig_xy: Subfigure showing the neurons in xy view.
    :param subfig_yz: Subfigure showing the neurons in yz view.
    :param PA_data_path: Path to the folder containing all PA data.
    :param file_base_names: List with all functional data file names for each neuron.
    :param volume_file_base_names: List with all volume data file names for each neuron.
    :param cell_folders: List with all cell folders for each neuron.
    :param cell_mask_IDs: List with all segmentation mask IDs for each neuron.
    :param swc_ids: List with all swc IDs for each neuron.
    :param masks_path: Path to the folder wtih obj files of major brain regions.
    :param z_planes: List with all z_planes for each neuron.
    :param plot_color: Plot color for the neurons (mean trace and morphology).
    :param plot_color_2: Plot color for the neurons (single traces).
    :param timescalebar: If true draw the timescale bar.
    '''
    # Loop over all folders and collect the filenames of the preprocessed data hdf5 files.
    preproc_paths = []
    for file_base_name, cell_folder in zip(file_base_names, cell_folders):
        preproc_paths.append(fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}_preprocessed_data.h5')

    # Loop over all neurons to plot the functional activity
    first_fish = True
    if z_planes is None:
        z_planes = np.zeros(len(preproc_paths))
    for preproc_path, cell_mask_ID, z_plane in zip(preproc_paths, cell_mask_IDs, z_planes):
        # Load the preprocessed_data frames contain the functional activity of the targeted neuron.
        print(preproc_path, cell_mask_ID, z_plane)
        preproc_hdf5 = h5py.File(preproc_path, 'r')
        if '2025' in preproc_path or '20240115-1' in preproc_path or '20240524-1' in preproc_path:
            all_dynamics = np.array(preproc_hdf5[
                                        f'repeat00_tile000_z{z_plane:03d}_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/dLpN_dRpR_dNpL/F'])
        elif '2023' in preproc_path:
            all_dynamics = np.concatenate(
                [np.array(preproc_hdf5[
                              f'z_plane{z_plane:04d}/cellpose_segmentation/stimulus_aligned_dynamics/stimulus0000/F']),
                 np.array(
                     preproc_hdf5[
                         f'z_plane{z_plane:04d}/cellpose_segmentation/stimulus_aligned_dynamics/stimulus0000/F'][:2, :,
                     :]),
                 np.array(
                     preproc_hdf5[
                         f'z_plane{z_plane:04d}/cellpose_segmentation/stimulus_aligned_dynamics/stimulus0000/F'][:2, :,
                     :])], axis=2)
        else:
            all_dynamics = np.array(preproc_hdf5[
                                        f'repeat00_tile000_z{z_plane:03d}_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/stimulus0000/F'])

        preproc_hdf5.close()
        print(all_dynamics.shape)
        # Select the functional activity of our targeted neuron.
        cell_dynamics = all_dynamics[:, cell_mask_ID, :]

        # calculate and stack the df/f0 of all neurons.
        if first_fish:
            df_f0 = (cell_dynamics - np.nanmean(cell_dynamics[:, :10], axis=1)[:, None]) / np.nanmean(
                cell_dynamics[:, :10], axis=1)[:, None]
            df_f0_avg = np.nanmean(
                (cell_dynamics - np.nanmean(cell_dynamics[:, :10], axis=1)[:, None]) / np.nanmean(cell_dynamics[:, :10],
                                                                                                  axis=1)[:, None],
                axis=0)
            first_fish = False
        else:
            df_f0 = np.vstack([df_f0, (cell_dynamics - np.nanmean(cell_dynamics[:, :10], axis=1)[:, None]) / np.nanmean(
                cell_dynamics[:, :10], axis=1)[:, None]])
            df_f0_avg = np.vstack([df_f0_avg, np.nanmean(
                (cell_dynamics - np.nanmean(cell_dynamics[:, :10], axis=1)[:, None]) / np.nanmean(cell_dynamics[:, :10],
                                                                                                  axis=1)[:, None],
                axis=0)])

    # Plot the single trial traces.
    for trial in range(df_f0_avg.shape[0]):
        subfig.draw_line(np.arange(0, 120), df_f0_avg[trial, :120], lc=plot_color_2, lw=0.5)
        subfig.draw_line(np.arange(130, 250), df_f0_avg[trial, 120:240], lc=plot_color_2, lw=0.5)
        subfig.draw_line(np.arange(260, 380), df_f0_avg[trial, 240:], lc=plot_color_2, lw=0.5)

    # Plot the average response on top.
    subfig.draw_line(np.arange(0, 120), np.nanmean(df_f0_avg[:, :120], axis=0), lc=plot_color, lw=1)
    subfig.draw_line(np.arange(130, 250), np.nanmean(df_f0_avg[:, 120:240], axis=0), lc=plot_color, lw=1)
    subfig.draw_line(np.arange(260, 380), np.nanmean(df_f0_avg[:, 240:], axis=0), lc=plot_color, lw=1)

    # add the scale bars
    subfig.draw_line([-2, -2], [0, 1], lc='k')
    subfig.draw_text(-40, 0.5, '1 dF/F\u2080', textlabel_rotation=90, textlabel_ha='left')
    if timescalebar:
        subfig.draw_line([355, 375], [-0.8, -0.8], lc='k')
        subfig.draw_text(360, -2.2, '10s', textlabel_ha='center')

    # Load the major brain regions used as reference in the location plot
    brain_regions = [navis.read_mesh(fr'{masks_path}\superior_ventral_medulla_oblongata_(entire).obj', units='microns',
                                     output='volume'),
                     navis.read_mesh(fr'{masks_path}\mesencephalon_(midbrain).obj', units='microns', output='volume'),
                     navis.read_mesh(fr'{masks_path}\cerebellum.obj', units='microns', output='volume')]

    # Loop over all cells to plot the morphology
    swc_cells = []
    for cell_folder, file_base_name, swc_id in zip(cell_folders, volume_file_base_names, swc_ids):
        if os.path.exists(
                fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'):
            swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'
        elif os.path.exists(
                fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'):
            swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'
        else:
            print(
                fr'WARNING: {PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc does not exist. Skipping this neuron. ')
            continue
        # Load the neuron, add the soma and smoothen the skeleton.
        neuron = navis.read_swc(swc_file)
        neuron.soma = 1
        neuron.nodes.iloc[0, 5] = 2
        neuron.soma_radius = 4
        swc_cells.append(navis.smooth_skeleton(neuron))

    # Plot all neuron morphologies
    subfig_xy.draw_navis_neuron(swc_cells, brain_regions, navis_view=('x', '-y'), lc=plot_color, lw=0.25,
                                rasterized=True)
    subfig_xy.draw_line([311 * 0.798, 311 * 0.798], [350 * 0.798, 750 * 0.798], lc='w')

    subfig_yz.draw_navis_neuron(swc_cells, brain_regions, navis_view=('z', '-y'), lc=plot_color, lw=0.25,
                                rasterized=True)

    return


def subfig_all_PA_neurons_loc(subfig_xy, subfig_yz, subfigs_region_counts, PA_data_path, all_volume_file_base_names,
                              all_cell_folders, all_swc_ids, all_plot_colors, detailed_brain_regions, masks_path,
                              regions_path, video_path=None):
    '''
    This function plots all the PAed neurons in a single location plot as well as the projections per brain region.
    This relates to Fig. 5d and S8c
    :param subfig_xy: Subfigure showing all PAed neurons in xy view.
    :param subfig_yz: Subfigure showing all PAed neurons in yz view.
    :param subfigs_region_counts: Subfigure showing the count of projections per brain region, split by functional type.
    :param PA_data_path: Path to the folder containing all PA data.
    :param all_volume_file_base_names: List of 6 lists of volume file names of all neurons per functional type.
    :param all_cell_folders: List of 6 lists of cell folders of all neurons per functional type.
    :param all_swc_ids: List of 6 lists of swc IDs of all neurons per functional type.
    :param all_plot_colors: List of 6 plot colors per functional type.
    :param detailed_brain_regions: List of all brain_regions to use for the projections per brain region plot.
    :param masks_path: Path to the obj files of major brain regions.
    :param regions_path: Path to the hdf5 file of all brain regions.
    :param video_path: Optional path where to save a video of the rotating brain with all PAed neurons. If None, the movie making is skipped.
    '''

    # Load the brain regions used as reference plot.
    brain_regions = [navis.read_mesh(fr'{masks_path}\superior_ventral_medulla_oblongata_(entire).obj', units='microns',
                                     output='volume'),
                     navis.read_mesh(fr'{masks_path}\mesencephalon_(midbrain).obj', units='microns', output='volume'),
                     navis.read_mesh(fr'{masks_path}\cerebellum.obj', units='microns', output='volume')]

    # Initialize the nodes_per_region for each of the six functional types to be 0 for each brainregion.
    nodes_per_region = [[]] * 6
    for type in range(6):
        nodes_per_region[type] = np.zeros((len(detailed_brain_regions), len(all_volume_file_base_names[type])))

    # Create the brain regions numpy array mask in which 0 means outside any of the given regions, and all further regions get a unique index.
    region_masks = create_combined_region_npy_mask(regions_path, regions=detailed_brain_regions)
    # The tectum mask was used to find all neurons that leave the tectum and are plotted in Fig. 5g.
    tectum_mask = create_combined_region_npy_mask(regions_path, regions=['tectum'])

    # Loop over all functional types.
    first_set = True
    all_swc_cells = []
    all_colors = []
    for type, (volume_file_base_names, cell_folders, swc_ids, plot_color) in enumerate(
            zip(all_volume_file_base_names, all_cell_folders, all_swc_ids, all_plot_colors)):
        swc_cells = []
        # Loop over all neurons and plot them.
        for neuron_id, (cell_folder, file_base_name, swc_id) in enumerate(
                zip(cell_folders, volume_file_base_names, swc_ids)):
            if os.path.exists(
                    fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'):
                swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'
            elif os.path.exists(
                    fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'):
                swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'
            else:
                print(
                    fr'WARNING: {PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc does not exist. Skipping this neuron. ')
                continue
            # Load a single neuron, add the soma, smoothen the skeleton and add to swc_cells list (with only neurons of this functional type) and all_swc_cells list (with all neurons).
            neuron = navis.read_swc(swc_file)
            neuron.soma = 1
            neuron.nodes.iloc[0, 5] = 2
            neuron.soma_radius = 4
            swc_cells.append(navis.smooth_skeleton(neuron))
            all_swc_cells.append(navis.smooth_skeleton(neuron))
            all_colors.append(plot_color)
            print(
                f'NEURON {type} in {detailed_brain_regions[np.array(region_masks[(neuron.soma_pos[0][0] / 0.798).astype(int),
                (neuron.soma_pos[0][1] / 0.798).astype(int),
                (neuron.soma_pos[0][2] / 2).astype(int)]).astype(int) - 1]}')
            # Create a mask of the neuron and see which brain regions the mask hits.
            mask_neuron = np.zeros((621, 1406, 138))
            mask_neuron[(neuron.nodes['x'] / 0.798).astype(int), (neuron.nodes['y'] / 0.798).astype(int), (
                        neuron.nodes['z'] / 2).astype(int)] = 1
            region_values, region_counts = np.unique_counts(region_masks[mask_neuron.astype(bool)])
            # Our detailed_brain_regions list should cover the entire brain. Still sanity check if any neuron projects somewhere else (this could indicate bad ANTs registration).
            if 0 in region_values:
                print('Warning found nodes outside regions. ')
                neuron_regions = np.array(detailed_brain_regions)[region_values[1:].astype(int) - 1]
                nodes_per_region[type][region_values.astype(int)[1:] - 1, neuron_id] = region_counts[1:]
            else:
                # Get the list of all brainregions this neuron projects to, and how many nodes are found for each region.
                neuron_regions = np.array(detailed_brain_regions)[region_values.astype(int) - 1]
                nodes_per_region[type][region_values.astype(int) - 1, neuron_id] = region_counts
            print(neuron_regions, region_counts)

        # To later know which neurons to plot in Fig. 5g, we print the neurons that leave the tectum.
        outside_tectum = np.logical_not(np.array([np.array(
            tectum_mask[(np.array(swc_cells[i].nodes['x']) / 0.798).astype(int),
            (np.array(swc_cells[i].nodes['y']) / 0.798).astype(int),
            (np.array(swc_cells[i].nodes['z']) / 2).astype(int)]).astype(int).min()
                                                  for i in range(len(swc_cells))]).astype(bool))

        print('Found neurons outside the tectum: ')
        print(cell_folders, outside_tectum)

        # Draw the neurons (only add the reference brain_regions once).
        if first_set:
            subfig_xy.draw_navis_neuron(swc_cells, brain_regions, navis_view=('x', '-y'), lc=plot_color, lw=0.5,
                                        alpha=0.5, rasterized=True)
            subfig_xy.draw_line([311 * 0.798, 311 * 0.798], [350 * 0.798, 750 * 0.798], lc='w')

            subfig_yz.draw_navis_neuron(swc_cells, brain_regions, navis_view=('z', '-y'), lc=plot_color, lw=0.5,
                                        alpha=0.5, rasterized=True)
            first_set = False
        else:
            subfig_xy.draw_navis_neuron(swc_cells, [], navis_view=('x', '-y'), lc=plot_color, lw=0.5, alpha=0.5,
                                        rasterized=True)
            subfig_xy.draw_line([311 * 0.798, 311 * 0.798], [350 * 0.798, 750 * 0.798], lc='w')

            subfig_yz.draw_navis_neuron(swc_cells, [], navis_view=('z', '-y'), lc=plot_color, lw=0.5, alpha=0.5,
                                        rasterized=True)

    # Loop over the 6 functional types, plot the ratio of nodes per region for all regions that had at least one node.
    for type, color, subfig_index in zip(range(6), ['#359B73', '#2271B2', '#D55E00', '#E69F00', '#9F0162', '#F748A5', ],
                                         [1, 0, 4, 5, 3, 2]):
        node_counts = np.array(nodes_per_region[type] / np.nansum(nodes_per_region[type], axis=0)).flatten()
        x_ticks = np.repeat(np.arange(len(detailed_brain_regions)), nodes_per_region[type].shape[1])[::-1]
        subfigs_region_counts[subfig_index].draw_scatter(x_ticks[node_counts > 0], node_counts[node_counts > 0],
                                                         pc=color, ec=None)

    # If a path to store the movie is proivde, create a rotating brain movie with all PAed neurons.
    if video_path is not None:
        # Loop over all cells and flip them in x.
        for i in range(len(all_swc_cells)):
            all_swc_cells[i].nodes['x'] = 621 * 0.798 - all_swc_cells[i].nodes['x']

        # Load the major brain regions.
        brain_regions = [
            navis.read_mesh(fr'{masks_path}\rhombencephalon_(hindbrain).obj', units='microns', output='volume'),
            navis.read_mesh(fr'{masks_path}\mesencephalon_(midbrain).obj', units='microns', output='volume'),
            navis.read_mesh(fr'{masks_path}\prosencephalon_(forebrain).obj', units='microns', output='volume')]

        # Create the plot outlines.
        dpi = 300
        force_new = True
        fig, ax = navis.plot2d(all_swc_cells + brain_regions, linewidth=0.5, method='3d_complex', color=all_colors,
                               figsize=(3, 5), autoscale=True, view=('x', '-y'))
        fig.set_dpi(dpi)
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 1122)
        ax.set_zlim(0, 276)
        ax.set_axis_off()
        ax.set_box_aspect([276, 496, 1122])
        fig.set_layout_engine("none")
        ax.set_position([-0.5, -0.6, 2., 2.])
        ax.set_facecolor("none")

        # Loop over 360 degrees in 2 degree steps, save each frame of the movie.
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
        # Save the final movie.
        imageio.mimsave(f"{video_path}/spinning_brain/PAed_neurons.mp4", frames, fps=30, codec="libx264",
                        output_params=["-crf", "20"])

    return


def subfig_group_of_neurons(anterior_xy_plot, anterior_yz_plot, contralateral_xy_plot, contralateral_yz_plot,
                            local_xy_plot, local_yz_plot,
                            ant_contr_plot, ant_contr_x_loc, PA_data_path, file_base_names, cell_folders, swc_ids, type,
                            plot_color, masks_path):
    '''
    This function plots one functional type split by morphological type (e.g. local, anterior, contralateral projecting).
    It also plots the ratios of the morphological types at x-position ant_contr_x_loc.
    This relates to figure 5f
    :param anterior_xy_plot: Subfigure showing the anterior type in xy view.
    :param anterior_yz_plot: Subfigure showing the anterior type in yz view.
    :param contralateral_xy_plot: Subfigure showing the contrlateral type in xy view.
    :param contralateral_yz_plot: Subfigure showing the contralateral type in yz view.
    :param local_xy_plot: Subfigure showing the local type in xy view.
    :param local_yz_plot: Subfigure showing the local type in yz view.
    :param ant_contr_plot: Subfigure showing the ratio of morphological types.
    :param ant_contr_x_loc: X-location to plot the vertical bar with ratios of the morphological types in ant_contr_plot
    :param PA_data_path: Path to the folder containing all PA data.
    :param file_base_names: List of functional file names of all neurons.
    :param cell_folders: List of cell folders of all neurons.
    :param swc_ids: List of swc IDs of all neurons
    :param type: List of morphological type of all neurons (0-local, 1-contralateral, 2-anterior).
    :param plot_color: Color to use for plotting.
    :param masks_path: Path to the folder containing obj files of brain regions to plot as reference.
    '''

    # Load the obj file brain regions.
    brain_regions = [navis.read_mesh(fr'{masks_path}\superior_ventral_medulla_oblongata_(entire).obj', units='microns',
                                     output='volume'),
                     navis.read_mesh(fr'{masks_path}\mesencephalon_(midbrain).obj', units='microns', output='volume'),
                     navis.read_mesh(fr'{masks_path}\cerebellum.obj', units='microns', output='volume')]

    # Loop over the three morphological types.
    for t, xy_plot, yz_plot in zip(range(3), [local_xy_plot, contralateral_xy_plot, anterior_xy_plot],
                                   [local_yz_plot, contralateral_yz_plot, anterior_yz_plot]):
        # Loop over all cells that belong to the current morphological type t.
        swc_cells = []
        for cell_folder, file_base_name, swc_id in zip(np.array(cell_folders)[np.array(type) == t],
                                                       np.array(file_base_names)[np.array(type) == t],
                                                       np.array(swc_ids)[np.array(type) == t]):
            # Load the single cell data.
            if os.path.exists(
                    fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'):
                swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'
            elif os.path.exists(
                    fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'):
                swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'
            else:
                print(
                    fr'WARNING: {PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc does not exist. Skipping this neuron. ')
                continue

            # Load the neuron, add the soma and smoothen the skeleton.
            neuron = navis.read_swc(swc_file)
            neuron.soma = 1
            neuron.nodes.iloc[0, 5] = 2
            neuron.soma_radius = 4
            swc_cells.append(navis.smooth_skeleton(neuron))

        # Draw all neurons belonging to the current morphological type t.
        xy_plot.draw_navis_neuron(swc_cells, brain_regions, navis_view=('x', '-y'), lc=plot_color, lw=0.5,
                                  rasterized=True)
        yz_plot.draw_navis_neuron(swc_cells, brain_regions, navis_view=('z', '-y'), lc=plot_color, lw=0.5,
                                  rasterized=True)
        # Add the midline.
        xy_plot.draw_line([311 * 0.798, 311 * 0.798], [350 * 0.798, 750 * 0.798], lc='w')

    # Compute the number of neurons for each morphological type.
    n_local = np.sum(np.array(type) == 0)
    n_contra = np.sum(np.array(type) == 1)
    n_anterior = np.sum(np.array(type) == 2)
    n_tot = len(type)
    # the colors to use for the ratio plot are hardcoded.
    if ant_contr_x_loc == 1:
        plot_colors = ['#6E9FC8', '#2271B2', '#C0DFF8']
    else:
        plot_colors = ['#6AB799', '#359B73', '#C7F2E1', ]
    # Draw the ratio plot of the morphological types among the current functional type.
    ant_contr_plot.draw_vertical_bars([ant_contr_x_loc, ant_contr_x_loc, ant_contr_x_loc],
                                      [n_local / n_tot, n_contra / n_tot, n_anterior / n_tot],
                                      vertical_bar_bottom=[0, n_local / n_tot, (n_local + n_contra) / n_tot],
                                      lc=plot_colors)
    if ant_contr_x_loc == 1:
        ant_contr_plot.draw_vertical_bars([4, 4, 4], [1 / 8, 1 / 8, 1 / 8], vertical_bar_bottom=[2 / 8, 4 / 8, 6 / 8],
                                          lc=['#808080', '#414141', '#C0C0C0'])
        ant_contr_plot.draw_text(4.6, 2.5 / 8, 'local', textlabel_ha='left')
        ant_contr_plot.draw_text(4.6, 4.5 / 8, 'contralateral', textlabel_ha='left')
        ant_contr_plot.draw_text(4.6, 6.5 / 8, 'anterior', textlabel_ha='left')
        ant_contr_plot.draw_text(3.7, 1, 'projection type:', textlabel_ha='left')

    return


def subfig_zoomin_neurons(drive_change_neurons_xy_plot, drive_change_neurons_yz_plot, drive_lumi_neurons_xy_plot,
                          drive_lumi_neurons_yz_plot, PA_data_path,
                          volume_file_base_names, cell_folders, swc_ids, cell_types):
    '''
    This function plots two sets of specified zoom-in neurons in xy and yz view.
    This relates to Fig. 5g
    :param drive_change_neurons_xy_plot: Subfigure to plot the luminance change to multifeature integrator potential connection in xy view.
    :param drive_change_neurons_yz_plot: Subfigure to plot the luminance change to multifeature integrator potential connection in yz view.
    :param drive_lumi_neurons_xy_plot: Subfigure to plot the luminance integrator to multifeature integrator potential connection in xy view.
    :param drive_lumi_neurons_yz_plot: Subfigure to plot the luminance integrator to multifeature integrator potential connection in yz view.
    :param PA_data_path: path to the folder containing all fish in which a neuron was PAed.
    :param volume_file_base_names: List of filenames of the post-PA volume of each neuron.
    :param cell_folders: List of cell folders of each neuron.
    :param swc_ids: List of swc-IDs of each neuron
    :param cell_types: List of functional type of each neuron (0=multifeature, 1=luminance change, 2=luminance integrator).
    '''
    # Intialize the lists to store the swc files and plot colors for the neurons in each plot.
    drive_change_swc_cells = []
    drive_lumi_swc_cells = []
    drive_lumi_plot_colors = []
    drive_change_plot_colors = []
    # Loop over all multifeature cell folders, volume file names and swc ids and store the info in both plots.
    for cell_folder, file_base_name, swc_id in zip(np.array(cell_folders)[np.array(cell_types) == 0],
                                                   np.array(volume_file_base_names)[np.array(cell_types) == 0],
                                                   np.array(swc_ids)[np.array(cell_types) == 0]):
        if os.path.exists(
                fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'):
            swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'
        elif os.path.exists(
                fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'):
            swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'
        else:
            print(
                fr'WARNING: {PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc does not exist. Skipping this neuron. ')
            continue
        # Load the neuron, add the soma and smoothen the skeleton.
        neuron = navis.read_swc(swc_file)
        neuron.soma = 1
        neuron.nodes.iloc[0, 5] = 2
        neuron.soma_radius = 4
        drive_change_swc_cells.append(navis.smooth_skeleton(neuron))
        drive_lumi_swc_cells.append(navis.smooth_skeleton(neuron))
        drive_lumi_plot_colors.append('#2271B2')
        drive_change_plot_colors.append('#2271B2')

    # Loop over all lumi change cell folders, volume file names and swc ids and store the info in the change-multifeature plot.
    for cell_folder, file_base_name, swc_id in zip(np.array(cell_folders)[np.array(cell_types) == 1],
                                                   np.array(volume_file_base_names)[np.array(cell_types) == 1],
                                                   np.array(swc_ids)[np.array(cell_types) == 1]):
        if os.path.exists(
                fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'):
            swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'
        elif os.path.exists(
                fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'):
            swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'
        else:
            print(
                fr'WARNING: {PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc does not exist. Skipping this neuron. ')
            continue

        # Load the neuron, add the soma and smoothen the skeleton.
        neuron = navis.read_swc(swc_file)
        neuron.soma = 1
        neuron.nodes.iloc[0, 5] = 2
        neuron.soma_radius = 4
        drive_change_swc_cells.append(navis.smooth_skeleton(neuron))
        drive_change_plot_colors.append('#D55E00')

    # Loop over all lumi integrator cell folders, volume file names and swc ids and store the info in the lumiintegrator-multifeature plot.
    for cell_folder, file_base_name, swc_id in zip(np.array(cell_folders)[np.array(cell_types) == 2],
                                                   np.array(volume_file_base_names)[np.array(cell_types) == 2],
                                                   np.array(swc_ids)[np.array(cell_types) == 2]):
        if os.path.exists(
                fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'):
            swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc'
        elif os.path.exists(
                fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'):
            swc_file = fr'{PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_ZBRAIN.swc'
        else:
            print(
                fr'WARNING: {PA_data_path}\{cell_folder}\{file_base_name}\{file_base_name}-{swc_id:03d}_to_volume_to_ZBRAIN.swc does not exist. Skipping this neuron. ')
            continue

        # Load the neuron, add the soma and smoothen the skeleton.
        neuron = navis.read_swc(swc_file)
        neuron.soma = 1
        neuron.nodes.iloc[0, 5] = 2
        neuron.soma_radius = 4
        drive_lumi_swc_cells.append(navis.smooth_skeleton(neuron))
        drive_lumi_plot_colors.append('#E69F00')

    # Plot the lumi change and multifeature potential connections in xy view
    drive_change_neurons_xy_plot.draw_navis_neuron(drive_change_swc_cells, [], navis_view=('x', '-y'),
                                                   lc=drive_change_plot_colors, lw=0.5, rasterized=True)
    drive_change_neurons_xy_plot.draw_scatter([134, ], [340, ], pt=MarkerStyle('^', 'left', Affine2D().rotate_deg(225)),
                                              pc='k', ec=None, ps=3)
    drive_change_neurons_xy_plot.draw_line([70, 120], [550, 550], lc='k')
    drive_change_neurons_xy_plot.draw_line([70, 70], [500, 550], lc='k')
    drive_change_neurons_xy_plot.draw_text(130, 550, 'm')
    drive_change_neurons_xy_plot.draw_text(70, 490, 'a')

    # Plot the lumi change and multifeature potential connections in yz view
    drive_change_neurons_yz_plot.draw_navis_neuron(drive_change_swc_cells, [], navis_view=('z', '-y'),
                                                   lc=drive_change_plot_colors, lw=0.5, rasterized=True)
    drive_change_neurons_yz_plot.draw_scatter([100, ], [360, ], pt=MarkerStyle('^', 'left', Affine2D().rotate_deg(225)),
                                              pc='k', ec=None, ps=3)
    drive_change_neurons_yz_plot.draw_line([150, 200], [550, 550], lc='k')
    drive_change_neurons_yz_plot.draw_line([200, 200], [500, 550], lc='k')
    drive_change_neurons_yz_plot.draw_text(140, 550, 'v')
    drive_change_neurons_yz_plot.draw_text(200, 490, 'a')

    # Plot the lumi integrator and multifeature potential connections in xy view
    drive_lumi_neurons_xy_plot.draw_navis_neuron(drive_lumi_swc_cells, [], navis_view=('x', '-y'),
                                                 lc=drive_lumi_plot_colors, lw=0.5, rasterized=True)
    drive_lumi_neurons_xy_plot.draw_scatter([143, ], [535, ], pt=MarkerStyle('^', 'left', Affine2D().rotate_deg(-35)),
                                            pc='k', ec=None, ps=3)
    drive_lumi_neurons_xy_plot.draw_line([70, 120], [550, 550], lc='k')
    drive_lumi_neurons_xy_plot.draw_line([70, 70], [500, 550], lc='k')
    drive_lumi_neurons_xy_plot.draw_text(130, 550, 'm')
    drive_lumi_neurons_xy_plot.draw_text(70, 490, 'a')

    # Plot the lumi integrator and multifeature potential connections in yz view
    drive_lumi_neurons_yz_plot.draw_navis_neuron(drive_lumi_swc_cells, [], navis_view=('z', '-y'),
                                                 lc=drive_lumi_plot_colors, lw=0.5, rasterized=True)
    drive_lumi_neurons_yz_plot.draw_scatter([52, ], [535, ], pt=MarkerStyle('^', 'left', Affine2D().rotate_deg(-15)),
                                            pc='k', ec=None, ps=3)
    drive_lumi_neurons_yz_plot.draw_line([150, 200], [550, 550], lc='k')
    drive_lumi_neurons_yz_plot.draw_line([200, 200], [500, 550], lc='k')
    drive_lumi_neurons_yz_plot.draw_text(140, 550, 'v')
    drive_lumi_neurons_yz_plot.draw_text(200, 490, 'a')
    drive_lumi_neurons_yz_plot.draw_text(175, 570, '50 \u00b5m')
    return


def mapzebrain_neuron_analysis(path_to_swc_folder_ahb, path_to_swc_folder_tectum, mapzebrain_nrrd_paths, nrrd_plot,
                               all_neurons_xy, all_neurons_yz, xy_plots, yz_plots, masks_path, regions_path):
    '''
    This function performs the mapzebrain analysis.
    This is related to Fig. 5a-b.
    :param path_to_swc_folder_ahb: Path to the folder which contains all swc-files with mapzebrain neurons that have their cellbody in the anterior hindbrain.
    :param path_to_swc_folder_tectum: Path to the folder which contains all swc-files with mapzebrain neurons that have their cellbody in the tectum.
    :param mapzebrain_nrrd_paths: List of 6 paths to nrrd files containing the KDE based masks for each region.
    :param nrrd_plot: Subfigure to plot the KDE based merge masks for the anterior hindbrain and the tectum (Those are the same ones that were used to find the mapzebrain neurons on the mapzebrain website given above.)
    :param all_neurons_xy: Subfigure to draw all mapzebrain neurons in xy view.
    :param all_neurons_yz: Subfigure to draw all mapzebrain neurons in yz view.
    :param xy_plots: List of 4 Subfigures to plot the xy view of selected mapzebrain neurons.
    :param yz_plots: List of 4 Subfigures to plot the yz view of selected mapzebrain neurons.
    :param masks_path: Path to the folder containing all obj file brain regions.
    :param regions_path: Path to the hdf5 file containing all brain regions.
    '''
    # Create the merged KDE-based mask for the anterior hindbrain. 0-motion, 1-multifeature
    ahb = np.logical_or(nrrd.read(mapzebrain_nrrd_paths[0])[0], nrrd.read(mapzebrain_nrrd_paths[1])[0])
    # Create the merged KDE-based mask for the tectum. 2-lumi, 3-change, 4-increase, 5-decrease
    tectum = np.logical_or(
        np.logical_or(np.logical_or(nrrd.read(mapzebrain_nrrd_paths[2])[0], nrrd.read(mapzebrain_nrrd_paths[3])[0]),
                      nrrd.read(mapzebrain_nrrd_paths[4])[0]), nrrd.read(mapzebrain_nrrd_paths[5])[0])

    # Create the KDE-based masks visualization with the anterior hindbrain mask in gray and the tectal mask in black.
    # We start with a fully white 3-channel array. (The 3 channels are a legacy from when Fig. 5a used to be colorful).
    rgb_xy = np.ones((621, 1406, 3))
    rgb_yz = np.ones((1406, 138, 3))
    # Make the anterior hindbrain mask pixels gray (0.6 in each channel)
    ahb_xy = np.nanmean(ahb, axis=2).astype(bool)
    rgb_xy[ahb_xy == True, 0] = 0.6
    rgb_xy[ahb_xy == True, 1] = 0.6
    rgb_xy[ahb_xy == True, 2] = 0.6
    ahb_yz = np.nanmean(ahb, axis=0).astype(bool)
    rgb_yz[ahb_yz == True, 0] = 0.6
    rgb_yz[ahb_yz == True, 1] = 0.6
    rgb_yz[ahb_yz == True, 2] = 0.6
    # Make the tectal mask pixels black by subtracting 1 (in each channel)
    tectum_xy = np.nanmean(tectum, axis=2).astype(bool)
    rgb_xy[tectum_xy == True, 0] -= 1
    rgb_xy[tectum_xy == True, 1] -= 1
    rgb_xy[tectum_xy == True, 2] -= 1
    tectum_yz = np.nanmean(tectum, axis=0).astype(bool)
    rgb_yz[tectum_yz == True, 0] -= 1
    rgb_yz[tectum_yz == True, 1] -= 1
    rgb_yz[tectum_yz == True, 2] -= 1
    # Now the pixels in which the anterior hindbrain and tectum overlap have value -0.4. We set them to darkgray (0.3 in each channel).
    # Note the aHb and tectum can overlap because of the 2D projection of a 3d brain.
    rgb_xy[rgb_xy == -0.4] = 0.3
    rgb_yz[rgb_yz == -0.4] = 0.3

    # Draw the KDE-based masks
    nrrd_plot.draw_image(np.swapaxes(rgb_xy, 0, 1), image_origin='lower', extent=(0, 621 * 0.798, 0, 1406 * 0.798))
    nrrd_plot.draw_image(rgb_yz, image_origin='lower', extent=(515, 515 + 138 * 2, 0, 1406 * 0.798))
    # Add a red-boxed outline to indicate where the ZBRAIN cartoon outline needs to go (this is added later in Affinity).
    nrrd_plot.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
    nrrd_plot.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')

    # Load the major brain region obj files.
    brain_regions = [navis.read_mesh(fr'{masks_path}\prosencephalon_(forebrain).obj', units='microns', output='volume'),
                     navis.read_mesh(fr'{masks_path}\mesencephalon_(midbrain).obj', units='microns', output='volume'),
                     navis.read_mesh(fr'{masks_path}\rhombencephalon_(hindbrain).obj', units='microns',
                                     output='volume')]

    # Prepare lists for anterior hindbrain mapzebrain cells and soma locations.
    swc_cells = []
    soma_x = []
    soma_y = []
    soma_z = []
    # Loop over all anterior hindbrain neurons and store the somas and smoothened neurites.
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

    # Double check that all the anterior hindbrain neurons truly have their soma in the anterior hindbrain (superior_medulla_oblongata).
    regions = ['superior_medulla_oblongata']

    region_mask = create_combined_region_npy_mask(regions_path, regions=regions)
    region_mask = binary_erosion(region_mask, iterations=3)

    region_ids = region_mask[
        (np.array(soma_x) / 0.798).astype(int), (np.array(soma_y) / 0.789).astype(int), (np.array(soma_z) / 2).astype(
            int)]

    good_neurons = np.logical_and(region_ids == 1, np.array(soma_x) < 311 * 0.798)

    good_swc_cells_ahb = np.array(swc_cells)[good_neurons].tolist()

    # Store the neuron ids of neurons that cross over to the contralateral side (any swc node has x > 311 pixels)
    hemisphere_crossing_neurons = np.array(
        [np.sum(good_swc_cells_ahb[i].nodes['x'] > 311 * 0.798) for i in range(len(good_swc_cells_ahb))]).astype(bool)

    # Store the neuron ids of neurons that project anterior (any swc node has y < 500 pixels)
    anterior = np.array([np.sum(good_swc_cells_ahb[i].nodes['y'] < 500 * 0.798) for i in
                         range(len(good_swc_cells_ahb))]).astype(bool)

    # Store the neuron ids of neurons that project ventral (any swc node has z < 30 pixels)
    ventral = np.array(
        [np.sum(good_swc_cells_ahb[i].nodes['z'] < 30 * 2) for i in range(len(good_swc_cells_ahb))]).astype(
        bool)

    # Store the neuron ids of neurons that project posterior (any swc node has y > 780 pixels)
    posterior = np.array(
        [np.sum(good_swc_cells_ahb[i].nodes['y'] > 780 * 0.798) for i in range(len(good_swc_cells_ahb))]).astype(bool)

    # Store the neuron ids of neurons that project dorsal (any swc node has z > 90 pixels)
    dorsal = np.array(
        [np.sum(good_swc_cells_ahb[i].nodes['z'] > 90 * 2) for i in range(len(good_swc_cells_ahb))]).astype(
        bool)

    # Store the neuron ids of neurons that project lateral (any swc node has x < 400 pixels)
    lateral = np.array(
        [np.sum(good_swc_cells_ahb[i].nodes['x'] > 400 * 0.798) for i in range(len(good_swc_cells_ahb))]).astype(bool)

    # Find the neurons that project anterior on the ipsilateral side (Type 2 in Fig. 5b).
    # E.g.: neurons that do not cross the midline, do project anterior, do not project ventral, and do not project dorsal.
    ant_ipsi = np.logical_and(
        np.logical_and(np.logical_and(np.logical_not(hemisphere_crossing_neurons), anterior), np.logical_not(ventral)),
        np.logical_not(dorsal))
    # Find the neurons that project anterior on the contralateral side (Type 3 in Fig. 5b)
    # E.g.: neurons that do cross the midline, do project anterior, and do project lateral.
    ant_contra = np.logical_and(np.logical_and(hemisphere_crossing_neurons, anterior), lateral)
    # Find the neurons that project local on the ipsilateral side (Type 1 and 4 in Fig. 5b)
    # E.g.: neurons that do not cross the midline, do not project anterior, do not project posterior, and do not project dorsal.
    local_ipsi = np.logical_and(
        np.logical_and(np.logical_and(np.logical_not(hemisphere_crossing_neurons), np.logical_not(anterior)),
                       np.logical_not(posterior)), np.logical_not(dorsal))

    # Prepare lists for tectum mapzebrain cells and soma locations.
    swc_cells = []
    soma_x = []
    soma_y = []
    soma_z = []
    # Loop over all tectum neurons and store the somas and smoothened neurites.
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

    # Double check that all the tectum neurons truly have their soma in the tectum.
    regions = ['tectum']

    region_mask = create_combined_region_npy_mask(regions_path, regions=regions)
    region_mask = binary_erosion(region_mask, iterations=3)

    region_ids = region_mask[
        (np.array(soma_x) / 0.798).astype(int), (np.array(soma_y) / 0.789).astype(int), (np.array(soma_z) / 2).astype(
            int)]

    good_neurons = np.logical_and(region_ids == 1, np.array(soma_x) < 311 * 0.798)

    good_swc_cells_tectum = np.array(swc_cells)[good_neurons].tolist()

    # Draw all the good anterior hindbrain (gray) and tectum cells (black) (Fig. 5a).
    all_neurons_xy.draw_navis_neuron(np.array(good_swc_cells_ahb).tolist(), brain_regions, navis_view=('x', '-y'),
                                     lc='gray', lw=0.25, rasterized=True)
    all_neurons_yz.draw_navis_neuron(np.array(good_swc_cells_ahb).tolist(), brain_regions, navis_view=('z', '-y'),
                                     lc='gray', lw=0.25, rasterized=True)
    all_neurons_xy.draw_navis_neuron(np.array(good_swc_cells_tectum).tolist(), [], navis_view=('x', '-y'), lc='k',
                                     lw=0.25, rasterized=True)
    all_neurons_yz.draw_navis_neuron(np.array(good_swc_cells_tectum).tolist(), [], navis_view=('z', '-y'), lc='k',
                                     lw=0.25, rasterized=True)

    # Find the tectal neurons that leave the tectum (e.g. at least one swc-node is outside the tectum region).
    region_of_interest_cells = np.array([np.array(region_mask[
                                                      (np.array(good_swc_cells_tectum[i].nodes['x']) / 0.798).astype(
                                                          int), (np.array(
                                                          good_swc_cells_tectum[i].nodes['y']) / 0.798).astype(int), (
                                                              np.array(
                                                                  good_swc_cells_tectum[i].nodes['z']) / 2).astype(
                                                          int)]).astype(int).min() for i in
                                         range(len(good_swc_cells_tectum))]).astype(bool)
    outside_region_of_interest_cells = np.logical_not(region_of_interest_cells)

    # Store the neuron ids of neurons that cross over to the contralateral side (any swc node has x > 311 pixels)
    hemisphere_crossing_neurons = np.array(
        [np.sum(good_swc_cells_tectum[i].nodes['x'] > 311 * 0.798) for i in range(len(good_swc_cells_tectum))]).astype(
        bool)

    # Store the neuron IDs of neurons that leave the tectum and cross, respectively, do not cross the midline.
    outside_and_crossing = np.logical_and(outside_region_of_interest_cells, hemisphere_crossing_neurons)
    outside_not_crossing = np.logical_and(outside_region_of_interest_cells, np.logical_not(hemisphere_crossing_neurons))

    # Store the neuron ids of neurons that project anterior (any swc node has y < 400 pixels)
    anterior = np.array([np.sum(good_swc_cells_tectum[i].nodes['y'] < 400 * 0.798) for i in
                         range(len(good_swc_cells_tectum))]).astype(bool)

    # Store the neuron ids of neurons that project ventral (any swc node has z < 50 pixels)
    ventral = np.array([np.sum(good_swc_cells_tectum[i].nodes['z'] < 50 * 2) for i in
                        range(len(good_swc_cells_tectum))]).astype(bool)

    # Store the neuron ids of neurons that project posterior (any swc node has y > 780 pixels)
    posterior = np.array(
        [np.sum(good_swc_cells_tectum[i].nodes['y'] > 780 * 0.798) for i in range(len(good_swc_cells_tectum))]).astype(
        bool)

    # Store the neuron ids of neurons that project dorsal (any swc node has z > 120 pixels)
    dorsal = np.array([np.sum(good_swc_cells_tectum[i].nodes['z'] > 120 * 2) for i in
                       range(len(good_swc_cells_tectum))]).astype(bool)

    # Find the neurons that project outside the tectum but remain ipsilateral and anterior (Type 2 in Fig. 5b)
    # E.g.: neurons that are outside the tectum, do not cross the midline, do project anterior, and do not project posterior.
    front_pathway_neurons = np.logical_and(np.logical_and(outside_not_crossing, anterior), np.logical_not(posterior))

    # Find the neurons that project outside the tectum, remain ipsilateral and project ventral (Type 1 and 3 in Fig. 5b)
    # E.g.: neurons that are outside the tectum, do not cross the midline, do not project anterior, do project ventral, and do not project posterior.
    lateral_pathway_neurons = np.logical_and(
        np.logical_and(np.logical_and(outside_not_crossing, np.logical_not(anterior)), ventral),
        np.logical_not(posterior))

    # Find the neurons that project outside the tectum and cross the midline (Type 4 in Fig. 5b)
    # E.g.: neurons that are outside the tectum and that cross the midline.
    lateral_cross_neurons = np.logical_and(np.logical_and(np.logical_and(outside_and_crossing, ventral), dorsal),
                                           np.logical_not(posterior))

    # Loop over the four subfigures related to Fig. 5b and plot 4 possible connections between the tectum and anterior hindbrain.
    for ahb_neurons, tectum_neurons, xy_plot, yz_plot, name, flip in zip([local_ipsi, ant_ipsi,
                                                                          ant_contra, local_ipsi],
                                                                         [lateral_pathway_neurons,
                                                                          front_pathway_neurons,
                                                                          lateral_pathway_neurons,
                                                                          lateral_cross_neurons, ],
                                                                         xy_plots, yz_plots,
                                                                         [f'ipsilateral posterior', 'ipsilateral anterior',
                                                                          f'contralateral aHb crossing', 'contralateral tectal crossing', ],
                                                                         [False, False, True, True]):
        # Select the subset of neurons fitting the current pathway.
        print(f'Analysing {name}...')
        subset_swc_cells_ahb = np.array(good_swc_cells_ahb)[ahb_neurons].tolist()
        subset_swc_cells_tectum = np.array(good_swc_cells_tectum)[tectum_neurons].tolist()
        # If in the current pathway the tectal neurons project contralaterally and the anterior hindbrain neurons project ipsilaterally, we flip the anteriorhindbrain neurons.
        if flip:
            for i in range(len(subset_swc_cells_ahb)):
                subset_swc_cells_ahb[i].nodes['x'] = 621 * 0.798 - subset_swc_cells_ahb[i].nodes['x']
        xy_plot.draw_navis_neuron(subset_swc_cells_ahb, brain_regions, navis_view=('x', '-y'), lc='gray', lw=0.25,
                                  rasterized=True)
        yz_plot.draw_navis_neuron(subset_swc_cells_ahb, brain_regions, navis_view=('z', '-y'), lc='gray', lw=0.25,
                                  rasterized=True)
        # Flip back the anterior hindbrain flipped neurons for the next pathway.
        if flip:
            for i in range(len(subset_swc_cells_ahb)):
                subset_swc_cells_ahb[i].nodes['x'] = 621 * 0.798 - subset_swc_cells_ahb[i].nodes['x']
        xy_plot.draw_navis_neuron(subset_swc_cells_tectum, [], navis_view=('x', '-y'), lc='k', lw=0.25, rasterized=True)
        yz_plot.draw_navis_neuron(subset_swc_cells_tectum, [], navis_view=('z', '-y'), lc='k', lw=0.25, rasterized=True)

        xy_plot.draw_line([311 * 0.798, 311 * 0.798], [100 * 0.798, 1006 * 0.798], lc='w')
        xy_plot.draw_text(0, -10, name, textlabel_ha='left')

    return


def PA_quality_check_figure(plot_funcs, plot_HDs, success_rate_plot, main_path, cell_folders, file_paths_functional,
                            cell_ids, extra_cell_ids, file_paths_HD, HD_xs, HD_ys, HD_zs, func_sizes, HD_sizes,
                            z_planes_func):
    '''
    This function creates 8 zoomed-in examples of pre- and post photoactivation centered around the target neuron. Besides the target neuron (red outline) also neighbouring cells are shown (cyan) to aid visual matching using landmarks.
    This is related to figure S8a-b.
    :param plot_funcs: list of 8 subfigures that will show the pre-PA target neuron.
    :param plot_HDs: list of 8 subfigures that will show the post-PA target neuron.
    :param success_rate_plot: Subfigure to plot the success_rate of the photoactivations.
    :param main_path: Path to the data folder containing all fish of which we plot the PAed neuron here.
    :param cell_folders: List of 8 cell folder names (those folders should be found in main_path).
    :param file_paths_functional: List of 8 functional data filenames that match the cell_folders.
    :param cell_ids: List of 8 target cell IDs
    :param extra_cell_ids: List of 8 lists of multiple neighbouring cell IDs
    :param file_paths_HD: List of 8 close-up data filenames that match the cell_folders.
    :param HD_xs: List of 8 x locations of the PAed cell in the close_up stack.
    :param HD_ys: List of 8 y locations of the PAed cell in the close_up stack.
    :param HD_zs: List of 8 z locations of the PAed cell in the close_up stack.
    :param func_sizes: List of 8 width/heights of the functional data files.
    :param HD_sizes: List of 8 width/height sof the close-up stacks.
    :param z_planes_func: List of 8 z_planes of the targeted cell in the functional stack.
    '''

    # Draw the successrate as a stacked horizontal bar. The values are hard-coded.
    success_rate_plot.draw_horizontal_bars([63 / 102, ], [1, ],
                                           horizontal_bar_left=[0, ],
                                           lc=['tab:green', ], label='target neuron successfully labeled')
    success_rate_plot.draw_horizontal_bars([12 / 102, ], [1, ],
                                           horizontal_bar_left=[63 / 102, ],
                                           lc=['#606060', ],
                                           label='labeled multiple cells')
    success_rate_plot.draw_horizontal_bars([4 / 102, ], [1, ],
                                           horizontal_bar_left=[(63 + 12) / 102, ],
                                           lc=['#808080', ],
                                           label='targeted incorrect cell')
    success_rate_plot.draw_horizontal_bars([23 / 102], [1],
                                           horizontal_bar_left=[(63 + 12 + 4) / 102],
                                           lc=['#404040'],
                                           label='fluorescence increase too small')
    success_rate_plot.draw_text(0.5, 2, 'success rate')

    # Loop over the 8 example neurons and plot the pre- and post- zoom-in on the targeted neuron.
    for fig_idx, (
    plot_func, plot_HD, cell_folder, file_path_functional, cell_id, extra_cells, file_path_HD, HD_x, HD_y, HD_z,
    func_size, HD_size, z_plane_func) in enumerate(
            zip(plot_funcs, plot_HDs, cell_folders, file_paths_functional, cell_ids, extra_cell_ids, file_paths_HD,
                HD_xs, HD_ys, HD_zs, func_sizes, HD_sizes, z_planes_func)):
        # Add the column titles
        if fig_idx == 0:
            plot_func.draw_text(func_sizes[0], 2.2 * func_size, 'pre photoactivation')
            plot_HD.draw_text(HD_sizes[0], 2.2 * HD_size, 'post photoactivation')

        # Load the pre-PA functional data: the average image, the targeted cell outline, the x/y position of the cell.
        preproc_func_hdf5 = h5py.File(
            f'{main_path}/{cell_folder}/{file_path_functional}/{file_path_functional}_preprocessed_data.h5', "r")
        func_im = np.array(preproc_func_hdf5[
                               f'repeat00_tile000_z{z_plane_func:03d}_950nm/preprocessed_data/fish00/imaging_data_channel0_time_averaged'])
        cell_outline = np.array(preproc_func_hdf5[
                                    f'repeat00_tile000_z{z_plane_func:03d}_950nm/preprocessed_data/fish00/cellpose_segmentation/unit_contours/{10000 + cell_id}'])
        cell_x, cell_y = np.array(preproc_func_hdf5[
                                      f'repeat00_tile000_z{z_plane_func:03d}_950nm/preprocessed_data/fish00/cellpose_segmentation/unit_centroids'])[
                         cell_id, :].astype(int)

        # Load the post-PA close-up data: the average image.
        preproc_hd_hdf5 = h5py.File(f'{main_path}/{cell_folder}/{file_path_HD}/{file_path_HD}_preprocessed_data.h5',
                                    "r")
        hd_im = np.array(preproc_hd_hdf5['average_stack_repeat00_tile000_950nm_channel0'])

        # Draw the pre-PA image centered on the targeted neuron.
        plot_func.draw_image(
            np.clip(func_im[cell_y - func_size:cell_y + func_size, cell_x - func_size:cell_x + func_size],
                    np.nanpercentile(func_im, 1), np.nanpercentile(func_im, 99)),
            colormap='gray', extent=(0, 2 * func_size, 0, 2 * func_size), image_origin='upper')

        # In case the neuron was not imaged in the center of the full stack, we fill in the missing edge with black pixels. Here we calculate the start and end offset of all four edges.
        if HD_y - HD_size < 0:
            HD_y_start = 0
            y_end_offset = 2 * HD_size - np.abs(HD_y - HD_size)
        else:
            HD_y_start = HD_y - HD_size
            y_end_offset = 2 * HD_size

        if HD_x - HD_size < 0:
            HD_x_start = 0
            x_offset = np.abs(HD_x - HD_size)
        else:
            HD_x_start = HD_x - HD_size
            x_offset = 0

        if HD_y + HD_size > 800:
            HD_y_end = 800
            y_offset = HD_y + HD_size - 800
        else:
            HD_y_end = HD_y + HD_size
            y_offset = 0
        if HD_x + HD_size > 800:
            HD_x_end = 800
            x_end_offset = 2 * HD_size - (HD_x + HD_size - 800)
        else:
            HD_x_end = HD_x + HD_size
            x_end_offset = 2 * HD_size

        # Draw a black background behind the post-PA close-up image.
        plot_HD.draw_image(np.zeros((2 * HD_size, 2 * HD_size)), colormap='gray',
                           extent=(0, 2 * HD_size, 0, 2 * HD_size), image_origin='upper')

        # Draw the post-PA close-up image centered on the target neuron.
        plot_HD.draw_image(
            np.clip(hd_im[HD_z, HD_y_start:HD_y_end, HD_x_start:HD_x_end], np.nanpercentile(hd_im[HD_z, :, :], 2),
                    np.nanpercentile(hd_im[HD_z, :, :], 98)), colormap='gray',
            extent=(x_offset, x_end_offset, y_offset, y_end_offset), image_origin='upper')

        # Add the outlines of the extra neighbouring cells as segmented from the pre-PA image. Draw those outlines both in the pre and post-PA image.
        for extra_cell in extra_cells:
            cell_outline_extra = np.array(preproc_func_hdf5[
                                              f'repeat00_tile000_z{z_plane_func:03d}_950nm/preprocessed_data/fish00/cellpose_segmentation/unit_contours/{10000 + extra_cell}'])

            plot_func.draw_line(cell_outline_extra[:, 0] - (cell_x - func_size),
                                2 * func_size - (cell_outline_extra[:, 1] - (cell_y - func_size)), lc='cadetblue',
                                lw=0.5)

            plot_HD.draw_line(2 * HD_size / (2 * func_size) * (cell_outline_extra[:, 0] - (cell_x - func_size)),
                              2 * HD_size / (2 * func_size) * (
                                      2 * func_size - (cell_outline_extra[:, 1] - (cell_y - func_size))),
                              lc='cadetblue', lw=0.5)
        # Add the scale bar.
        if fig_idx == 7:
            if HD_size == 350:
                plot_HD.draw_line([2 * HD_size - 258, 2 * HD_size - 20], [20, 20], lc='white')
            elif HD_size == 175:
                plot_HD.draw_line([2 * HD_size - 129, 2 * HD_size - 10], [10, 10], lc='white')

        # Draw the outline of the targeted cell as segmented from the pre-PA image in both pre- and post-PA images.
        plot_func.draw_line(cell_outline[:, 0] - (cell_x - func_size),
                            2 * func_size - (cell_outline[:, 1] - (cell_y - func_size)), lc='tab:red', lw=0.5)

        plot_HD.draw_line(2 * HD_size / (2 * func_size) * (cell_outline[:, 0] - (cell_x - func_size)),
                          2 * HD_size / (2 * func_size) * (2 * func_size - (cell_outline[:, 1] - (cell_y - func_size))),
                          lc='tab:red', lw=0.5)

    return


if __name__ == '__main__':
    # Provide the path to save the figures.
    fig_save_path = 'C:/users/katja/Desktop/fig_5.pdf'
    supfig_save_path = 'C:/users/katja/Desktop/fig_S8.pdf'

    # Provide the path to the figure_5 folder.
    fig_5_folder_path = r'Z:\Bahl lab member directories\Katja\paper_data\figure_5'

    # Provide the path where to store the supplemental rotating brain movie. If None, the movie is not made.
    video_path_PA = f'C:/users/katja/Desktop/rotating_brain'
    video_path_PA = None

    # Get the path to the mapzebrain analysis folder
    mapzebrain_analysis_path = fr'{fig_5_folder_path}\mapzebrain_analysis'

    # Get the mapzebrain-analysis related specific file paths.
    mapzebrain_nrrd_paths = [f'{mapzebrain_analysis_path}/region_cut_motion left.nrrd',
                             f'{mapzebrain_analysis_path}/region_cut_drive left.nrrd',
                             f'{mapzebrain_analysis_path}/region_cut_lumi left.nrrd',
                             f'{mapzebrain_analysis_path}/region_cut_diff left.nrrd',
                             f'{mapzebrain_analysis_path}/region_cut_bright left.nrrd',
                             f'{mapzebrain_analysis_path}/region_cut_dark left.nrrd']
    path_to_swc_folder_ahb = f'{mapzebrain_analysis_path}/Soma_in_mapzebrain_ahb'
    path_to_swc_folder_tectum = f'{mapzebrain_analysis_path}/Soma_in_mapzebrain_tectum'
    masks_path = f'{mapzebrain_analysis_path}'
    regions_path = rf'{mapzebrain_analysis_path}\all_masks_indexed.hdf5'

    # Get the paths to all data (functional, close-up post PA stack, volume, neuron-swc file) of one example neuron.
    example_func_path = fr'{fig_5_folder_path}\20250128-2\2025-01-28_13-21-48_fish002_setup0_arena0_KS\2025-01-28_13-21-48_fish002_setup0_arena0_KS_preprocessed_data.h5'
    example_HD_path = fr'{fig_5_folder_path}\20250128-2\2025-01-28_14-22-35_fish002_setup0_arena0_KS\2025-01-28_14-22-35_fish002_setup0_arena0_KS_preprocessed_data.h5'
    example_volume_path = fr'{fig_5_folder_path}\20250128-2\2025-01-28_14-58-07_fish002_setup0_arena0_KS\2025-01-28_14-58-07_fish002_setup0_arena0_KS_preprocessed_data.h5'
    example_swc_path = fr'{fig_5_folder_path}\20250128-2\2025-01-28_14-58-07_fish002_setup0_arena0_KS\2025-01-28_14-58-07_fish002_setup0_arena0_KS-000_to_ZBRAIN.swc'
    # Pick the ID of the example neuron.
    example_cell_id = 160

    # Create a list of mapzebrain brain regions for which we check whether the PAed neurons project into or pass through it.
    detailed_brain_regions = ['superior_ventral_medulla_oblongata_(entire)', 'superior_raphe',
                              'interpeduncular_nucleus', 'nucleus_isthmi',
                              'superior_dorsal_medulla_oblongata_stripe_1_(entire)',
                              'superior_dorsal_medulla_oblongata_stripe_2&3',
                              'superior_dorsal_medulla_oblongata_stripe_4',
                              'cerebellum', 'mesencephalon_(midbrain)', 'tegmentum',
                              'stratum_marginale', 'stratum_opticum',
                              'stratum_fibrosum_et_griseum_superficiale', 'sfgs__sgc',
                              'stratum_griseum_centrale', 'stratum_album_centrale', 'sac__spv',
                              'periventricular_layer',
                              'pretectum', 'dorsal_thalamus_proper', 'intermediate_hypothalamus_(entire)',
                              'caudal_hypothalamus',
                              'posterior_tuberculum_(basal_part_of_prethalamus_and_thalamus)']
    # Give shorter region names as figure labels. Make sure this list matches the detailed_brain_regions list.
    short_names_brain_regions = ['other sup. vMO', 'sup. Raphe',
                                 'interpeduncular nuc.', 'nuc. isthmi', 'sup. dMO stripe 1', 'sup. dMO stripe 2&3',
                                 'sup. dMO stripe 4',
                                 'cerebellum', 'other midbrain', 'tegmentum',
                                 'SM', 'SO', 'SFGS', 'SFGS-SGC', 'SGC', 'SAC', 'SAC-SPV',
                                 'periventricular layer', 'pretectum', 'dThalamus', 'int. hypothalamus',
                                 'cau. hypothalamus',
                                 'posterior tuberculum']

    # Give the motion integrator functional data file names, volume file names, folder file names, segmentation IDs, swc-file IDs, z_planes, and projection types (0-local, 1-contralateral, 2-anterior).
    mot_file_base_names = ['2024-01-15_14-32-16_fish000_KS', '2024-01-16_10-23-16_fish000_KS',
                           '2024-01-29_13-28-48_fish000_KS', '2024-01-30_14-29-11_fish002_KS',
                           '2024-02-26_15-33-10_fish002_KS', '2024-02-27_10-10-39_fish000_KS',
                           '2024-04-08_10-57-50_fish001_KS', '2024-04-09_14-45-31_fish002_KS',
                           '2025-04-03_09-40-14_fish000_setup0_arena0_HN',
                           '2025-04-03_09-40-14_fish000_setup0_arena0_HN',
                           '2025-04-03_09-40-14_fish000_setup0_arena0_HN',
                           '2025-04-03_09-40-14_fish000_setup0_arena0_HN']
    mot_volume_file_base_names = ['2024-01-15_15-27-36_fish000_KS', '2024-01-16_11-28-32_fish000_KS',
                                  '2024-01-29_14-27-53_fish000_KS', '2024-01-30_15-15-23_fish002_KS',
                                  '2024-02-26_16-38-23_fish002_KS', '2024-02-27_11-19-48_fish000_KS',
                                  '2024-04-08_12-06-09_fish001_KS', '2024-04-09_15-39-07_fish002_KS',
                                  '2025-04-03_17-38-56_fish000_setup0_arena0_HN',
                                  '2025-04-03_17-38-56_fish000_setup0_arena0_HN',
                                  '2025-04-03_17-38-56_fish000_setup0_arena0_HN',
                                  '2025-04-03_17-38-56_fish000_setup0_arena0_HN']
    mot_cell_folders = ['20240115-1', '20240116-1', '20240129-1', '20240130-3',
                        '20240226-3', '20240227-1', '20240408-2', '20240409-3',
                        '20250403-0', '20250403-0', '20250403-0', '20250403-0']
    mot_cell_mask_IDs = [439, 210, 358, 550, 255, 231, 838, 187, 307, 433, 637, 464]
    mot_swc_ids = [0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 2, 3]
    mot_z_planes = [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0]
    mot_type = [0, 0, 0, 1, 0, 2, 1, 0, 0, 0, 0, 0]

    # Give the multifeature integrator functional data file names, volume file names, folder file names, segmentation IDs, swc-file IDs, z_planes, and projection types (0-local, 1-contralateral, 2-anterior).
    drive_file_base_names = ['2023-12-12_12-14-52', '2024-01-16_14-10-16_fish001_KS', '2024-01-22_13-58-17_fish001_KS',
                             '2024-02-26_11-03-33_fish000_KS',
                             '2024-02-27_14-48-03_fish002_KS', '2024-04-09_14-46-28_fish003_KS',
                             '2025-04-03_09-40-14_fish000_setup0_arena0_HN',
                             '2025-02-17_09-29-26_fish001_setup0_arena0_KS',
                             '2025-02-17_09-29-26_fish001_setup0_arena0_KS',
                             '2025-02-17_09-29-26_fish001_setup0_arena0_KS',
                             '2025-02-17_09-29-26_fish001_setup0_arena0_KS',
                             '2025-02-17_09-29-26_fish001_setup0_arena0_KS']
    drive_volume_file_base_names = ['2023-12-12_14-06-43', '2024-01-16_15-07-59_fish001_KS',
                                    '2024-01-22_15-05-28_fish001_KS', '2024-02-26_12-26-15_fish000_KS',
                                    '2024-02-27_15-44-31_fish002_KS', '2024-04-09_15-40-32_fish003_KS',
                                    '2025-04-03_17-38-56_fish000_setup0_arena0_HN',
                                    '2025-02-17_15-27-37_fish001_setup0_arena0_KS',
                                    '2025-02-17_15-27-37_fish001_setup0_arena0_KS',
                                    '2025-02-17_15-27-37_fish001_setup0_arena0_KS',
                                    '2025-02-17_15-27-37_fish001_setup0_arena0_KS',
                                    '2025-02-17_15-27-37_fish001_setup0_arena0_KS']
    drive_cell_folders = ['20231212-1', '20240116-2', '20240122-2', '20240226-1',
                          '20240227-3', '20240409-4', '20250403-0',
                          '20250217-1', '20250217-1', '20250217-1',
                          '20250217-1', '20250217-1']
    drive_cell_mask_IDs = [87, 215, 617, 558, 134, 458, 361, 498, 517, 561, 530, 585]
    drive_swc_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4]
    drive_z_planes = [0, 0, 0, 0, 0, 0, 3, 0, 3, 1, 0, 2]
    drive_type = [1, 1, 2, 2, 2, 0, 0, 0, 2, 1, 0, 0, ]

    # Give the luminance integrator functional data file names, volume file names, folder file names, segmentation IDs, swc-file IDs, and z_planes.
    lumi_file_base_names = ['2025-01-13_10-46-04_fish000_setup0_arena0_KS',
                            '2025-04-15_09-55-21_fish000_setup0_arena0_HN',
                            '2025-04-15_09-55-21_fish000_setup0_arena0_HN',
                            '2025-04-15_09-55-21_fish000_setup0_arena0_HN',
                            '2025-05-06_08-44-53_fish000_setup0_arena0_HN',
                            '2025-05-06_08-44-53_fish000_setup0_arena0_HN',
                            '2025-05-06_08-44-53_fish000_setup0_arena0_HN',
                            '2025-05-06_08-44-53_fish000_setup0_arena0_HN',
                            '2025-05-06_08-44-53_fish000_setup0_arena0_HN',
                            '2025-05-06_08-44-53_fish000_setup0_arena0_HN']
    lumi_volume_file_base_names = ['2025-01-13_12-00-41_fish000_setup0_arena0_KS',
                                   '2025-04-15_12-22-11_fish000_setup0_arena0_HN',
                                   '2025-04-15_16-53-32_fish000_setup0_arena0_HN',
                                   '2025-04-15_16-53-32_fish000_setup0_arena0_HN',
                                   '2025-05-06_10-44-27_fish000_setup0_arena0_HN',
                                   '2025-05-06_17-48-53_fish000_setup0_arena0_HN',
                                   '2025-05-06_17-48-53_fish000_setup0_arena0_HN',
                                   '2025-05-06_17-48-53_fish000_setup0_arena0_HN',
                                   '2025-05-06_17-48-53_fish000_setup0_arena0_HN',
                                   '2025-05-06_17-48-53_fish000_setup0_arena0_HN']
    lumi_cell_folders = ['20250113-0', '20250415-0', '20250415-0', '20250415-0',
                         '20250506-0', '20250506-0', '20250506-0', '20250506-0',
                         '20250506-0', '20250506-0']
    lumi_cell_mask_IDs = [347, 388, 269, 388, 381, 100, 428, 14, 420, 350]
    lumi_swc_ids = [0, 0, 1, 2, 0, 1, 4, 5, 2, 3]
    lumi_z_planes = [0, 3, 2, 1, 3, 2, 1, 1, 0, 0]

    # Give the luminance change detectors functional data file names, volume file names, folder file names, segmentation IDs, swc-file IDs, and z_planes.
    diff_file_base_names = ['2024-04-09_10-00-45_fish001_KS', '2024-05-24_10-57-08_fish001_KS',
                            '2025-01-27_11-37-08_fish001_setup1_arena0_KS',
                            '2025-01-28_13-21-48_fish002_setup0_arena0_KS',
                            '2025-02-25_09-36-39_fish000_setup0_arena0_HN',
                            '2025-03-04_09-31-15_fish000_setup0_arena0_HN',
                            '2025-03-04_09-31-15_fish000_setup0_arena0_HN',
                            '2025-03-04_09-31-15_fish000_setup0_arena0_HN',
                            '2025-03-04_09-31-15_fish000_setup0_arena0_HN',
                            '2025-03-04_09-31-15_fish000_setup0_arena0_HN',
                            '2025-03-04_09-31-15_fish000_setup0_arena0_HN',
                            '2025-03-04_09-31-15_fish000_setup0_arena0_HN',
                            '2025-03-04_09-31-15_fish000_setup0_arena0_HN',
                            '2025-03-04_09-31-15_fish000_setup0_arena0_HN',
                            '2025-03-18_08-56-10_fish000_setup0_arena0_HN',
                            '2025-03-18_08-56-10_fish000_setup0_arena0_HN',
                            '2025-03-18_08-56-10_fish000_setup0_arena0_HN',
                            '2025-03-18_08-56-10_fish000_setup0_arena0_HN']
    diff_volume_file_base_names = ['2024-04-09_10-56-23_fish001_KS', '2024-05-24_11-37-43_fish001_KS',
                                   '2025-01-27_12-35-14_fish001_setup1_arena0_KS',
                                   '2025-01-28_14-58-07_fish002_setup0_arena0_KS',
                                   '2025-02-25_11-07-34_fish000_setup0_arena0_HN',
                                   '2025-03-04_18-17-30_fish000_setup0_arena0_HN',
                                   '2025-03-04_18-17-30_fish000_setup0_arena0_HN',
                                   '2025-03-04_18-17-30_fish000_setup0_arena0_HN',
                                   '2025-03-04_18-17-30_fish000_setup0_arena0_HN',
                                   '2025-03-04_18-17-30_fish000_setup0_arena0_HN',
                                   '2025-03-04_18-17-30_fish000_setup0_arena0_HN',
                                   '2025-03-04_18-17-30_fish000_setup0_arena0_HN',
                                   '2025-03-04_18-17-30_fish000_setup0_arena0_HN',
                                   '2025-03-04_18-17-30_fish000_setup0_arena0_HN',
                                   '2025-03-18_10-44-23_fish000_setup0_arena0_HN',
                                   '2025-03-18_15-07-25_fish000_setup0_arena0_HN',
                                   '2025-03-18_15-07-25_fish000_setup0_arena0_HN',
                                   '2025-03-18_15-07-25_fish000_setup0_arena0_HN']
    diff_cell_folders = ['20240409-2', '20240524-1', '20250127-1', '20250128-2',
                         '20250225-0', '20250304-0', '20250304-0', '20250304-0',
                         '20250304-0', '20250304-0', '20250304-0', '20250304-0',
                         '20250304-0', '20250304-0', '20250318-0', '20250318-0',
                         '20250318-0', '20250318-0']
    diff_cell_mask_IDs = [219, 162, 417, 160, 358, 295, 109, 323, 119, 138, 174, 230, 245, 273, 351, 445, 486, 345]
    diff_swc_ids = [0, 0, 0, 0, 0, 0, 8, 7, 1, 2, 3, 6, 5, 4, 0, 2, 1, 3]
    diff_z_planes = [0, 0, 0, 0, 3, 3, 0, 0, 1, 1, 1, 3, 3, 3, 3, 0, 0, 1]

    # Give the luminance increase detectors functional data file names, volume file names, folder file names, segmentation IDs, swc-file IDs, and z_planes.
    bright_file_base_names = ['2025-02-16_09-32-03_fish001_setup0_arena0_KS',
                              '2025-04-15_09-55-21_fish000_setup0_arena0_HN',
                              '2025-04-15_09-55-21_fish000_setup0_arena0_HN',
                              '2025-04-15_09-55-21_fish000_setup0_arena0_HN']
    bright_volume_file_base_names = ['2025-02-16_11-05-39_fish001_setup0_arena0_KS',
                                     '2025-04-15_16-53-32_fish000_setup0_arena0_HN',
                                     '2025-04-15_16-53-32_fish000_setup0_arena0_HN',
                                     '2025-04-15_16-53-32_fish000_setup0_arena0_HN']
    bright_cell_folders = ['20250216-1', '20250415-0', '20250415-0', '20250415-0']
    bright_cell_mask_IDs = [332, 49, 390, 275]
    bright_swc_ids = [0, 3, 1, 4]
    bright_z_planes = [0, 1, 1, 0]

    # Give the luminance decrease functional data file names, volume file names, folder file names, segmentation IDs, swc-file IDs, and z_planes.
    dark_file_base_names = ['2025-02-16_09-29-42_fish000_setup1_arena0_KS',
                            '2025-02-16_09-29-42_fish000_setup1_arena0_KS',
                            '2025-02-16_09-29-42_fish000_setup1_arena0_KS',
                            '2025-02-16_09-29-42_fish000_setup1_arena0_KS',
                            '2025-02-16_09-29-42_fish000_setup1_arena0_KS',
                            '2025-02-16_09-29-42_fish000_setup1_arena0_KS',
                            '2025-02-16_09-29-42_fish000_setup1_arena0_KS',
                            '2025-02-16_09-29-42_fish000_setup1_arena0_KS',
                            '2025-02-16_09-29-42_fish000_setup1_arena0_KS']
    dark_volume_file_base_names = ['2025-02-16_15-12-29_fish000_setup1_arena0_KS',
                                   '2025-02-16_15-12-29_fish000_setup1_arena0_KS',
                                   '2025-02-16_15-12-29_fish000_setup1_arena0_KS',
                                   '2025-02-16_15-12-29_fish000_setup1_arena0_KS',
                                   '2025-02-16_15-12-29_fish000_setup1_arena0_KS',
                                   '2025-02-16_15-12-29_fish000_setup1_arena0_KS',
                                   '2025-02-16_15-12-29_fish000_setup1_arena0_KS',
                                   '2025-02-16_15-12-29_fish000_setup1_arena0_KS',
                                   '2025-02-16_15-12-29_fish000_setup1_arena0_KS']
    dark_cell_folders = ['20250216-0', '20250216-0', '20250216-0', '20250216-0',
                         '20250216-0', '20250216-0', '20250216-0', '20250216-0',
                         '20250216-0', ]
    dark_cell_mask_IDs = [199, 217, 385, 183, 168, 348, 341]
    dark_swc_ids = [0, 6, 5, 4, 3, 1, 2]
    dark_z_planes = [0, 2, 2, 1, 1, 0, 0]

    # Combine the functional file names, data folders, swc-ids and colors for each functional type.
    all_volume_file_base_names = [mot_volume_file_base_names, drive_volume_file_base_names, diff_volume_file_base_names,
                                  lumi_volume_file_base_names, dark_volume_file_base_names,
                                  bright_volume_file_base_names, ]
    all_cell_folders = [mot_cell_folders, drive_cell_folders, diff_cell_folders, lumi_cell_folders, dark_cell_folders,
                        bright_cell_folders, ]
    all_swc_ids = [mot_swc_ids, drive_swc_ids, diff_swc_ids, lumi_swc_ids, dark_swc_ids, bright_swc_ids, ]
    all_plot_colors = ['#359B73', '#2271B2', '#D55E00', '#E69F00', '#9F0162', '#F748A5', ]

    # Give the highlighted multifeature/lumi integrator/lumi change neurons (Fig. 5g) volume file names, folder file names, swc-file IDs, and zoom-in types (0-multifeature, 1-lumi change, 2-lumi integrator).
    zoomin_volume_file_base_names = ['2024-01-22_15-05-28_fish001_KS', '2024-02-26_12-26-15_fish000_KS',
                                     '2024-02-27_15-44-31_fish002_KS',
                                     '2024-04-09_10-56-23_fish001_KS', '2025-01-28_14-58-07_fish002_setup0_arena0_KS',
                                     '2025-02-25_11-07-34_fish000_setup0_arena0_HN',
                                     '2025-03-04_18-17-30_fish000_setup0_arena0_HN',
                                     '2025-03-04_18-17-30_fish000_setup0_arena0_HN',
                                     '2025-03-04_18-17-30_fish000_setup0_arena0_HN',
                                     '2025-04-15_12-22-11_fish000_setup0_arena0_HN',
                                     '2025-04-15_16-53-32_fish000_setup0_arena0_HN',
                                     '2025-05-06_17-48-53_fish000_setup0_arena0_HN',
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

    # Give 8 example PAed neurons for the quality control figure S7. Get the data folders, functional file names, close-up file names, segmentation IDs, z_planes in functional, neighbouring segmentation IDs, z_planes in close_up, x location in close_up, y location in close_up, width/height of functional data, width/height of close_up data.
    PA_quality_cell_folders = ['20240116-2', '20240122-2', '20250304-0', '20250128-2',
                               '20250216-1', '20240129-1', '20240524-1', '20240227-1']
    PA_quality_file_paths_functional = ['2024-01-16_14-10-16_fish001_KS', '2024-01-22_13-58-17_fish001_KS',
                                        '2025-03-04_09-31-15_fish000_setup0_arena0_HN',
                                        '2025-01-28_13-21-48_fish002_setup0_arena0_KS',
                                        '2025-02-16_09-32-03_fish001_setup0_arena0_KS',
                                        '2024-01-29_13-28-48_fish000_KS', '2024-05-24_10-57-08_fish001_KS',
                                        '2024-02-27_10-10-39_fish000_KS']
    PA_quality_file_paths_HD = ['2024-01-16_14-42-21_fish001_KS', '2024-01-22_14-47-43_fish000_KS',
                                '2025-03-04_10-51-27_fish000_setup0_arena0_HN',
                                '2025-01-28_14-22-35_fish002_setup0_arena0_KS',
                                '2025-02-16_10-41-02_fish001_setup0_arena0_KS', '2024-01-29_13-50-23_fish000_KS',
                                '2024-05-24_11-18-50_fish001_KS', '2024-02-27_10-37-40_fish000_KS']
    PA_quality_cell_ids = [215, 617, 295, 160, 332, 358, 162, 231]
    PA_quality_z_planes_func = [0, 0, 3, 0, 0, 0, 0, 0]
    PA_quality_extra_cell_ids = [[228, 234, 183, 184, 232, 212, 274, 245, 273],
                                 [512, 732, 560, 540, 644, 636, 568, 729, 643],
                                 [275, 297, 342, 343, 304, 249, 353, 254, 293],
                                 [145, 146, 163, 173, 129, 127, 204, 208, 140],
                                 [322, 373, 356, 316, 345, 358, 328, 377, 359],
                                 [370, 304, 301, 268, 384, 403, 442, 348, 318],
                                 [168, 175, 160, 171, 177, 146, 187, 169],
                                 [209, 215, 244, 281, 220, 212, 236, 262, 234]]
    PA_quality_HD_zs = [16, 19, 26, 21, 24, 44, 36, 32, ]
    PA_quality_HD_xs = [421, 418, 484, 440, 456, 416, 225, 488, ]
    PA_quality_HD_ys = [358, 364, 369, 280, 397, 255, 409, 308]
    PA_quality_func_sizes = [100, 70, 70, 70, 70, 70, 70, 70]
    PA_quality_HD_sizes = [350, 350, 175, 175, 175, 350, 350, 350]

    # Prepare the figures for Figure 5 and S8.
    fig = Figure(fig_width=18, fig_height=14.5, dpi=900)
    supfig = Figure(fig_width=18, fig_height=17)

    # Fig. S5a
    nrrd_plot = supfig.create_plot(xpos=0.7, ypos=14.25, plot_height=2.3, plot_width=2.3, axis_off=True, xmin=30, xmax=800,
                                ymin=850, ymax=80)
    all_neurons_xy = supfig.create_plot(xpos=0.1, ypos=11.75, plot_height=2.3, plot_width=2.3 * 1.5525, axis_off=True)
    all_neurons_yz = supfig.create_plot(xpos=1.9, ypos=11.75, plot_height=2.3, plot_width=2.3 / 1.1565, axis_off=True)

    # Fig. S5b
    xy_plots = [[]] * 4
    yz_plots = [[]] * 4
    for i in range(4):
        xy_plots[i] = supfig.create_plot(xpos=0.1, ypos=9.25 - 2.5 * i, plot_height=2.3, plot_width=2.3 * 1.5525,
                                      axis_off=True)
        yz_plots[i] = supfig.create_plot(xpos=1.9, ypos=9.25 - 2.5 * i, plot_height=2.3, plot_width=2.3 / 1.1565,
                                      axis_off=True)

    # Fig. 5a
    example_loc_plot_zoomout = fig.create_plot(xpos=3.5, ypos=11, plot_height=2., plot_width=2., axis_off=True,
                                               xmin=0, xmax=800, ymin=800, ymax=0, )
    example_loc_plot_zoominpre = fig.create_plot(xpos=5.75, ypos=11.55, plot_height=1.45, plot_width=1.45,
                                                 axis_off=True,
                                                 xmin=539 - 220 / 2, xmax=539 + 220 / 2, ymin=355 + 220 / 2,
                                                 ymax=355 - 220 / 2, )
    example_loc_plot_zoominpost = fig.create_plot(xpos=5.75, ypos=10, plot_height=1.45, plot_width=1.45, axis_off=True,
                                                  xmin=437 - 548 / 2, xmax=437 + 548 / 2, ymin=274 + 548 / 2,
                                                  ymax=274 - 548 / 2, )
    example_traces_plot = fig.create_plot(xpos=3.5, ypos=10, plot_height=0.5, plot_width=2,
                                          xmin=-5, xmax=385, ymin=-1, ymax=5,
                                          vspans=[[20, 80, 'lightgray', 1.0], [150, 210, 'lightgray', 1.0],
                                                  [280, 340, 'lightgray', 1.0]])
    example_tracings_plots = [[]] * 3
    for i, (x, y) in enumerate(zip([338, 290, 254], [465, 415, 364])):  # 243, 264
        example_tracings_plots[i] = fig.create_plot(xpos=7.4, ypos=12.1 - i * 1.05, plot_height=0.9, plot_width=0.9,
                                                    axis_off=True,
                                                    xmin=x - 20, xmax=x + 20, ymin=y + 20, ymax=y - 20)
    example_neuron_xy_plot = fig.create_plot(xpos=9.1, ypos=11.5, plot_height=1.5, plot_width=1.5 * 1.5525,
                                             axis_off=True)
    example_neuron_yz_plot = fig.create_plot(xpos=11.2, ypos=11.5, plot_height=1.5, plot_width=1.5 / 1.1565,
                                             axis_off=True)
    example_brain_xy_overview_plot = fig.create_plot(xpos=10.8, ypos=9.6, plot_height=2, plot_width=2 / 2.274,
                                                     axis_off=True)
    example_brain_yz_overview_plot = fig.create_plot(xpos=11.8, ypos=9.6, plot_height=2, plot_width=2 / 4.395,
                                                     axis_off=True)

    # Fig. 5b
    all_neurons_xy_plot = fig.create_plot(xpos=12.5, ypos=10, plot_height=2.5, plot_width=2.5 * 1.5525, axis_off=True)
    all_neurons_yz_plot = fig.create_plot(xpos=15.75, ypos=10, plot_height=2.5, plot_width=2.5 / 1.1565, axis_off=True)

    # Fig. 5c
    mot_functional_plot = fig.create_plot(xpos=0.5, ypos=6.5, plot_height=1, plot_width=3,
                                          xmin=-5, xmax=385, ymin=-1, ymax=12,
                                          vspans=[[20, 80, 'lightgray', 1.0], [150, 210, 'lightgray', 1.0],
                                                  [280, 340, 'lightgray', 1.0]])
    drive_functional_plot = fig.create_plot(xpos=0.5, ypos=8, plot_height=1, plot_width=3,
                                            xmin=-5, xmax=385, ymin=-1, ymax=6,
                                            vspans=[[20, 80, 'lightgray', 1.0], [150, 210, 'lightgray', 1.0],
                                                    [280, 340, 'lightgray', 1.0]])
    lumi_functional_plot = fig.create_plot(xpos=0.5, ypos=0.5, plot_height=1, plot_width=3,
                                           xmin=-5, xmax=385, ymin=-1, ymax=4,
                                           vspans=[[20, 80, 'lightgray', 1.0], [150, 210, 'lightgray', 1.0],
                                                   [280, 340, 'lightgray', 1.0]])
    diff_functional_plot = fig.create_plot(xpos=0.5, ypos=2, plot_height=1, plot_width=3,
                                           xmin=-5, xmax=385, ymin=-1, ymax=4,
                                           vspans=[[20, 80, 'lightgray', 1.0], [150, 210, 'lightgray', 1.0],
                                                   [280, 340, 'lightgray', 1.0]])
    bright_functional_plot = fig.create_plot(xpos=0.5, ypos=5, plot_height=1, plot_width=3,
                                             xmin=-5, xmax=385, ymin=-1, ymax=3,
                                             vspans=[[20, 80, 'lightgray', 1.0], [150, 210, 'lightgray', 1.0],
                                                     [280, 340, 'lightgray', 1.0]])
    dark_functional_plot = fig.create_plot(xpos=0.5, ypos=3.5, plot_height=1, plot_width=3,
                                           xmin=-5, xmax=385, ymin=-1, ymax=6,
                                           vspans=[[20, 80, 'lightgray', 1.0], [150, 210, 'lightgray', 1.0],
                                                   [280, 340, 'lightgray', 1.0]])
    mot_xy_plot = fig.create_plot(xpos=3.75, ypos=6.5, plot_height=1., plot_width=1.5525, axis_off=True)
    mot_yz_plot = fig.create_plot(xpos=5.25, ypos=6.5, plot_height=1., plot_width=1 / 1.1565, axis_off=True)
    drive_xy_plot = fig.create_plot(xpos=3.75, ypos=8, plot_height=1., plot_width=1.5525, axis_off=True)
    drive_yz_plot = fig.create_plot(xpos=5.25, ypos=8, plot_height=1., plot_width=1 / 1.1565, axis_off=True)
    lumi_xy_plot = fig.create_plot(xpos=3.75, ypos=0.5, plot_height=1., plot_width=1.5525, axis_off=True)
    lumi_yz_plot = fig.create_plot(xpos=5.25, ypos=0.5, plot_height=1., plot_width=1 / 1.1565, axis_off=True)
    diff_xy_plot = fig.create_plot(xpos=3.75, ypos=2, plot_height=1., plot_width=1.5525, axis_off=True)
    diff_yz_plot = fig.create_plot(xpos=5.25, ypos=2, plot_height=1., plot_width=1 / 1.1565, axis_off=True)
    bright_xy_plot = fig.create_plot(xpos=3.75, ypos=5, plot_height=1., plot_width=1.5525, axis_off=True)
    bright_yz_plot = fig.create_plot(xpos=5.25, ypos=5, plot_height=1., plot_width=1 / 1.1565, axis_off=True)
    dark_xy_plot = fig.create_plot(xpos=3.75, ypos=3.5, plot_height=1., plot_width=1.5525, axis_off=True)
    dark_yz_plot = fig.create_plot(xpos=5.25, ypos=3.5, plot_height=1., plot_width=1 / 1.1565, axis_off=True)

    # Fig. S8e
    brain_region_plots = [[]] * 6
    for type in range(6):
        if type == 0:
            brain_region_plots[type] = supfig.create_plot(xpos=9.75, ypos=14 - type * 2, plot_height=1.5, plot_width=4,
                                                          yticks=[0.0, 0.5, 1.0],
                                                          yl='nodes in region (ratio per neuron)', xmin=-1,
                                                          xmax=len(detailed_brain_regions), ymin=-0.05, ymax=1.05,
                                                          vspans=[[4.5, 14.5, 'lightgray', 1.0], ])
        elif type == 5:
            brain_region_plots[type] = supfig.create_plot(xpos=9.75, ypos=14 - type * 2, plot_height=1.5,
                                                          plot_width=4, xticks=np.arange(len(detailed_brain_regions)),
                                                          xticklabels=short_names_brain_regions[::-1],
                                                          yticks=[0.0, 0.5, 1.0], xticklabels_rotation=90,
                                                          xmin=-1, xmax=len(detailed_brain_regions), ymin=-0.05,
                                                          ymax=1.05,
                                                          vspans=[[4.5, 14.5, 'lightgray', 1.0], ])

        else:
            brain_region_plots[type] = supfig.create_plot(xpos=9.75, ypos=14 - type * 2, plot_height=1.5, plot_width=4,
                                                          yticks=[0.0, 0.5, 1.0],
                                                          xmin=-1, xmax=len(detailed_brain_regions), ymin=-0.05,
                                                          ymax=1.05,
                                                          vspans=[[4.5, 14.5, 'lightgray', 1.0], ])

    # Fig. 5d
    anterior_vs_contra_count_plot = fig.create_plot(xpos=6.5, ypos=6, plot_height=3, plot_width=1.25, axis_off=True,
                                                    xmin=0, xmax=5, ymin=0, ymax=1, yl='proportion neurons')
    anterior_xy_plot = fig.create_plot(xpos=8.8, ypos=7.5, plot_height=1.5, plot_width=1.5 * 1.5525, axis_off=True)
    anterior_yz_plot = fig.create_plot(xpos=10.8, ypos=7.5, plot_height=1.5, plot_width=1.5 / 1.1565, axis_off=True)
    contralateral_xy_plot = fig.create_plot(xpos=11.8, ypos=7.5, plot_height=1.5, plot_width=1.5 * 1.5525,
                                            axis_off=True)
    contralateral_yz_plot = fig.create_plot(xpos=13.8, ypos=7.5, plot_height=1.5, plot_width=1.5 / 1.1565,
                                            axis_off=True)
    local_xy_plot = fig.create_plot(xpos=14.8, ypos=7.5, plot_height=1.5, plot_width=1.5 * 1.5525, axis_off=True)
    local_yz_plot = fig.create_plot(xpos=16.8, ypos=7.5, plot_height=1.5, plot_width=1.5 / 1.1565, axis_off=True)
    anterior_xy_mot_plot = fig.create_plot(xpos=8.8, ypos=6, plot_height=1.5, plot_width=1.5 * 1.5525, axis_off=True)
    anterior_yz_mot_plot = fig.create_plot(xpos=10.8, ypos=6, plot_height=1.5, plot_width=1.5 / 1.1565, axis_off=True)
    contralateral_xy_mot_plot = fig.create_plot(xpos=11.8, ypos=6, plot_height=1.5, plot_width=1.5 * 1.5525,
                                                axis_off=True)
    contralateral_yz_mot_plot = fig.create_plot(xpos=13.8, ypos=6, plot_height=1.5, plot_width=1.5 / 1.1565,
                                                axis_off=True)
    local_xy_mot_plot = fig.create_plot(xpos=14.8, ypos=6, plot_height=1.5, plot_width=1.5 * 1.5525, axis_off=True)
    local_yz_mot_plot = fig.create_plot(xpos=16.8, ypos=6, plot_height=1.5, plot_width=1.5 / 1.1565, axis_off=True)

    # Fig. 5e
    drive_change_neurons_xy_plot = fig.create_plot(xpos=6.4, ypos=3.5, plot_height=2., plot_width=2, axis_off=True)
    drive_change_neurons_yz_plot = fig.create_plot(xpos=8.5, ypos=3.5, plot_height=2, plot_width=2, axis_off=True)
    drive_lumi_neurons_xy_plot = fig.create_plot(xpos=6.4, ypos=1.5, plot_height=2., plot_width=2, axis_off=True)
    drive_lumi_neurons_yz_plot = fig.create_plot(xpos=8.5, ypos=1.5, plot_height=2, plot_width=2, axis_off=True)

    # Fig. S8c
    plot_funcs = []
    plot_HDs = []
    for fig_idx, (func_size, HD_size) in enumerate(zip(PA_quality_func_sizes, PA_quality_HD_sizes)):
        plot_func = supfig.create_plot(xpos=4.5, ypos=14.5 - fig_idx * 1.8, plot_height=1.6, plot_width=1.6,
                                    xmin=0, xmax=2 * func_size, ymin=0., ymax=2 * func_size)
        plot_HD = supfig.create_plot(xpos=6.5, ypos=14.5 - fig_idx * 1.8, plot_height=1.6, plot_width=1.6,
                                  xmin=0, xmax=2 * HD_size, ymin=0., ymax=2 * HD_size)
        plot_funcs = np.append(plot_funcs, plot_func)
        plot_HDs = np.append(plot_HDs, plot_HD)

    # Fig. S8d
    success_rate_plot = supfig.create_plot(xpos=4.5, ypos=0.5, ymin=0, ymax=2, xmin=0, xmax=1, plot_width=3.6, plot_height=1,
                                        xticks=[0, 0.25, 0.5, 0.75, 1], xticklabels=['0', '25', '50', '75', '100'],
                                        xl='amount (%)', legend_xpos=5., legend_ypos=1.5)

    # Perform the mapzebrain analysis (Fig. S5a-b)
    mapzebrain_neuron_analysis(path_to_swc_folder_ahb, path_to_swc_folder_tectum, mapzebrain_nrrd_paths, nrrd_plot,
                               all_neurons_xy, all_neurons_yz, xy_plots, yz_plots, masks_path, regions_path)

    # Create the method outline plot with one example neuron (Fig. 5a).
    sub_plot_method_outline_PA(example_func_path, example_HD_path, example_volume_path, example_swc_path,
                               example_loc_plot_zoomout, example_loc_plot_zoominpre, example_loc_plot_zoominpost,
                               example_traces_plot, example_tracings_plots, example_neuron_xy_plot,
                               example_neuron_yz_plot,
                               example_brain_xy_overview_plot, example_brain_yz_overview_plot,
                               cell_id=example_cell_id, xs=[338, 290, 254, 243], ys=[465, 415, 364, 264],
                               zs=[28, 26, 25, 23],
                               masks_path=masks_path)  # xs, ys, zs are the centers of the zoomed-in following of the PAed neuron under ' volumetric imaging' in Fig. 5c.

    # Plot all the PAed neurons (Fig. 5b)
    subfig_all_PA_neurons_loc(all_neurons_xy_plot, all_neurons_yz_plot, brain_region_plots, fig_5_folder_path,
                              all_volume_file_base_names, all_cell_folders, all_swc_ids, all_plot_colors,
                              detailed_brain_regions, masks_path, regions_path, video_path=video_path_PA)

    # Plot the functional traces and location per functional type (Fig. 5c)
    subfig_PA_functional_loc(mot_functional_plot, mot_xy_plot, mot_yz_plot, fig_5_folder_path, mot_file_base_names,
                             mot_volume_file_base_names, mot_cell_folders, mot_cell_mask_IDs, mot_swc_ids, masks_path,
                             z_planes=mot_z_planes, plot_color='#359B73', plot_color_2='#8DCDB4')
    subfig_PA_functional_loc(drive_functional_plot, drive_xy_plot, drive_yz_plot, fig_5_folder_path,
                             drive_file_base_names, drive_volume_file_base_names, drive_cell_folders,
                             drive_cell_mask_IDs, drive_swc_ids, masks_path, z_planes=drive_z_planes,
                             plot_color='#2271B2', plot_color_2='#93BADA')
    subfig_PA_functional_loc(lumi_functional_plot, lumi_xy_plot, lumi_yz_plot, fig_5_folder_path, lumi_file_base_names,
                             lumi_volume_file_base_names, lumi_cell_folders, lumi_cell_mask_IDs, lumi_swc_ids,
                             masks_path, z_planes=lumi_z_planes, plot_color='#E69F00', plot_color_2='#F7D280')
    subfig_PA_functional_loc(diff_functional_plot, diff_xy_plot, diff_yz_plot, fig_5_folder_path, diff_file_base_names,
                             diff_volume_file_base_names, diff_cell_folders, diff_cell_mask_IDs, diff_swc_ids,
                             masks_path, z_planes=diff_z_planes, plot_color='#D55E00', plot_color_2='#EEAE7C')
    subfig_PA_functional_loc(bright_functional_plot, bright_xy_plot, bright_yz_plot, fig_5_folder_path,
                             bright_file_base_names, bright_volume_file_base_names, bright_cell_folders,
                             bright_cell_mask_IDs, bright_swc_ids, masks_path, z_planes=bright_z_planes,
                             plot_color='#F748A5', plot_color_2='#F7A4D0')
    subfig_PA_functional_loc(dark_functional_plot, dark_xy_plot, dark_yz_plot, fig_5_folder_path, dark_file_base_names,
                             dark_volume_file_base_names, dark_cell_folders, dark_cell_mask_IDs, dark_swc_ids,
                             masks_path, z_planes=dark_z_planes, plot_color='#9F0162', plot_color_2='#CC7CAD',
                             timescalebar=True)

    # Plot the morphological types for the multifeature and motion neurons (Fig. 5d)
    subfig_group_of_neurons(anterior_xy_mot_plot, anterior_yz_mot_plot, contralateral_xy_mot_plot,
                            contralateral_yz_mot_plot, local_xy_mot_plot, local_yz_mot_plot,
                            anterior_vs_contra_count_plot, 2, fig_5_folder_path, mot_volume_file_base_names,
                            mot_cell_folders, mot_swc_ids, mot_type, '#359B73', masks_path)
    subfig_group_of_neurons(anterior_xy_plot, anterior_yz_plot, contralateral_xy_plot, contralateral_yz_plot,
                            local_xy_plot, local_yz_plot,
                            anterior_vs_contra_count_plot, 1, fig_5_folder_path, drive_volume_file_base_names,
                            drive_cell_folders, drive_swc_ids, drive_type, '#2271B2', masks_path)

    # Plot the zoom-in lumi-change, lumi-integrator and multifeature neurons that might connect (Fig. 5e).
    subfig_zoomin_neurons(drive_change_neurons_xy_plot, drive_change_neurons_yz_plot, drive_lumi_neurons_xy_plot,
                          drive_lumi_neurons_yz_plot, fig_5_folder_path,
                          zoomin_volume_file_base_names, zoomin_cell_folders, zoomin_swc_ids, zoomin_types)

    # Plot the 8 example neurons for mapping quality checks (Fig. S8c-d)
    PA_quality_check_figure(plot_funcs, plot_HDs, success_rate_plot, fig_5_folder_path, PA_quality_cell_folders,
                            PA_quality_file_paths_functional, PA_quality_cell_ids, PA_quality_extra_cell_ids,
                            PA_quality_file_paths_HD, PA_quality_HD_xs, PA_quality_HD_ys, PA_quality_HD_zs,
                            PA_quality_func_sizes, PA_quality_HD_sizes, PA_quality_z_planes_func)

    fig.save(fig_save_path)
    supfig.save(supfig_save_path)

    # Note that Figure 5h only contain explanatory cartoons without actual data and therefore is not part of this code.