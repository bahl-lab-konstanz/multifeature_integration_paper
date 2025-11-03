import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
import navis
import math
from multifeature_integration_paper.figure_helper import Figure
from scipy.stats import ttest_ind
from multifeature_integration_paper.useful_small_funcs import cohens_d, create_combined_region_npy_mask

def subplot_traces_overview(traces_df, subfigs, regions_path, regions):
    '''
    This function plots the overview of all functional activity used for the PCA analysis.
    This is related to Figure S5a.
    :param traces_df: Dataframe containing the single trial activity of all neurons.
    :param subfigs: List of 9 subfigures to plot the functional activity for each stimulus.
    :param regions_path: Path to the hdf5 file containing all mapzebrain regions.
    :param regions: List with the major brain regions used for the per-region PCA analysis (we here sort the functional activity by region for visualization).
    '''

    # Check which brain region each neuron is part of.
    region_masks = create_combined_region_npy_mask(regions_path, regions=regions)
    region_ids = region_masks[
        traces_df['ZB_x'].astype(int), traces_df['ZB_y'].astype(int), traces_df['ZB_z'].astype(int)]

    # Find the total amount of neurons to set the extent of the activity images.
    n_neurons = np.sum(region_ids > 0)

    # Loop over the 9 stimuli.
    for stim, subfig in zip(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off',
                             'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                             'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'],
                            subfigs):
        # Loop over the 3 trials. They will be plotted with a slight x and y offset to mimmick a stack of activity images.
        for t, x_offset, y_offset in zip(range(3), [0, 25, 50], [0, 25000, 50000]):
            region_offset = 0
            # Plot the trial activity for all regions.
            for region_id in range(len(regions) + 1):
                # Skip region_id 0 as this belong to all neurons outside any of the specified regions.
                if region_id == 0:
                    continue
                # Select the region specific data and load the traces of the current trial.
                region_df = traces_df[region_ids == region_id]
                trial = [region_df[f'{stim}_trial_{t}_trace_{i}'].astype(float) for i in range(120)]
                # Normalize the trial activity.
                trial = (trial - np.nanmin(trial, axis=0)) / (np.nanmax(trial, axis=0) - np.nanmin(trial, axis=0))
                # Plot the trial activity for all neurons as an image.
                subfig.draw_image(np.array(trial).reshape(120, len(region_df)).T, colormap='viridis',
                                  extent=(
                                  (50 - x_offset), 120 + (50 - x_offset), len(region_df) + y_offset + region_offset,
                                  y_offset + region_offset), image_origin='upper', rasterized=True)
                region_offset += len(region_df)
            region_offset = 0
            # Add a white border around each region
            for region_id in range(len(regions) + 1):
                if region_id == 0:
                    continue
                region_df = traces_df[region_ids == region_id]
                subfig.draw_line([50 - x_offset, 120 + (50 - x_offset)],
                                 [len(region_df) + y_offset + region_offset, len(region_df) + y_offset + region_offset],
                                 lc='w', lw=0.75)
                region_offset += len(region_df)
            # Add a white border around the activity of all regions of this trial
            subfig.draw_line(
                [(50 - x_offset), 120 + (50 - x_offset), 120 + (50 - x_offset), (50 - x_offset), (50 - x_offset)],
                [n_neurons + y_offset, n_neurons + y_offset, y_offset, y_offset, n_neurons + y_offset], lc='w', lw=0.5)

    subfigs[-1].draw_line([50 + 95, 50 + 115], [135000, 135000], lc='k')
    subfigs[-1].draw_text(50 + 105, 150000, '10s')
    return


def try_out_3d_rotation(traces_df):
    '''
    This function creates the 3D PCA plots in the interactive matplotlib.pyplot plots instead of PDF figures. The interactive plot allows manually finding of the best orienation for visualization.
    The orientation values elev and azim are printed when the plot is closed.
    :param traces_df: Dataframe with the single trial activity traces of all neurons.
    '''
    print(len(traces_df))

    # Loop over the stimuli to create the full functional activity array with 3 trials per stimulus
    for stim_idx, stim_name in enumerate(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off',
                                          'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                                          'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off']):

        if stim_idx == 0:
            all_traces = np.vstack(([traces_df[f'{stim_name}_trial_0_trace_{i}'].astype(float) for i in range(120)],
                                    [traces_df[f'{stim_name}_trial_1_trace_{i}'].astype(float) for i in range(120)],
                                    [traces_df[f'{stim_name}_trial_2_trace_{i}'].astype(float) for i in range(120)]))
        else:
            all_traces = np.vstack((all_traces,
                                    [traces_df[f'{stim_name}_trial_0_trace_{i}'].astype(float) for i in range(120)],
                                    [traces_df[f'{stim_name}_trial_1_trace_{i}'].astype(float) for i in range(120)],
                                    [traces_df[f'{stim_name}_trial_2_trace_{i}'].astype(float) for i in range(120)]))
    # Normalize the activity array.
    all_traces[np.isnan(all_traces)] = 0
    all_traces = (all_traces - np.nanmin(all_traces)) / (np.nanmax(all_traces) - np.nanmin(all_traces))

    # Perform PCA on the activity array.
    pca = PCA(n_components=np.min(all_traces.shape))
    out = pca.fit_transform(all_traces)

    print(np.sum(pca.explained_variance_ratio_[:50]))
    print(np.sum(pca.explained_variance_ratio_[:30]))
    print(np.sum(pca.explained_variance_ratio_[:3]))

    # Plot the first 3 PCs in the interactive matplotlib.pyplot figure. Colored by stimulus.
    fig = plt.figure(figsize=(1, 1))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter3D(out[:, 0], out[:, 1], out[:, 2],
                  c=np.repeat(['g', 'g', 'g', 'm', 'm', 'm', 'orange', 'orange', 'orange',
                               'm', 'm', 'm', 'g', 'g', 'g', 'orange', 'orange', 'orange',
                               'b', 'b', 'b', 'b', 'b', 'b', 'gray', 'gray', 'gray'], 120))
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    plt.show()

    # Print the last chosen Elev and Azim variables as well as the minimum and maximum sizes of the PCA plot.
    print('ELEV and AZIM')
    print(ax1.elev)
    print(ax1.azim)

    print('MAX sizes PCA')
    print(np.nanmin(out[:, :3], axis=0))
    print(np.nanmax(out[:, :3], axis=0))

    out_red = out[:, :3]
    print(np.sum(pca.explained_variance_ratio_[:3]))

    # Compute the control distance between pairs of trials of the same stimulus.
    first_trial = True
    for stim_idx in [0, 360, 720, 1080, 1440, 1800, 2160, 2520]:
        for trial_a, trial_b in zip([0, 120, 240], [120, 240, 0]):
            if first_trial:
                ctrl_stim_distances = [math.dist(out_red[stim_idx + trial_a + i, :], out_red[stim_idx + trial_b + i, :])
                                       for i in range(120)]
                first_trial = False
            else:
                ctrl_stim_distances = np.vstack((ctrl_stim_distances, [
                    math.dist(out_red[stim_idx + trial_a + i, :], out_red[stim_idx + trial_b + i, :]) for i in
                    range(120)]))

    # Compute the distance between motion left and right trials.
    first_trial = True
    stim_idx_a = 2160
    stim_idx_b = 2520
    for trial_a, trial_b in zip([0, 0, 0, 120, 120, 120, 240, 240, 240], [0, 120, 240, 0, 120, 240, 0, 120, 240]):
        if first_trial:
            mot_stim_distances = [math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :])
                                  for i in range(120)]
            first_trial = False
        else:
            mot_stim_distances = np.vstack((mot_stim_distances, [
                math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :]) for i in
                range(120)]))

    # Compute the distance between luminance left and right trials.
    first_trial = True
    stim_idx_a = 720
    stim_idx_b = 1800
    for trial_a, trial_b in zip([0, 0, 0, 120, 120, 120, 240, 240, 240], [0, 120, 240, 0, 120, 240, 0, 120, 240]):
        if first_trial:
            lumi_stim_distances = [math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :])
                                   for i in range(120)]
            first_trial = False
        else:
            lumi_stim_distances = np.vstack((lumi_stim_distances, [
                math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :]) for i in
                range(120)]))

    # Compute the distance between congruent left and right trials.
    first_trial = True
    stim_idx_a = 0
    stim_idx_b = 1440
    for trial_a, trial_b in zip([0, 0, 0, 120, 120, 120, 240, 240, 240], [0, 120, 240, 0, 120, 240, 0, 120, 240]):
        if first_trial:
            same_stim_distances = [math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :])
                                   for i in range(120)]
            first_trial = False
        else:
            same_stim_distances = np.vstack((same_stim_distances, [
                math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :]) for i in
                range(120)]))

    # Compute the distance between conflicting left and right trials.
    first_trial = True
    stim_idx_a = 360
    stim_idx_b = 1080
    for trial_a, trial_b in zip([0, 0, 0, 120, 120, 120, 240, 240, 240], [0, 120, 240, 0, 120, 240, 0, 120, 240]):
        if first_trial:
            oppo_stim_distances = [math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :])
                                   for i in range(120)]
            first_trial = False
        else:
            oppo_stim_distances = np.vstack((oppo_stim_distances, [
                math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :]) for i in
                range(120)]))

    # Plot the distances (both control and for the 4 stimulus classes) and print the maximum distance to be able to set the y-axis size for the PDF plots.
    fig, ax = plt.subplots(1, 1)
    ax.fill_between(np.arange(120), np.nanmean(ctrl_stim_distances, axis=0) - np.nanstd(ctrl_stim_distances, axis=0),
                    np.nanmean(ctrl_stim_distances, axis=0) + np.nanstd(ctrl_stim_distances, axis=0), color='gray',
                    alpha=0.5)
    ax.plot(np.arange(120), np.nanmean(ctrl_stim_distances, axis=0), c='gray')
    print(np.max(mot_stim_distances))
    print(np.max(lumi_stim_distances))
    print(np.max(same_stim_distances))
    print(np.max(oppo_stim_distances))
    for t in range(9):
        ax.plot(np.arange(120), mot_stim_distances[t, :], c='b')
        ax.plot(np.arange(120), lumi_stim_distances[t, :], c='orange')
        ax.plot(np.arange(120), same_stim_distances[t, :], c='g')
        ax.plot(np.arange(120), oppo_stim_distances[t, :], c='m')
    plt.show()
    return


def subplot_pca_wholebrain(traces_df, subfig_expvar, subfig_expvar_zoomin, subfig_pca, subfig_pca_mot, subfig_pca_lumi,
                           subfig_dist, brain_xy_overview_plot, brain_yz_overview_plot):
    '''
    This function plots the PCA analysis of the total brain with the explained variance over PCs.
    This is related to Fig. S5b-d, column 1 in c-d
     :param traces_df: Dataframe with the single trial functional activity per cell.
    :param subfig_expvar: Subfigure to show the full explained variance over PCs.
    :param subfig_expvar_zoomin: Subfigure to show the explained variance of the first 200 PCs.
    :param subfig_pca: Subfigure to plot the first 3 PCs colored by stimulus.
    :param subfig_pca_mot: Subfigure to plot the first 3 PCs colored by motion left/right.
    :param subfig_pca_lumi: Subfigure to plot the first 3 PCs colored by luminance left/right.
    :param subfig_dist: Subfigure to plot the distance between left and right trials of the same stimulus in the first 3 PCs.
    :param brain_xy_overview_plot: Subfigure to show the brain overview reference in xy.
    :param brain_yz_overview_plot: Subfigure to show the brain overview reference in yz.
    '''
    print(len(traces_df))
    # Loop over the stimuli to create the full functional activity array with 3 trials per stimulus
    for stim_idx, stim_name in enumerate(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off',
                                          'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                                          'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off']):
        if stim_idx == 0:
            all_traces = np.vstack(([traces_df[f'{stim_name}_trial_0_trace_{i}'].astype(float) for i in range(120)],
                                    [traces_df[f'{stim_name}_trial_1_trace_{i}'].astype(float) for i in range(120)],
                                    [traces_df[f'{stim_name}_trial_2_trace_{i}'].astype(float) for i in range(120)]))
        else:
            all_traces = np.vstack((all_traces,
                                    [traces_df[f'{stim_name}_trial_0_trace_{i}'].astype(float) for i in range(120)],
                                    [traces_df[f'{stim_name}_trial_1_trace_{i}'].astype(float) for i in range(120)],
                                    [traces_df[f'{stim_name}_trial_2_trace_{i}'].astype(float) for i in range(120)]))
    # Normalize the activity array.
    all_traces[np.isnan(all_traces)] = 0
    all_traces = (all_traces - np.nanmin(all_traces)) / (np.nanmax(all_traces) - np.nanmin(all_traces))

    # Perform PCA on the activity array.
    pca = PCA(n_components=np.min(all_traces.shape))
    out = pca.fit_transform(all_traces)

    # Plot the explained variance for all PCs as well as a zoom-in to the first 200 PCs.
    subfig_expvar.draw_line(np.arange(len(pca.explained_variance_ratio_)), np.cumsum(pca.explained_variance_ratio_),
                            lc='tab:blue')
    subfig_expvar_zoomin.draw_line(np.arange(200), np.cumsum(pca.explained_variance_ratio_[:200]), lc='tab:blue')
    print(np.sum(pca.explained_variance_ratio_[:50]))
    print(np.sum(pca.explained_variance_ratio_[:30]))
    print(np.sum(pca.explained_variance_ratio_[:3]))

    # Prepare the color labels, by stimulus (pca_colors), motion left/right (pca_mot_colors), and luminance left/right (pca_lumi_colors).
    pca_colors = ['g', 'g', 'g', 'm', 'm', 'm', 'orange', 'orange', 'orange',
                  'm', 'm', 'm', 'g', 'g', 'g', 'orange', 'orange', 'orange',
                  'b', 'b', 'b', 'b', 'b', 'b', 'gray', 'gray', 'gray']
    pca_mot_colors = ['darkgreen', 'darkgreen', 'darkgreen', 'springgreen', 'springgreen', 'springgreen', 'gray',
                      'gray', 'gray',
                      'darkgreen', 'darkgreen', 'darkgreen', 'springgreen', 'springgreen', 'springgreen', 'gray',
                      'gray', 'gray',
                      'darkgreen', 'darkgreen', 'darkgreen', 'springgreen', 'springgreen', 'springgreen', 'gray',
                      'gray', 'gray']
    pca_lumi_colors = ['saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown',
                       'saddlebrown', 'saddlebrown', 'saddlebrown',
                       'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown',
                       'sandybrown', 'sandybrown',
                       'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray']

    # Plot the first 3 PCs labeled by stimulus, motion left/right, and luminance left/right.
    subfig_pca.draw_scatter3D(out[:, 0], out[:, 1], out[:, 2], ec=None, ps=1,
                              pc=np.repeat(pca_colors, 120))

    subfig_pca_mot.draw_scatter3D(out[:, 0], out[:, 1], out[:, 2], ec=None, ps=1,
                                  pc=np.repeat(pca_mot_colors, 120))
    subfig_pca_lumi.draw_scatter3D(out[:, 0], out[:, 1], out[:, 2], ec=None, ps=1,
                                   pc=np.repeat(pca_lumi_colors, 120))

    # Print the max sizes (this can be used to fine-tune the size of the x-, y- and z-axis of the 3D plots.
    print('MAX sizes PCA')
    print(np.nanmin(out[:, :3], axis=0))
    print(np.nanmax(out[:, :3], axis=0))

    out_red = out[:, :3]
    print(np.sum(pca.explained_variance_ratio_[:3]))

    # Compute the control distance between pairs of trials of the same stimulus.
    first_trial = True
    for stim_idx in [0, 360, 720, 1080, 1440, 1800, 2160, 2520]:
        for trial_a, trial_b in zip([0, 120, 240], [120, 240, 0]):
            if first_trial:
                ctrl_stim_distances = [math.dist(out_red[stim_idx + trial_a + i, :], out_red[stim_idx + trial_b + i, :])
                                       for i in range(120)]
                first_trial = False
            else:
                ctrl_stim_distances = np.vstack((ctrl_stim_distances, [
                    math.dist(out_red[stim_idx + trial_a + i, :], out_red[stim_idx + trial_b + i, :]) for i in
                    range(120)]))

    # Compute the distance between motion left and right trials.
    first_trial = True
    stim_idx_a = 2160
    stim_idx_b = 2520
    for trial_a, trial_b in zip([0, 120, 240], [0, 120, 240]):
        if first_trial:
            mot_stim_distances = [math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :])
                                  for i in range(120)]
            first_trial = False
        else:
            mot_stim_distances = np.vstack((mot_stim_distances, [
                math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :]) for i in
                range(120)]))

    # Compute the distance between luminance left and right trials.
    first_trial = True
    stim_idx_a = 720
    stim_idx_b = 1800
    for trial_a, trial_b in zip([0, 120, 240], [0, 120, 240]):
        if first_trial:
            lumi_stim_distances = [math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :])
                                   for i in range(120)]
            first_trial = False
        else:
            lumi_stim_distances = np.vstack((lumi_stim_distances, [
                math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :]) for i in
                range(120)]))

    # Compute the distance between congruent left and right trials.
    first_trial = True
    stim_idx_a = 0
    stim_idx_b = 1440
    for trial_a, trial_b in zip([0, 120, 240], [0, 120, 240]):
        if first_trial:
            same_stim_distances = [math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :])
                                   for i in range(120)]
            first_trial = False
        else:
            same_stim_distances = np.vstack((same_stim_distances, [
                math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :]) for i in
                range(120)]))

    # Compute the distance between conflicting left and right trials.
    first_trial = True
    stim_idx_a = 360
    stim_idx_b = 1080
    for trial_a, trial_b in zip([0, 120, 240], [0, 120, 240, ]):
        if first_trial:
            oppo_stim_distances = [math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :])
                                   for i in range(120)]
            first_trial = False
        else:
            oppo_stim_distances = np.vstack((oppo_stim_distances, [
                math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :]) for i in
                range(120)]))

    # Plot the control distance as mean +- STD in each plot.
    subfig_dist.draw_line(np.arange(120), np.nanmean(ctrl_stim_distances, axis=0),
                          yerr=np.nanstd(ctrl_stim_distances, axis=0), lc='#676767', eafc='#989898', eaalpha=1.0, lw=1,
                          ealw=1, eaec='#989898')

    # Use the maximum distance to set the yaxis size of the distance plot.
    print(np.max(mot_stim_distances))
    print(np.max(lumi_stim_distances))
    print(np.max(same_stim_distances))
    print(np.max(oppo_stim_distances))
    # Loop over the 3 trials and plot the distance of the 4 stimulus classes.
    for t in range(3):
        subfig_dist.draw_line(np.arange(120), mot_stim_distances[t, :], lc='b')
        subfig_dist.draw_line(np.arange(120), lumi_stim_distances[t, :], lc='orange')
        subfig_dist.draw_line(np.arange(120), same_stim_distances[t, :], lc='g')
        subfig_dist.draw_line(np.arange(120), oppo_stim_distances[t, :], lc='m')

    # Perform a ttest to check a significant difference between congruent and conflicting stimuli.
    _, pval = ttest_ind(same_stim_distances[:, 20:80].mean(axis=1), oppo_stim_distances[:, 20:80].mean(axis=1))
    print(
        f'Cohen D: {cohens_d(same_stim_distances[:, 20:80].mean(axis=1), oppo_stim_distances[:, 20:80].mean(axis=1))}')
    print(f'pval: {pval}')

    if pval < 0.001:
        subfig_dist.draw_text(50, 1.1 * np.max([same_stim_distances.max(), oppo_stim_distances.max()]), '***')
    elif pval < 0.01:
        subfig_dist.draw_text(50, 1.1 * np.max([same_stim_distances.max(), oppo_stim_distances.max()]), '**')
    elif pval < 0.05:
        subfig_dist.draw_text(50, 1.1 * np.max([same_stim_distances.max(), oppo_stim_distances.max()]), '*')

    # Add the small brain region plots to highlight that the total brain was tested.
    total_brain_regions = [navis.read_mesh(
        r'Z:\Zebrafish atlases\z_brain_atlas\region_masks_mapzebrain_atlas2024\prosencephalon_(forebrain).obj',
        units='microns', output='volume'),
                           navis.read_mesh(
                               r'Z:\Zebrafish atlases\z_brain_atlas\region_masks_mapzebrain_atlas2024\mesencephalon_(midbrain).obj',
                               units='microns', output='volume'),
                           navis.read_mesh(
                               r'Z:\Zebrafish atlases\z_brain_atlas\region_masks_mapzebrain_atlas2024\rhombencephalon_(hindbrain).obj',
                               units='microns', output='volume')]

    brain_xy_overview_plot.draw_navis_neuron(None, total_brain_regions, navis_color='gray', navis_view=('x', '-y'),
                                             lw=0.5, rasterized=True)
    brain_yz_overview_plot.draw_navis_neuron(None, total_brain_regions, navis_color='gray', navis_view=('z', '-y'),
                                             lw=0.5, rasterized=True)

    return


def subplot_pca_per_region(traces_df, regions_path, regions, subfig_pca_regions, subfig_pca_mot_regions,
                           subfig_pca_lumi_regions, subfig_dist_regions, brain_xy_overview_plot_regions,
                           brain_yz_overview_plot_regions):
    '''
    This function plots the PCA analysis per brain-region.
    This is related to Fig. S5c-d, column 2-4.
    :param traces_df: Dataframe with the single trial functional activity per cell.
    :param regions_path: Path to the hdf5 file containing all mapzebrain regions.
    :param regions: List with mapzebrain region names to analysis.
    :param subfig_pca_regions: List of Subfigures to plot the first 3 PCs per brain region colored by stimulus.
    :param subfig_pca_mot_regions: List of Subfigures to plot the first 3 PCs per brain region colored by motion left/right.
    :param subfig_pca_lumi_regions:  List of Subfigures to plot the first 3 PCs per brain region colored by luminance left/right.
    :param subfig_dist_regions: List of Subfigures to plot the distance between left and right trials of the same stimulus in the first 3 PCs per brain region.
    :param brain_xy_overview_plot_regions: List of Subfigures to show the overview of the brain region reference in xy.
    :param brain_yz_overview_plot_regions: List of Subfigures to show the overview of the brain region reference in yz.
    '''
    # Check for each neuron which brain region it is part of by using the regions_mask.
    print(len(traces_df))
    region_masks = create_combined_region_npy_mask(regions_path, regions=regions)
    region_ids = region_masks[
        traces_df['ZB_x'].astype(int), traces_df['ZB_y'].astype(int), traces_df['ZB_z'].astype(int)]

    # Loop over the brain regions.
    print(len(traces_df))
    for region_id in range(len(regions) + 1):
        # Region ID 0 means outside any of the listed regions, we skip this one.
        if region_id == 0:
            continue
        else:
            print(regions[region_id - 1])

        # Get the dataframe subset for the current region
        region_df = traces_df[region_ids == region_id]

        # Loop over the stimuli to create the full functional activity array with 3 trials per stimulus
        print(len(region_df))
        for stim_idx, stim_name in enumerate(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off',
                                              'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                                              'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off']):
            if stim_idx == 0:
                all_traces = np.vstack(([region_df[f'{stim_name}_trial_0_trace_{i}'].astype(float) for i in range(120)],
                                        [region_df[f'{stim_name}_trial_1_trace_{i}'].astype(float) for i in range(120)],
                                        [region_df[f'{stim_name}_trial_2_trace_{i}'].astype(float) for i in
                                         range(120)]))
            else:
                all_traces = np.vstack((all_traces,
                                        [region_df[f'{stim_name}_trial_0_trace_{i}'].astype(float) for i in range(120)],
                                        [region_df[f'{stim_name}_trial_1_trace_{i}'].astype(float) for i in range(120)],
                                        [region_df[f'{stim_name}_trial_2_trace_{i}'].astype(float) for i in
                                         range(120)]))
        # Normalize the activity array.
        all_traces[np.isnan(all_traces)] = 0
        all_traces = (all_traces - np.nanmin(all_traces)) / (np.nanmax(all_traces) - np.nanmin(all_traces))

        # Perform PCA on the activity array.
        pca = PCA(n_components=np.min(all_traces.shape))
        out = pca.fit_transform(all_traces)

        # Print the explained variance for 50, 30, and 3 components.
        print(np.sum(pca.explained_variance_ratio_[:50]))
        print(np.sum(pca.explained_variance_ratio_[:30]))
        print(np.sum(pca.explained_variance_ratio_[:3]))

        # Prepare the color labels, by stimulus (pca_colors), motion left/right (pca_mot_colors), and luminance left/right (pca_lumi_colors).
        pca_colors = ['g', 'g', 'g', 'm', 'm', 'm', 'orange', 'orange', 'orange',
                      'm', 'm', 'm', 'g', 'g', 'g', 'orange', 'orange', 'orange',
                      'b', 'b', 'b', 'b', 'b', 'b', 'gray', 'gray', 'gray']
        pca_mot_colors = ['darkgreen', 'darkgreen', 'darkgreen', 'springgreen', 'springgreen', 'springgreen', 'gray',
                          'gray', 'gray',
                          'darkgreen', 'darkgreen', 'darkgreen', 'springgreen', 'springgreen', 'springgreen', 'gray',
                          'gray', 'gray',
                          'darkgreen', 'darkgreen', 'darkgreen', 'springgreen', 'springgreen', 'springgreen', 'gray',
                          'gray', 'gray']
        pca_lumi_colors = ['saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown',
                           'saddlebrown', 'saddlebrown', 'saddlebrown',
                           'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown',
                           'sandybrown', 'sandybrown', 'sandybrown',
                           'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray']

        # Plot the first 3 PCs labeled by stimulus, motion left/right, and luminance left/right.
        subfig_pca_regions[region_id - 1].draw_scatter3D(out[:, 0], out[:, 1], out[:, 2], ec=None, ps=1,
                                                         pc=np.repeat(pca_colors, 120))
        subfig_pca_mot_regions[region_id - 1].draw_scatter3D(out[:, 0], out[:, 1], out[:, 2], ec=None, ps=1,
                                                             pc=np.repeat(pca_mot_colors, 120))
        subfig_pca_lumi_regions[region_id - 1].draw_scatter3D(out[:, 0], out[:, 1], out[:, 2], ec=None, ps=1,
                                                              pc=np.repeat(pca_lumi_colors, 120))

        # Print the max sizes (this can be used to fine-tune the size of the x-, y- and z-axis in the xmaxs, ymaxs and zmaxs parameters.
        print('MAX sizes PCA')
        print(np.nanmin(out[:, :3], axis=0))
        print(np.nanmax(out[:, :3], axis=0))

        out_red = out[:, :3]
        print(np.sum(pca.explained_variance_ratio_[:3]))

        # Compute the control distance between pairs of trials of the same stimulus.
        first_trial = True
        for stim_idx in [0, 360, 720, 1080, 1440, 1800, 2160, 2520]:
            for trial_a, trial_b in zip([0, 120, 240], [120, 240, 0]):
                if first_trial:
                    ctrl_stim_distances = [
                        math.dist(out_red[stim_idx + trial_a + i, :], out_red[stim_idx + trial_b + i, :])
                        for i in range(120)]
                    first_trial = False
                else:
                    ctrl_stim_distances = np.vstack((ctrl_stim_distances, [
                        math.dist(out_red[stim_idx + trial_a + i, :], out_red[stim_idx + trial_b + i, :]) for i in
                        range(120)]))

        # Compute the distance between motion left and right trials.
        first_trial = True
        stim_idx_a = 2160
        stim_idx_b = 2520
        for trial_a, trial_b in zip([0, 120, 240], [0, 120, 240]):
            if first_trial:
                mot_stim_distances = [
                    math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :])
                    for i in range(120)]
                first_trial = False
            else:
                mot_stim_distances = np.vstack((mot_stim_distances, [
                    math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :]) for i in
                    range(120)]))

        # Compute the distance between luminance left and right trials.
        first_trial = True
        stim_idx_a = 720
        stim_idx_b = 1800
        for trial_a, trial_b in zip([0, 120, 240], [0, 120, 240]):
            if first_trial:
                lumi_stim_distances = [
                    math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :])
                    for i in range(120)]
                first_trial = False
            else:
                lumi_stim_distances = np.vstack((lumi_stim_distances, [
                    math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :]) for i in
                    range(120)]))

        # Compute the distance between congruent left and right trials.
        first_trial = True
        stim_idx_a = 0
        stim_idx_b = 1440
        for trial_a, trial_b in zip([0, 120, 240], [0, 120, 240]):
            if first_trial:
                same_stim_distances = [
                    math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :])
                    for i in range(120)]
                first_trial = False
            else:
                same_stim_distances = np.vstack((same_stim_distances, [
                    math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :]) for i in
                    range(120)]))

        # Compute the distance between conflicting left and right trials.
        first_trial = True
        stim_idx_a = 360
        stim_idx_b = 1080
        for trial_a, trial_b in zip([0, 120, 240], [0, 120, 240]):
            if first_trial:
                oppo_stim_distances = [
                    math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :])
                    for i in range(120)]
                first_trial = False
            else:
                oppo_stim_distances = np.vstack((oppo_stim_distances, [
                    math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :]) for i in
                    range(120)]))

        # Plot the control distance as mean +- STD in each plot.
        subfig_dist_regions[region_id - 1].draw_line(np.arange(120), np.nanmean(ctrl_stim_distances, axis=0),
                                                     yerr=np.nanstd(ctrl_stim_distances, axis=0), lc='#676767',
                                                     eafc='#989898', eaalpha=1.0, lw=1, ealw=1, eaec='#989898')

        # Use the maximum distance to set the dmaxs plotting parameters.
        print(np.max(mot_stim_distances))
        print(np.max(lumi_stim_distances))
        print(np.max(same_stim_distances))
        print(np.max(oppo_stim_distances))
        # Loop over the 3 trials and plot the distance of the 4 stimulus classes.
        for t in range(3):
            subfig_dist_regions[region_id - 1].draw_line(np.arange(120), mot_stim_distances[t, :], lc='b')
            subfig_dist_regions[region_id - 1].draw_line(np.arange(120), lumi_stim_distances[t, :], lc='orange')
            subfig_dist_regions[region_id - 1].draw_line(np.arange(120), same_stim_distances[t, :], lc='g')
            subfig_dist_regions[region_id - 1].draw_line(np.arange(120), oppo_stim_distances[t, :], lc='m')
        if region_id == len(regions):
            subfig_dist_regions[region_id - 1].draw_line([100, 120], [0.01, 0.01], lc='k')
            subfig_dist_regions[region_id - 1].draw_text(110, -0.15, '10s')

        # Perform a ttest to check a significant difference between congruent and conflicting stimuli.
        _, pval = ttest_ind(same_stim_distances[:, 20:80].mean(axis=1), oppo_stim_distances[:, 20:80].mean(axis=1))
        print(
            f'Cohen D: {cohens_d(same_stim_distances[:, 20:80].mean(axis=1), oppo_stim_distances[:, 20:80].mean(axis=1))}')
        print(f'pval: {pval}')

        if pval < 0.001:
            subfig_dist_regions[region_id - 1].draw_text(50, 1.1 * np.max(
                [same_stim_distances.max(), oppo_stim_distances.max()]), '***')
        elif pval < 0.01:
            subfig_dist_regions[region_id - 1].draw_text(50, 1.1 * np.max(
                [same_stim_distances.max(), oppo_stim_distances.max()]), '**')
        elif pval < 0.05:
            subfig_dist_regions[region_id - 1].draw_text(50, 1.1 * np.max(
                [same_stim_distances.max(), oppo_stim_distances.max()]), '*')

        # Add the small brain region plots to highlight which region was tested.
        total_brain_regions = [navis.read_mesh(
            r'Z:\Zebrafish atlases\z_brain_atlas\region_masks_mapzebrain_atlas2024\prosencephalon_(forebrain).obj',
            units='microns', output='volume'),
            navis.read_mesh(
                r'Z:\Zebrafish atlases\z_brain_atlas\region_masks_mapzebrain_atlas2024\mesencephalon_(midbrain).obj',
                units='microns', output='volume'),
            navis.read_mesh(
                r'Z:\Zebrafish atlases\z_brain_atlas\region_masks_mapzebrain_atlas2024\rhombencephalon_(hindbrain).obj',
                units='microns', output='volume')]
        brain_regions = [navis.read_mesh(
            rf'Z:\Zebrafish atlases\z_brain_atlas\region_masks_mapzebrain_atlas2024\{regions[region_id - 1]}.obj',
            units='microns', output='volume'), ]

        brain_xy_overview_plot_regions[region_id - 1].draw_navis_neuron(None, total_brain_regions,
                                                                        navis_view=('x', '-y'), lw=0.5, rasterized=True)
        brain_xy_overview_plot_regions[region_id - 1].draw_navis_neuron(None, brain_regions, navis_color='gray',
                                                                        navis_view=('x', '-y'), lw=0.5, rasterized=True)
        brain_yz_overview_plot_regions[region_id - 1].draw_navis_neuron(None, total_brain_regions,
                                                                        navis_view=('z', '-y'), lw=0.5, rasterized=True)
        brain_yz_overview_plot_regions[region_id - 1].draw_navis_neuron(None, brain_regions, navis_color='gray',
                                                                        navis_view=('z', '-y'), lw=0.5, rasterized=True)

    return


if __name__ == '__main__':
    # Provide the path to save the figure.
    fig_save_path = 'C:/users/katja/Desktop/fig_S5.pdf'

    # Provide the path to the figure_3 folder.
    fig_3_folder_path = r'Z:\Bahl lab member directories\Katja\paper_data\figure_3'

    # Load the csv file with single trial traces.
    print('Loading traces')
    traces_df = pd.read_csv(fr'{fig_3_folder_path}\imaging_traces_trials_baseline.csv')
    print('Traces loaded')

    # Get the path to the hdf5 file containing all mapzebrain regions and define the major ones to analyse (Fig. S5c-d)
    regions_path = fr'{fig_3_folder_path}\all_masks_indexed.hdf5'
    regions = ['prosencephalon_(forebrain)', 'mesencephalon_(midbrain)', 'rhombencephalon_(hindbrain)']

    # If try_out_and_set_3d_orientation_angles is True, the interactive matplotlib.pyplot window will pop up and allow you to find the preferred rotation.
    # The rotation parameters and printed and should be filled in the corresponding elevs and azims parameters.
    try_out_and_set_3d_orientation_angles = False

    if try_out_and_set_3d_orientation_angles:
        # Use the region_masks to find which major brain region each neuron is part of.
        region_masks = create_combined_region_npy_mask(regions_path, regions=regions)
        region_ids = region_masks[
            traces_df['ZB_x'].astype(int), traces_df['ZB_y'].astype(int), traces_df['ZB_z'].astype(int)]

        # Loop over the regions and manually optimize the orientation of the 3D plots.
        print(len(traces_df))
        for region_id in range(len(regions) + 1):
            if region_id < 1:
                print('Rest of the brain')
            else:
                print(regions[region_id - 1])

            region_df = traces_df[region_ids == region_id]
            try_out_3d_rotation(region_df)

    # Prepare the figures for Figure S5.
    fig = Figure(fig_width=18, fig_height=17)

    # Fig. S5a
    subfigs_trial_traces = [[]] * 9
    for t in range(9):
        if t == 2:
            subfigs_trial_traces[t] = fig.create_plot(xpos=1 + t % 3 * 4.2, ypos=14.8 - int(t / 3) * 2.2, plot_height=2,
                                                      plot_width=3.5, axis_off=True,
                                                      xmin=0, xmax=170, ymin=108118 + 50000, ymax=0,
                                                      show_colormap=True, zmin=0, zmax=1, colormap='viridis',
                                                      zticks=[0, 1], zl='norm. fluorescence')
        else:
            subfigs_trial_traces[t] = fig.create_plot(xpos=1 + t % 3 * 4.2, ypos=14.8 - int(t / 3) * 2.2, plot_height=2,
                                                      plot_width=3.5, axis_off=True,
                                                      xmin=0, xmax=170, ymin=108118 + 50000, ymax=0)

    # Fig. S5b
    subfig_expvar = fig.create_plot(xpos=15, ypos=11.5, plot_height=2, plot_width=2, xmin=0, xmax=3240, ymin=0, ymax=1,
                                    xl='PC', yl='cum. explained (%)', xticks=[0, 1000, 2000, 3000],
                                    yticks=[0, 0.25, 0.5, 0.75, 1],
                                    yticklabels=['0', '25', '50', '75', '100'])
    subfig_expvar_zoomin = fig.create_plot(xpos=15, ypos=14.5, plot_height=2, plot_width=2, xmin=0, xmax=200, ymin=0,
                                           ymax=1, yl='cum. explained (%)', xticks=[0, 50, 100, 150, 200],
                                           yticks=[0, 0.25, 0.5, 0.75, 1],
                                           yticklabels=['0', '25', '50', '75', '100'])

    # Fig. S5c 1st column
    subfig_pca = fig.create_plot3D(xpos=1, ypos=5.5, plot_height=3.4, plot_width=3.4, xmin=-1.1 * 0.4, xmax=1.1 * 0.4,
                                   ymin=-1.1 * 0.4,
                                   ymax=1.1 * 0.4,
                                   zmin=-1.1 * 0.4, zmax=1.1 * 0.4, helper_lines_dashes=None, xyzl_3d=True,
                                   xticks=[-0.4, -0.2, 0, 0.2, 0.4], yticks=[-0.4, -0.2, 0, 0.2, 0.4],
                                   zticks=[-0.4, 0, 0.4],
                                   xticklabels=['', '', '', '', ''], yticklabels=['', '', '', '', ''],
                                   zticklabels=['', '', ''],
                                   xl_distance=0, yl_distance=0, zl_distance=0,
                                   elev=52.22, azim=-38.07, xl='PC1', yl='PC2', zl='PC3')
    subfig_pca_mot = fig.create_plot3D(xpos=1, ypos=3.5, plot_height=1.7, plot_width=1.7, xmin=-1.1 * 0.4,
                                       xmax=1.1 * 0.4, ymin=-1.1 * 0.4,
                                       ymax=1.1 * 0.4,
                                       zmin=-1.1 * 0.4, zmax=1.1 * 0.4, helper_lines_dashes=None, xyzl_3d=True,
                                       xticks=[-0.4, -0.2, 0, 0.2, 0.4], yticks=[-0.4, -0.2, 0, 0.2, 0.4],
                                       zticks=[-0.4, 0, 0.4],
                                       xticklabels=['', '', '', '', ''], yticklabels=['', '', '', '', ''],
                                       zticklabels=['', '', ''],
                                       elev=52.22, azim=-38.07)
    subfig_pca_lumi = fig.create_plot3D(xpos=2.7, ypos=3.5, plot_height=1.7, plot_width=1.7, xmin=-1.1 * 0.4,
                                        xmax=1.1 * 0.4, ymin=-1.1 * 0.4,
                                        ymax=1.1 * 0.4,
                                        zmin=-1.1 * 0.4, zmax=1.1 * 0.4, helper_lines_dashes=None, xyzl_3d=True,
                                        xticks=[-0.4, -0.2, 0, 0.2, 0.4], yticks=[-0.4, -0.2, 0, 0.2, 0.4],
                                        zticks=[-0.4, 0, 0.4],
                                        xticklabels=['', '', '', '', ''], yticklabels=['', '', '', '', ''],
                                        zticklabels=['', '', ''],
                                        elev=52.22, azim=-38.07)

    # Fig. S5d first column
    subfig_dist = fig.create_plot(xpos=1, ypos=0.5, plot_height=2.75, plot_width=3.4, xmin=0, xmax=120, ymin=0,
                                  ymax=1.0,
                                  yticks=[0.0, 0.5, 1.0, ], vspans=[[20, 80, 'lightgray', 1.0]],
                                  yl='distance in PC1-3')

    # Fig. S5c brain region reference overview.
    brain_xy_overview_plot = fig.create_plot(xpos=1., ypos=8.3, plot_height=2, plot_width=2 / 2.274, axis_off=True)
    brain_yz_overview_plot = fig.create_plot(xpos=2.2, ypos=8.3, plot_height=2, plot_width=2 / 4.395, axis_off=True)

    # Prepare lists for the region-based plots in Fig. S5c-d
    subfig_pca_regions = [[]] * len(regions)
    subfig_pca_mot_regions = [[]] * len(regions)
    subfig_pca_lumi_regions = [[]] * len(regions)
    subfig_dist_regions = [[]] * len(regions)
    brain_xy_overview_plot_regions = [[]] * len(regions)
    brain_yz_overview_plot_regions = [[]] * len(regions)
    # 3D Orientation parameters, these can be found using try_out_and_set_3d_orientation_angles=True.
    elevs = [25.05, 33.32, 11.47]
    azims = [159.66, 33.94, 73.93]
    xmaxs = [0.5, 2.1, 2.3]
    ymaxs = [0.3, 1.8, 2.7]
    zmaxs = [0.3, 1.6, 2.5]
    dmaxs = [0.6, 4, 5.0]
    dticks = [[0.0, 0.2, 0.4, 0.6], [0, 2, 4], [0, 2, 4]]
    # Loop over the brain-regions.
    for r, elev, azim, xmax, ymax, zmax, dmax, dtick in zip(range(len(regions)), elevs, azims, xmaxs, ymaxs, zmaxs,
                                                            dmaxs, dticks):
        # Fig. S5c column 2-4
        subfig_pca_regions[r] = fig.create_plot3D(xpos=5.5 + r * 4, ypos=5.5, plot_height=3.4, plot_width=3.4,
                                                  xmin=-1.1 * xmax, xmax=1.1 * xmax, ymin=-1.1 * ymax,
                                                  ymax=1.1 * ymax,
                                                  zmin=-1.1 * zmax, zmax=1.1 * zmax, helper_lines_dashes=None,
                                                  xyzl_3d=True,
                                                  xticks=[-xmax, -xmax / 2, 0, xmax / 2, xmax],
                                                  yticks=[-ymax, -ymax / 2, 0, ymax / 2, ymax],
                                                  zticks=[-zmax, 0, zmax],
                                                  xticklabels=['', '', '', '', ''], yticklabels=['', '', '', '', ''],
                                                  zticklabels=['', '', ''],
                                                  xl_distance=0, yl_distance=0, zl_distance=0,
                                                  elev=elev, azim=azim, xl='PC1', yl='PC2', zl='PC3')
        subfig_pca_mot_regions[r] = fig.create_plot3D(xpos=5.5 + r * 4, ypos=3.5, plot_height=1.7, plot_width=1.7,
                                                      xmin=-1.1 * xmax, xmax=1.1 * xmax, ymin=-1.1 * ymax,
                                                      ymax=1.1 * ymax,
                                                      zmin=-1.1 * zmax, zmax=1.1 * zmax, helper_lines_dashes=None,
                                                      xyzl_3d=True,
                                                      xticks=[-xmax, -xmax / 2, 0, xmax / 2, xmax],
                                                      yticks=[-ymax, -ymax / 2, 0, ymax / 2, ymax],
                                                      zticks=[-zmax, 0, zmax],
                                                      xticklabels=['', '', '', '', ''],
                                                      yticklabels=['', '', '', '', ''],
                                                      zticklabels=['', '', ''],
                                                      elev=elev, azim=azim)
        subfig_pca_lumi_regions[r] = fig.create_plot3D(xpos=7.2 + r * 4, ypos=3.5, plot_height=1.7, plot_width=1.7,
                                                       xmin=-1.1 * xmax, xmax=1.1 * xmax, ymin=-1.1 * ymax,
                                                       ymax=1.1 * ymax,
                                                       zmin=-1.1 * zmax, zmax=1.1 * zmax, helper_lines_dashes=None,
                                                       xyzl_3d=True,
                                                       xticks=[-xmax, -xmax / 2, 0, xmax / 2, xmax],
                                                       yticks=[-ymax, -ymax / 2, 0, ymax / 2, ymax],
                                                       zticks=[-zmax, 0, zmax],
                                                       xticklabels=['', '', '', '', ''],
                                                       yticklabels=['', '', '', '', ''],
                                                       zticklabels=['', '', ''],
                                                       elev=elev, azim=azim)
        # Fig. S5d column 2-4
        subfig_dist_regions[r] = fig.create_plot(xpos=5.5 + r * 4, ypos=0.5, plot_height=2.75, plot_width=3, xmin=0,
                                                 xmax=120, ymin=0,
                                                 ymax=dmax,
                                                 yticks=dtick, vspans=[[20, 80, 'lightgray', 1.0]])
        # Fig. S5c brain region reference in column 2-4
        brain_xy_overview_plot_regions[r] = fig.create_plot(xpos=5.5 + r * 4, ypos=8.3, plot_height=2,
                                                            plot_width=2 / 2.274,
                                                            axis_off=True)
        brain_yz_overview_plot_regions[r] = fig.create_plot(xpos=6.7 + r * 4, ypos=8.3, plot_height=2,
                                                            plot_width=2 / 4.395,
                                                            axis_off=True)

    # Plot the overview of the functional activity (Fig. S5a)
    subplot_traces_overview(traces_df, subfigs_trial_traces, regions_path, regions)
    # Plot the whole-brain PCA analayis (Fig. S5b-d - 1st columns in c-d)
    subplot_pca_wholebrain(traces_df, subfig_expvar, subfig_expvar_zoomin, subfig_pca, subfig_pca_mot, subfig_pca_lumi,
                           subfig_dist, brain_xy_overview_plot, brain_yz_overview_plot)
    # Plot the per-region PCA analysis (Fig. S5c-d - 2-4th columns)
    subplot_pca_per_region(traces_df, regions_path, regions, subfig_pca_regions, subfig_pca_mot_regions,
                           subfig_pca_lumi_regions, subfig_dist_regions, brain_xy_overview_plot_regions,
                           brain_yz_overview_plot_regions)

    fig.save(fig_save_path)

