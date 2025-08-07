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

    region_masks = create_combined_region_npy_mask(regions_path, regions=regions)
    region_ids = region_masks[traces_df['ZB_x'].astype(int), traces_df['ZB_y'].astype(int), traces_df['ZB_z'].astype(int)]

    n_neurons = np.sum(region_ids > 0)

    for stim, subfig in zip(['lumi_left_dots_left',  'lumi_left_dots_right',  'lumi_left_dots_off',
                             'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                             'lumi_off_dots_left',   'lumi_off_dots_right',   'lumi_off_dots_off'],
                            subfigs):
        for t, x_offset, y_offset in zip(range(3), [0, 25, 50], [0, 25000, 50000]):
            region_offset = 0
            for region_id in range(len(regions) + 1):
                if region_id == 0:
                    continue
                region_df = traces_df[region_ids == region_id]
                trial = [region_df[f'{stim}_trial_{t}_trace_{i}'].astype(float) for i in range(120)]
                trial = (trial - np.nanmin(trial, axis=0)) / (np.nanmax(trial, axis=0) - np.nanmin(trial, axis=0))
                subfig.draw_image(np.array(trial).reshape(120, len(region_df)).T, colormap='viridis',
                                  extent=((50-x_offset), 120+(50-x_offset), len(region_df)+y_offset+region_offset, y_offset+region_offset), image_origin='upper', rasterized=True)
                region_offset += len(region_df)
            region_offset = 0
            for region_id in range(len(regions) + 1):
                if region_id == 0:
                    continue
                region_df = traces_df[region_ids == region_id]
                subfig.draw_line([50 - x_offset, 120 + (50 - x_offset)],
                                 [len(region_df) + y_offset + region_offset, len(region_df) + y_offset + region_offset], lc='w', lw=0.75)
                region_offset += len(region_df)
            subfig.draw_line([(50-x_offset), 120+(50-x_offset), 120+(50-x_offset), (50-x_offset), (50-x_offset)],
                             [n_neurons+y_offset, n_neurons+y_offset, y_offset, y_offset, n_neurons+y_offset], lc='w', lw=0.5)

    subfigs[-1].draw_line([50+95, 50+115], [135000, 135000], lc='k')
    subfigs[-1].draw_text(50+105, 150000, '10s')
    return

def try_out_3d_rotation(traces_df):
    print(len(traces_df))

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

    all_traces[np.isnan(all_traces)] = 0
    all_traces = (all_traces - np.nanmin(all_traces)) / (np.nanmax(all_traces) - np.nanmin(all_traces))

    pca = PCA(n_components=np.min(all_traces.shape))
    out = pca.fit_transform(all_traces)

    print(np.sum(pca.explained_variance_ratio_[:50]))
    print(np.sum(pca.explained_variance_ratio_[:30]))
    print(np.sum(pca.explained_variance_ratio_[:3]))

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

    print('ELEV and AZIM')
    print(ax1.elev)
    print(ax1.azim)

    print('MAX sizes PCA')
    print(np.nanmin(out[:, :3], axis=0))
    print(np.nanmax(out[:, :3], axis=0))

    out_red = out[:, :3]
    print(np.sum(pca.explained_variance_ratio_[:3]))

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


    fig, ax = plt.subplots(1, 1)
    ax.fill_between(np.arange(120), np.nanmean(ctrl_stim_distances, axis=0) - np.nanstd(ctrl_stim_distances, axis=0),
                    np.nanmean(ctrl_stim_distances, axis=0) + np.nanstd(ctrl_stim_distances, axis=0), color='gray', alpha=0.5)
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


def subplot_pca_wholebrain(traces_df, subfig_expvar, subfig_expvar_zoomin, subfig_pca, subfig_pca_mot, subfig_pca_lumi, subfig_dist, brain_xy_overview_plot, brain_yz_overview_plot):
    print(len(traces_df))

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

    all_traces[np.isnan(all_traces)] = 0
    all_traces = (all_traces - np.nanmin(all_traces)) / (np.nanmax(all_traces) - np.nanmin(all_traces))

    pca = PCA(n_components=np.min(all_traces.shape))
    out = pca.fit_transform(all_traces)

    subfig_expvar.draw_line(np.arange(len(pca.explained_variance_ratio_)), np.cumsum(pca.explained_variance_ratio_), lc='tab:blue')
    subfig_expvar_zoomin.draw_line(np.arange(200), np.cumsum(pca.explained_variance_ratio_[:200]), lc='tab:blue')
    print(np.sum(pca.explained_variance_ratio_[:50]))
    print(np.sum(pca.explained_variance_ratio_[:30]))
    print(np.sum(pca.explained_variance_ratio_[:3]))

    pca_colors = ['g', 'g', 'g', 'm', 'm', 'm', 'orange', 'orange', 'orange',
                                            'm', 'm', 'm', 'g', 'g', 'g', 'orange', 'orange', 'orange',
                                            'b', 'b', 'b', 'b', 'b', 'b', 'gray', 'gray', 'gray']
    pca_mot_colors = ['darkgreen', 'darkgreen', 'darkgreen', 'springgreen', 'springgreen', 'springgreen', 'gray', 'gray', 'gray',
                                            'darkgreen', 'darkgreen', 'darkgreen', 'springgreen', 'springgreen', 'springgreen', 'gray', 'gray', 'gray',
                                            'darkgreen', 'darkgreen', 'darkgreen', 'springgreen', 'springgreen', 'springgreen', 'gray', 'gray', 'gray']
    pca_lumi_colors = ['saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown',
                                                'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown',
                                                'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray']

    subfig_pca.draw_scatter3D(out[:, 0], out[:, 1], out[:, 2], ec=None, ps=1,
                              pc=np.repeat(pca_colors, 120))

    subfig_pca_mot.draw_scatter3D(out[:, 0], out[:, 1], out[:, 2], ec=None, ps=1,
                              pc=np.repeat(pca_mot_colors, 120))
    subfig_pca_lumi.draw_scatter3D(out[:, 0], out[:, 1], out[:, 2], ec=None, ps=1,
                              pc=np.repeat(pca_lumi_colors, 120))

    print('MAX sizes PCA')
    print(np.nanmin(out[:, :3], axis=0))
    print(np.nanmax(out[:, :3], axis=0))

    out_red = out[:, :3]
    print(np.sum(pca.explained_variance_ratio_[:3]))

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

    first_trial = True
    stim_idx_a = 360
    stim_idx_b = 1080
    for trial_a, trial_b in zip([0, 120, 240], [0, 120, 240,]):
        if first_trial:
            oppo_stim_distances = [math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :])
                                   for i in range(120)]
            first_trial = False
        else:
            oppo_stim_distances = np.vstack((oppo_stim_distances, [
                math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :]) for i in
                range(120)]))

    subfig_dist.draw_line(np.arange(120), np.nanmean(ctrl_stim_distances, axis=0), yerr=np.nanstd(ctrl_stim_distances, axis=0), lc='#676767',  eafc='#989898', eaalpha=1.0, lw=1, ealw=1, eaec='#989898')

    print(np.max(mot_stim_distances))
    print(np.max(lumi_stim_distances))
    print(np.max(same_stim_distances))
    print(np.max(oppo_stim_distances))
    for t in range(3):
        subfig_dist.draw_line(np.arange(120), mot_stim_distances[t, :], lc='b')
        subfig_dist.draw_line(np.arange(120), lumi_stim_distances[t, :], lc='orange')
        subfig_dist.draw_line(np.arange(120), same_stim_distances[t, :], lc='g')
        subfig_dist.draw_line(np.arange(120), oppo_stim_distances[t, :], lc='m')

    _, pval = ttest_ind(same_stim_distances[:, 20:80].mean(axis=1), oppo_stim_distances[:, 20:80].mean(axis=1))
    _, pvalm = ttest_ind(same_stim_distances[:, 20:80].mean(axis=1), mot_stim_distances[:, 20:80].mean(axis=1))
    _, pvall = ttest_ind(same_stim_distances[:, 20:80].mean(axis=1), lumi_stim_distances[:, 20:80].mean(axis=1))
    print(f'Cohen D: {cohens_d(same_stim_distances[:, 20:80].mean(axis=1), oppo_stim_distances[:, 20:80].mean(axis=1))}')
    print(f'Cohen D: {cohens_d(same_stim_distances[:, 20:80].mean(axis=1), mot_stim_distances[:, 20:80].mean(axis=1))}')
    print(f'Cohen D: {cohens_d(same_stim_distances[:, 20:80].mean(axis=1), lumi_stim_distances[:, 20:80].mean(axis=1))}')
    print(f'pval: {pval}')
    print(f'pval: {pvalm}')
    print(f'pval: {pvall}')

    if pval < 0.001:
        subfig_dist.draw_text(50, 1.1 * np.max([same_stim_distances.max(), oppo_stim_distances.max()]), '***')
    elif pval < 0.01:
        subfig_dist.draw_text(50, 1.1 * np.max([same_stim_distances.max(), oppo_stim_distances.max()]), '**')
    elif pval < 0.05:
        subfig_dist.draw_text(50, 1.1 * np.max([same_stim_distances.max(), oppo_stim_distances.max()]), '*')

    total_brain_regions = [navis.read_mesh(r'Z:\Zebrafish atlases\z_brain_atlas\region_masks_mapzebrain_atlas2024\prosencephalon_(forebrain).obj',
                                 units='microns', output='volume'),
                 navis.read_mesh(r'Z:\Zebrafish atlases\z_brain_atlas\region_masks_mapzebrain_atlas2024\mesencephalon_(midbrain).obj',
                                 units='microns', output='volume'),
                 navis.read_mesh(r'Z:\Zebrafish atlases\z_brain_atlas\region_masks_mapzebrain_atlas2024\rhombencephalon_(hindbrain).obj',
                                 units='microns', output='volume')]

    brain_xy_overview_plot.draw_navis_neuron(None, total_brain_regions, navis_color='gray', navis_view=('x', '-y'), lw=0.5, rasterized=True)
    brain_yz_overview_plot.draw_navis_neuron(None, total_brain_regions, navis_color='gray', navis_view=('z', '-y'), lw=0.5, rasterized=True)

    return

def subplot_pca_per_region(traces_df, regions_path, regions, subfig_pca_regions, subfig_pca_mot_regions, subfig_pca_lumi_regions, subfig_dist_regions, brain_xy_overview_plot_regions, brain_yz_overview_plot_regions):
    print(len(traces_df))
    region_masks = create_combined_region_npy_mask(regions_path, regions=regions)
    region_ids = region_masks[
        traces_df['ZB_x'].astype(int), traces_df['ZB_y'].astype(int), traces_df['ZB_z'].astype(int)]

    print(len(traces_df))
    for region_id in range(len(regions) + 1):
        if region_id == 0:
            continue
        else:
            print(regions[region_id - 1])

        region_df = traces_df[region_ids == region_id]

        print(len(region_df))
        for stim_idx, stim_name in enumerate(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off',
                                              'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                                              'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off']):
            if stim_idx == 0:
                all_traces = np.vstack(([region_df[f'{stim_name}_trial_0_trace_{i}'].astype(float) for i in range(120)],
                                        [region_df[f'{stim_name}_trial_1_trace_{i}'].astype(float) for i in range(120)],
                                        [region_df[f'{stim_name}_trial_2_trace_{i}'].astype(float) for i in range(120)]))
            else:
                all_traces = np.vstack((all_traces,
                                        [region_df[f'{stim_name}_trial_0_trace_{i}'].astype(float) for i in range(120)],
                                        [region_df[f'{stim_name}_trial_1_trace_{i}'].astype(float) for i in range(120)],
                                        [region_df[f'{stim_name}_trial_2_trace_{i}'].astype(float) for i in range(120)]))

        all_traces[np.isnan(all_traces)] = 0
        all_traces = (all_traces - np.nanmin(all_traces)) / (np.nanmax(all_traces) - np.nanmin(all_traces))

        pca = PCA(n_components=np.min(all_traces.shape))
        out = pca.fit_transform(all_traces)

        print(np.sum(pca.explained_variance_ratio_[:50]))
        print(np.sum(pca.explained_variance_ratio_[:30]))
        print(np.sum(pca.explained_variance_ratio_[:3]))

        pca_colors = ['g', 'g', 'g', 'm', 'm', 'm', 'orange', 'orange', 'orange',
                                                'm', 'm', 'm', 'g', 'g', 'g', 'orange', 'orange', 'orange',
                                                'b', 'b', 'b', 'b', 'b', 'b', 'gray', 'gray', 'gray']
        pca_mot_colors = ['darkgreen', 'darkgreen', 'darkgreen', 'springgreen', 'springgreen', 'springgreen', 'gray', 'gray', 'gray',
                                                'darkgreen', 'darkgreen', 'darkgreen', 'springgreen', 'springgreen', 'springgreen', 'gray', 'gray', 'gray',
                                                'darkgreen', 'darkgreen', 'darkgreen', 'springgreen', 'springgreen', 'springgreen', 'gray', 'gray', 'gray']
        pca_lumi_colors = ['saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown', 'saddlebrown',
                                                    'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown', 'sandybrown',
                                                    'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray']

        subfig_pca_regions[region_id-1].draw_scatter3D(out[:, 0], out[:, 1], out[:, 2], ec=None, ps=1,
                                  pc=np.repeat(pca_colors, 120))

        subfig_pca_mot_regions[region_id-1].draw_scatter3D(out[:, 0], out[:, 1], out[:, 2], ec=None, ps=1,
                                  pc=np.repeat(pca_mot_colors, 120))
        subfig_pca_lumi_regions[region_id-1].draw_scatter3D(out[:, 0], out[:, 1], out[:, 2], ec=None, ps=1,
                                  pc=np.repeat(pca_lumi_colors, 120))

        print('MAX sizes PCA')
        print(np.nanmin(out[:, :3], axis=0))
        print(np.nanmax(out[:, :3], axis=0))

        out_red = out[:, :3]
        print(np.sum(pca.explained_variance_ratio_[:3]))

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

        first_trial = True
        stim_idx_a = 360
        stim_idx_b = 1080
        for trial_a, trial_b in zip([0, 120, 240], [0, 120, 240]):
            if first_trial:
                oppo_stim_distances = [math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :])
                                       for i in range(120)]
                first_trial = False
            else:
                oppo_stim_distances = np.vstack((oppo_stim_distances, [
                    math.dist(out_red[stim_idx_a + trial_a + i, :], out_red[stim_idx_b + trial_b + i, :]) for i in
                    range(120)]))

        subfig_dist_regions[region_id-1].draw_line(np.arange(120), np.nanmean(ctrl_stim_distances, axis=0), yerr=np.nanstd(ctrl_stim_distances, axis=0), lc='#676767',  eafc='#989898', eaalpha=1.0, lw=1, ealw=1, eaec='#989898')

        print(np.max(mot_stim_distances))
        print(np.max(lumi_stim_distances))
        print(np.max(same_stim_distances))
        print(np.max(oppo_stim_distances))
        for t in range(3):
            subfig_dist_regions[region_id-1].draw_line(np.arange(120), mot_stim_distances[t, :], lc='b')
            subfig_dist_regions[region_id-1].draw_line(np.arange(120), lumi_stim_distances[t, :], lc='orange')
            subfig_dist_regions[region_id-1].draw_line(np.arange(120), same_stim_distances[t, :], lc='g')
            subfig_dist_regions[region_id-1].draw_line(np.arange(120), oppo_stim_distances[t, :], lc='m')
        if region_id == len(regions):
            subfig_dist_regions[region_id-1].draw_line([100, 120], [0.01, 0.01], lc='k')
            subfig_dist_regions[region_id-1].draw_text(110, -0.15, '10s')

        _, pval = ttest_ind(same_stim_distances[:, 20:80].mean(axis=1), oppo_stim_distances[:, 20:80].mean(axis=1))
        _, pvalm = ttest_ind(same_stim_distances[:, 20:80].mean(axis=1), mot_stim_distances[:, 20:80].mean(axis=1))
        _, pvall = ttest_ind(same_stim_distances[:, 20:80].mean(axis=1), lumi_stim_distances[:, 20:80].mean(axis=1))
        print(f'Cohen D: {cohens_d(same_stim_distances[:, 20:80].mean(axis=1), oppo_stim_distances[:, 20:80].mean(axis=1))}')
        print(f'Cohen D: {cohens_d(same_stim_distances[:, 20:80].mean(axis=1), mot_stim_distances[:, 20:80].mean(axis=1))}')
        print(f'Cohen D: {cohens_d(same_stim_distances[:, 20:80].mean(axis=1), lumi_stim_distances[:, 20:80].mean(axis=1))}')
        print(f'pval: {pval}')
        print(f'pval: {pvalm}')
        print(f'pval: {pvall}')

        if pval < 0.001:
            subfig_dist_regions[region_id-1].draw_text(50, 1.1 * np.max([same_stim_distances.max(), oppo_stim_distances.max()]), '***')
        elif pval < 0.01:
            subfig_dist_regions[region_id-1].draw_text(50, 1.1 * np.max([same_stim_distances.max(), oppo_stim_distances.max()]), '**')
        elif pval < 0.05:
            subfig_dist_regions[region_id-1].draw_text(50, 1.1 * np.max([same_stim_distances.max(), oppo_stim_distances.max()]), '*')

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
            units='microns', output='volume'),]

        brain_xy_overview_plot_regions[region_id - 1].draw_navis_neuron(None, total_brain_regions, navis_view=('x', '-y'), lw=0.5, rasterized=True)
        brain_xy_overview_plot_regions[region_id - 1].draw_navis_neuron(None, brain_regions, navis_color='gray', navis_view=('x', '-y'), lw=0.5, rasterized=True)
        brain_yz_overview_plot_regions[region_id - 1].draw_navis_neuron(None, total_brain_regions, navis_view=('z', '-y'), lw=0.5, rasterized=True)
        brain_yz_overview_plot_regions[region_id - 1].draw_navis_neuron(None, brain_regions, navis_color='gray', navis_view=('z', '-y'), lw=0.5, rasterized=True)

    return


if __name__ == '__main__':
    print('Loading traces')
    traces_df = pd.read_csv(r'C:\Users\Katja\Desktop\imaging_traces_trials_baseline.csv')
    print('Traces loaded')
    regions_path = r'Z:\Zebrafish atlases\z_brain_atlas\region_masks_mapzebrain_atlas2024\all_masks_indexed.hdf5'
    regions = ['prosencephalon_(forebrain)', 'mesencephalon_(midbrain)', 'rhombencephalon_(hindbrain)']
    try_out_and_set_3d_orientation_angles = False

    if try_out_and_set_3d_orientation_angles:
        region_masks = create_combined_region_npy_mask(regions_path, regions=regions)
        region_ids = region_masks[traces_df['ZB_x'].astype(int), traces_df['ZB_y'].astype(int), traces_df['ZB_z'].astype(int)]

        print(len(traces_df))
        for region_id in range(len(regions) + 1):
            if region_id < 1:
                print('Rest of the brain')
            else:
                print(regions[region_id - 1])

            region_df = traces_df[region_ids == region_id]
            try_out_3d_rotation(region_df)

    fig = Figure(fig_width=18, fig_height=17)
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

    subfig_expvar = fig.create_plot(xpos=15, ypos=11.5, plot_height=2, plot_width=2, xmin=0, xmax=3240, ymin=0, ymax=1,
                                    xl='PC', yl='cum. explained (%)', xticks=[0, 1000, 2000, 3000],
                                    yticks=[0, 0.25, 0.5, 0.75, 1],
                                    yticklabels=['0', '25', '50', '75', '100'])
    subfig_expvar_zoomin = fig.create_plot(xpos=15, ypos=14.5, plot_height=2, plot_width=2, xmin=0, xmax=200, ymin=0,
                                           ymax=1, yl='cum. explained (%)', xticks=[0, 50, 100, 150, 200],
                                           yticks=[0, 0.25, 0.5, 0.75, 1],
                                           yticklabels=['0', '25', '50', '75', '100'])

    subfig_pca = fig.create_plot3D(xpos=1, ypos=5.5, plot_height=3.4, plot_width=3.4, xmin=-1.1*0.4, xmax=1.1*0.4, ymin=-1.1*0.4,
                                                   ymax=1.1*0.4,
                                                   zmin=-1.1*0.4, zmax=1.1*0.4, helper_lines_dashes=None, xyzl_3d=True,
                                   xticks=[-0.4, -0.2, 0, 0.2, 0.4], yticks=[-0.4, -0.2, 0, 0.2, 0.4], zticks=[-0.4, 0, 0.4],
                                   xticklabels=['', '', '', '', ''], yticklabels=['', '', '', '', ''], zticklabels=['', '', ''],
                                   xl_distance=0, yl_distance=0, zl_distance=0,
                                   elev=52.22, azim=-38.07, xl='PC1', yl='PC2', zl='PC3')
    subfig_pca_mot = fig.create_plot3D(xpos=1, ypos=3.5, plot_height=1.7, plot_width=1.7, xmin=-1.1*0.4, xmax=1.1*0.4, ymin=-1.1*0.4,
                                                   ymax=1.1*0.4,
                                                   zmin=-1.1*0.4, zmax=1.1*0.4,  helper_lines_dashes=None, xyzl_3d=True,
                                   xticks=[-0.4, -0.2, 0, 0.2, 0.4], yticks=[-0.4, -0.2, 0, 0.2, 0.4], zticks=[-0.4, 0, 0.4],
                                   xticklabels=['', '', '', '', ''], yticklabels=['', '', '', '', ''], zticklabels=['', '', ''],
                                       elev=52.22, azim=-38.07)
    subfig_pca_lumi = fig.create_plot3D(xpos=2.7, ypos=3.5, plot_height=1.7, plot_width=1.7, xmin=-1.1*0.4, xmax=1.1*0.4, ymin=-1.1*0.4,
                                                   ymax=1.1*0.4,
                                                   zmin=-1.1*0.4, zmax=1.1*0.4,  helper_lines_dashes=None, xyzl_3d=True,
                                   xticks=[-0.4, -0.2, 0, 0.2, 0.4], yticks=[-0.4, -0.2, 0, 0.2, 0.4], zticks=[-0.4, 0, 0.4],
                                   xticklabels=['', '', '', '', ''], yticklabels=['', '', '', '', ''], zticklabels=['', '', ''],
                                        elev=52.22, azim=-38.07)

    subfig_dist = fig.create_plot(xpos=1, ypos=0.5, plot_height=2.75, plot_width=3.4, xmin=0, xmax=120, ymin=0, ymax=1.0,
                                  yticks=[0.0, 0.5, 1.0,], vspans=[[20, 80, 'lightgray', 1.0]],
                                  yl='distance in PC1-3')

    brain_xy_overview_plot = fig.create_plot(xpos=1., ypos=8.3, plot_height=2, plot_width=2/2.274, axis_off=True)
    brain_yz_overview_plot = fig.create_plot(xpos=2.2, ypos=8.3, plot_height=2, plot_width=2/4.395, axis_off=True)

    subfig_pca_regions = [[]] * len(regions)
    subfig_pca_mot_regions = [[]] * len(regions)
    subfig_pca_lumi_regions = [[]] * len(regions)
    subfig_dist_regions = [[]] * len(regions)
    brain_xy_overview_plot_regions = [[]] * len(regions)
    brain_yz_overview_plot_regions = [[]] * len(regions)
    elevs = [25.05, 33.32, 11.47]
    azims = [159.66, 33.94, 73.93]
    xmaxs = [0.5, 2.1, 2.3]
    ymaxs = [0.3, 1.8, 2.7]
    zmaxs = [0.3, 1.6, 2.5]
    dmaxs = [0.6, 4, 5.0]
    dticks = [[0.0, 0.2, 0.4, 0.6], [0, 2, 4], [0, 2, 4]]
    for r, elev, azim, xmax, ymax, zmax, dmax, dtick in zip(range(len(regions)), elevs, azims, xmaxs, ymaxs, zmaxs, dmaxs, dticks):
        subfig_pca_regions[r] = fig.create_plot3D(xpos=5.5 + r * 4, ypos=5.5, plot_height=3.4, plot_width=3.4, xmin=-1.1*xmax, xmax=1.1*xmax, ymin=-1.1*ymax,
                                                   ymax=1.1*ymax,
                                                   zmin=-1.1*zmax, zmax=1.1*zmax, helper_lines_dashes=None, xyzl_3d=True,
                                                   xticks=[-xmax, -xmax/2, 0, xmax/2, xmax], yticks=[-ymax, -ymax/2, 0, ymax/2, ymax],
                                                   zticks=[-zmax, 0, zmax],
                                                   xticklabels=['', '', '', '', ''], yticklabels=['', '', '', '', ''],
                                                   zticklabels=['', '', ''],
                                                   xl_distance=0, yl_distance=0, zl_distance=0,
                                                   elev=elev, azim=azim, xl='PC1', yl='PC2', zl='PC3')
        subfig_pca_mot_regions[r] = fig.create_plot3D(xpos=5.5 + r * 4, ypos=3.5, plot_height=1.7, plot_width=1.7, xmin=-1.1*xmax, xmax=1.1*xmax, ymin=-1.1*ymax,
                                                   ymax=1.1*ymax,
                                                   zmin=-1.1*zmax, zmax=1.1*zmax, helper_lines_dashes=None, xyzl_3d=True,
                                                   xticks=[-xmax, -xmax/2, 0, xmax/2, xmax], yticks=[-ymax, -ymax/2, 0, ymax/2, ymax],
                                                   zticks=[-zmax, 0, zmax],
                                                   xticklabels=['', '', '', '', ''], yticklabels=['', '', '', '', ''],
                                                   zticklabels=['', '', ''],
                                                   elev=elev, azim=azim)
        subfig_pca_lumi_regions[r] = fig.create_plot3D(xpos=7.2 + r * 4, ypos=3.5, plot_height=1.7, plot_width=1.7, xmin=-1.1*xmax, xmax=1.1*xmax, ymin=-1.1*ymax,
                                                   ymax=1.1*ymax,
                                                   zmin=-1.1*zmax, zmax=1.1*zmax, helper_lines_dashes=None, xyzl_3d=True,
                                                   xticks=[-xmax, -xmax/2, 0, xmax/2, xmax], yticks=[-ymax, -ymax/2, 0, ymax/2, ymax],
                                                   zticks=[-zmax, 0, zmax],
                                                    xticklabels=['', '', '', '', ''], yticklabels=['', '', '', '', ''],
                                                    zticklabels=['', '', ''],
                                                    elev=elev, azim=azim)

        subfig_dist_regions[r] = fig.create_plot(xpos=5.5 + r * 4, ypos=0.5, plot_height=2.75, plot_width=3, xmin=0, xmax=120, ymin=0,
                                              ymax=dmax,
                                              yticks=dtick, vspans=[[20, 80, 'lightgray', 1.0]])

        brain_xy_overview_plot_regions[r] = fig.create_plot(xpos=5.5 + r * 4, ypos=8.3, plot_height=2, plot_width=2 / 2.274,
                                                 axis_off=True)
        brain_yz_overview_plot_regions[r] = fig.create_plot(xpos=6.7 + r * 4, ypos=8.3, plot_height=2, plot_width=2 / 4.395,
                                                 axis_off=True)

    subplot_traces_overview(traces_df, subfigs_trial_traces, regions_path, regions)

    subplot_pca_wholebrain(traces_df, subfig_expvar, subfig_expvar_zoomin, subfig_pca, subfig_pca_mot, subfig_pca_lumi, subfig_dist, brain_xy_overview_plot, brain_yz_overview_plot)
    subplot_pca_per_region(traces_df, regions_path, regions, subfig_pca_regions, subfig_pca_mot_regions, subfig_pca_lumi_regions, subfig_dist_regions, brain_xy_overview_plot_regions, brain_yz_overview_plot_regions)

    fig.save('C:/users/katja/Desktop/figS3.pdf')

