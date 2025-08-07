import h5py
import numpy as np
from pathlib import Path
import pandas as pd
from multifeature_integration_paper.figure_helper import Figure
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel
from multifeature_integration_paper.useful_small_funcs import cohens_d

def additive_model_coen(input, vr, vl, ar, al, b, gamma):
    VR, VL, AR, AL = input
    x = vr * VR ** gamma - vl * VL ** gamma + ar * AR - al * AL + b
    pL = 1 / (1 + np.exp(-x))
    return pL

def strongest_stim_model(input, vr, vl, ar, al, b, gamma):
    VR, VL, AR, AL = input
    x = b * np.ones(input.shape[1])
    strongest_stim = np.argmax(np.array(np.array([-vr*VR**gamma, -vl*VL**gamma, ar*AR, al*AL])), axis=0)
    x[strongest_stim == 0] += vr * VR[strongest_stim == 0] ** gamma
    x[strongest_stim == 1] -= vl * VL[strongest_stim == 1] ** gamma
    x[strongest_stim == 2] += ar * AR[strongest_stim == 2]
    x[strongest_stim == 3] -= al * AL[strongest_stim == 3]

    pL = 1 / (1 + np.exp(-x))
    return pL

def motion_only_model(input, vr, vl, ar, al, b, gamma):
    VR, VL, AR, AL = input
    x = vr * VR ** gamma - vl * VL ** gamma + b
    pL = 1 / (1 + np.exp(-x))
    return pL

def lumi_only_model(input, vr, vl, ar, al, b, gamma):
    VR, VL, AR, AL = input
    x = ar * AR - al * AL + b
    pL = 1 / (1 + np.exp(-x))
    return pL

def sub_fig_raw_trace(path_to_raw_data, subfig):
    raw_data = np.array(h5py.File(path_to_raw_data)['repeat00/raw_data/fish00/freely_swimming_tracking_data'])

    subfig.draw_line(raw_data[1420:1650, 0]-raw_data[0, 0], raw_data[1420:1650, 5], lc='tab:blue')
    subfig.draw_line([18.1, 18.3], [775, 775], lc='k')
    subfig.draw_text(18.2, 760, '200 ms')
    subfig.draw_line([16.845, 16.845, 17.7, 17.7], [780, 775, 775, 780], lc='k')
    subfig.draw_text(17.2725, 760, 'IBI')
    subfig.draw_line([16.12, 16.1, 16.1, 16.12], [775, 775, 796, 796], lc='k')
    subfig.draw_text(15.9, 785.5, '\u0394 \u03b1')
    return

def sub_fig_bout_distributions(path_to_combined_data, subfiga, subfigb):
    combined_df = pd.read_hdf(path_to_combined_data, key='all_bout_data_pandas')

    hist_values = np.histogram(combined_df['estimated_orientation_change'], bins=np.arange(-100, 102, 2), density=True)
    subfiga.draw_vertical_bars(hist_values[1][:-1]+1, hist_values[0], lc='tab:blue')
    subfiga.draw_text(-75, 0.035, 'Left bouts')
    subfiga.draw_text(75, 0.035, 'Right bouts')

    hist_values = np.histogram(combined_df['interbout_interval'], bins=np.arange(0, 4, 0.05), density=True)
    subfigb.draw_vertical_bars(hist_values[1][:-1]+0.025, hist_values[0], lc='tab:blue')
    return

def get_model_inputs():
    x_input = [-100, -50, -25, -10, 0, 10, 25, 50, 100]
    x_input_fine = np.arange(-100, 101, 2)

    model_in_left = [1, 0.5, 0.25, 0.1, 0, 0, 0, 0, 0]
    model_in_right = [0, 0, 0, 0, 0, 0.1, 0.25, 0.5, 1]

    model_in_left_fine = np.arange(1, -1.01, -0.02)
    model_in_left_fine[model_in_left_fine < 0] = 0
    model_in_right_fine = np.arange(-1, 1.01, 0.02)
    model_in_right_fine[model_in_right_fine < 0] = 0

    return x_input, x_input_fine, model_in_left, model_in_right, model_in_left_fine, model_in_right_fine

def reorder_analysed_df(path_to_analysed_data):
    analysed_df = pd.read_hdf(path_to_analysed_data)

    # Sort full data
    sorted_stim_names = analysed_df.index.unique('stimulus_name').sort_values()

    # Stimuli are alphabetic by default, this order sets them back to -100 to 100 order.
    reorder_order = [0, 1, 2, 9, 10, 11, 6, 7, 8, 3, 4, 5, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 15, 16, 17]
    reorder_stim_names = sorted_stim_names[reorder_order]

    return analysed_df, reorder_stim_names

def sub_fig_model_hypotheses(subfiga, subfigb):
    x_input, x_input_fine, model_in_left, model_in_right, model_in_left_fine, model_in_right_fine = get_model_inputs()

    params_init = [-1, -1, 0.5, 0.5, 0, 0.6]

    input_fine = np.array([model_in_left_fine, model_in_right_fine, np.ones(len(model_in_left_fine)), np.zeros(len(model_in_left_fine))])
    subfiga.draw_line(x=x_input_fine, y=additive_model_coen(input_fine, *params_init), lc='tab:blue', ec='tab:blue')
    subfigb.draw_line(x=x_input_fine, y=strongest_stim_model(input_fine, *params_init), lc='tab:blue', ec='tab:blue', label='Lumi right')

    input_fine = np.array([model_in_left_fine, model_in_right_fine, np.zeros(len(model_in_left_fine)), np.zeros(len(model_in_left_fine))])
    subfiga.draw_line(x=x_input_fine, y=additive_model_coen(input_fine, *params_init), lc='tab:gray', ec='tab:gray')
    subfigb.draw_line(x=x_input_fine, y=strongest_stim_model(input_fine, *params_init), lc='tab:gray', ec='tab:gray', label='Lumi off')

    input_fine = np.array([model_in_left_fine, model_in_right_fine, np.zeros(len(model_in_left_fine)), np.ones(len(model_in_left_fine))])
    subfiga.draw_line(x=x_input_fine, y=additive_model_coen(input_fine, *params_init), lc='tab:red', ec='tab:red')
    subfigb.draw_line(x=x_input_fine, y=strongest_stim_model(input_fine, *params_init), lc='tab:red', ec='tab:red', label='Lumi left')
    return

def sub_fig_data_plus_model_single_fish(path_to_analysed_data, plot_fish_IDs, subfigs, subfig_mse):
    x_input, x_input_fine, model_in_left, model_in_right, model_in_left_fine, model_in_right_fine = get_model_inputs()

    analysed_df, reorder_stim_names = reorder_analysed_df(path_to_analysed_data)

    params_init = [-1, -1, 0.5, 0.5, 0, 0.6]

    all_mses_add = []
    all_mses_wta = []
    all_mses_mot = []
    all_mses_lumi = []
    for single_fish_plot_id, single_fish_ID in enumerate(plot_fish_IDs):
        data_single = []
        for stim_name in reorder_stim_names:
            coherence = int(stim_name.split('_')[0])
            lumi = int(stim_name.split('_')[1])
            stim_df = analysed_df.xs(stim_name, level='stimulus_name')

            if single_fish_ID in stim_df.index.unique('experiment_ID'):
                fish_df = stim_df.xs(single_fish_ID, level='experiment_ID')
                grouped_df_single = fish_df.groupby('window_time')
                mean = np.array(grouped_df_single['percentage_left'].mean())[grouped_df_single['percentage_left'].mean().index == 20][0]
                std = np.array(grouped_df_single['percentage_left'].std())[grouped_df_single['percentage_left'].std().index == 20][0]
                sem = std / np.sqrt(1)

                if lumi == -1:
                    subfigs[single_fish_plot_id].draw_scatter(x=coherence, y=mean, yerr=sem, pc='tab:red', ec='tab:red', lw=0.5)
                elif lumi == 1:
                    subfigs[single_fish_plot_id].draw_scatter(x=coherence, y=mean, yerr=sem, pc='tab:blue', ec='tab:blue', lw=0.5)
                if lumi == 0:
                    subfigs[single_fish_plot_id].draw_scatter(x=coherence, y=mean, yerr=sem, pc='tab:gray', ec='tab:gray', lw=0.5)

                data_single = np.append(data_single, mean)
            else:
                print(f'No data found for selected fish {single_fish_ID} stimulus {stim_name}')

        input = np.array([model_in_left, model_in_right, np.ones(len(model_in_left)), np.ones(len(model_in_left))])
        input_fine = np.array([model_in_left_fine, model_in_right_fine, np.ones(len(model_in_left_fine)), np.ones(len(model_in_left_fine))])

        popt, pcov = curve_fit(additive_model_coen, input, data_single[::3], p0=params_init)

        subfigs[single_fish_plot_id].draw_line(x=x_input_fine, y=additive_model_coen(input_fine, *popt), lc='tab:red', ec='tab:red')

        input = np.array([model_in_left, model_in_right, np.zeros(len(model_in_left)), np.zeros(len(model_in_left))])
        input_fine = np.array([model_in_left_fine, model_in_right_fine, np.zeros(len(model_in_left_fine)), np.zeros(len(model_in_left_fine))])

        popt, pcov = curve_fit(additive_model_coen, input, data_single[1::3], p0=params_init)

        subfigs[single_fish_plot_id].draw_line(x=x_input_fine, y=additive_model_coen(input_fine, *popt), lc='tab:gray', ec='tab:gray')

        input = np.array([model_in_left, model_in_right, np.ones(len(model_in_left)), np.zeros(len(model_in_left))])
        input_fine = np.array([model_in_left_fine, model_in_right_fine, np.ones(len(model_in_left_fine)), np.zeros(len(model_in_left_fine))])
        popt, pcov = curve_fit(additive_model_coen, input, data_single[2::3], p0=params_init)

        subfigs[single_fish_plot_id].draw_line(x=x_input_fine, y=additive_model_coen(input_fine, *popt), lc='tab:blue', ec='tab:blue')
        subfigs[single_fish_plot_id].draw_text(-120, 1, f'example fish {single_fish_plot_id+1}', textlabel_ha='left')

    plot_fish_counter = []
    for loop_counter, single_fish_ID in enumerate(analysed_df.index.unique('experiment_ID')):
        if single_fish_ID in plot_fish_IDs:
            plot_fish_counter = np.append(plot_fish_counter, loop_counter)
        data_single = []
        for stim_name in reorder_stim_names:
            stim_df = analysed_df.xs(stim_name, level='stimulus_name')

            if single_fish_ID in stim_df.index.unique('experiment_ID'):
                fish_df = stim_df.xs(single_fish_ID, level='experiment_ID')
                grouped_df_single = fish_df.groupby('window_time')
                mean = np.array(grouped_df_single['percentage_left'].mean())[
                    grouped_df_single['percentage_left'].mean().index == 20][0]

                data_single = np.append(data_single, mean)
            else:
                print(f'No data found for selected fish {single_fish_ID} stimulus {stim_name}')

        input = np.array([model_in_left, model_in_right, np.zeros(len(model_in_left)), np.zeros(len(model_in_left))])

        popt, pcov = curve_fit(additive_model_coen, input, data_single[1::3], p0=params_init)
        mse_o = np.nanmean(np.square(additive_model_coen(input, *popt) - data_single[1::3]))
        popt_2, pcov_2 = curve_fit(strongest_stim_model, input, data_single[1::3], p0=params_init)
        mse_o_2 = np.nanmean(np.square(strongest_stim_model(input, *popt_2) - data_single[1::3]))
        popt_3, pcov_3 = curve_fit(motion_only_model, input, data_single[1::3], p0=params_init)
        mse_o_3 = np.nanmean(np.square(motion_only_model(input, *popt_3) - data_single[1::3]))
        popt_4, pcov_4 = curve_fit(lumi_only_model, input, data_single[1::3], p0=params_init)
        mse_o_4 = np.nanmean(np.square(lumi_only_model(input, *popt_4) - data_single[1::3]))

        input = np.array([model_in_left, model_in_right, np.ones(len(model_in_left)), np.ones(len(model_in_left))])

        try:
            popt, pcov = curve_fit(additive_model_coen, input, data_single[::3], p0=params_init)
            mse_r = np.nanmean(np.square(additive_model_coen(input, *popt) - data_single[::3]))
        except:
            mse_r = np.nan
        popt_2, pcov_2 = curve_fit(strongest_stim_model, input, data_single[::3], p0=params_init)
        mse_r_2 = np.nanmean(np.square(strongest_stim_model(input, *popt_2) - data_single[::3]))
        mse_r_3 = np.nanmean(np.square(motion_only_model(input, *popt_3) - data_single[::3]))
        popt_4, pcov_4 = curve_fit(lumi_only_model, input, data_single[::3], p0=params_init)
        mse_r_4 = np.nanmean(np.square(lumi_only_model(input, *popt_4) - data_single[::3]))

        input = np.array([model_in_left, model_in_right, np.ones(len(model_in_left)), np.zeros(len(model_in_left))])

        popt, pcov = curve_fit(additive_model_coen, input, data_single[2::3], p0=params_init)
        mse_l = np.nanmean(np.square(additive_model_coen(input, *popt) - data_single[2::3]))
        try:
            popt_2, pcov_2 = curve_fit(strongest_stim_model, input, data_single[2::3], p0=params_init)
            mse_l_2 = np.nanmean(np.square(strongest_stim_model(input, *popt_2) - data_single[2::3]))
        except:
            mse_l_2 = np.nan
        mse_l_3 = np.nanmean(np.square(motion_only_model(input, *popt_3) - data_single[2::3]))
        popt_4, pcov_4 = curve_fit(lumi_only_model, input, data_single[2::3], p0=params_init)
        mse_l_4 = np.nanmean(np.square(lumi_only_model(input, *popt_4) - data_single[2::3]))

        if np.nanmean([mse_r, mse_o, mse_l]) > np.nanmean([mse_r_2, mse_o_2, mse_l_2]) or np.nanmean([mse_r, mse_o, mse_l]) > np.nanmean([mse_r_3, mse_o_3, mse_l_3]) or np.nanmean([mse_r, mse_o, mse_l]) > np.nanmean([mse_r_4, mse_o_4, mse_l_4]):
            print('The additive model is not the best fit for fish ', single_fish_ID)
            print('MSE add, MSE wta, MSE mot, MSE lumi:')
            print(np.nanmean([mse_r, mse_o, mse_l]), np.nanmean([mse_r_2, mse_o_2, mse_l_2]), np.nanmean([mse_r_3, mse_o_3, mse_l_3]), np.nanmean([mse_r_4, mse_o_4, mse_l_4]))

        all_mses_add = np.append(all_mses_add, np.nanmean([mse_r, mse_o, mse_l]))
        all_mses_wta = np.append(all_mses_wta, np.nanmean([mse_r_2, mse_o_2, mse_l_2]))
        all_mses_mot = np.append(all_mses_mot, np.nanmean([mse_r_3, mse_o_3, mse_l_3]))
        all_mses_lumi = np.append(all_mses_lumi, np.nanmean([mse_r_4, mse_o_4, mse_l_4]))

        subfig_mse.draw_line([0, 1, 2, 3], [np.nanmean([mse_r, mse_o, mse_l]), np.nanmean([mse_r_2, mse_o_2, mse_l_2]), np.nanmean([mse_r_3, mse_o_3, mse_l_3]), np.nanmean([mse_r_4, mse_o_4, mse_l_4])], lc='darkgray', lw=0.25)

    for single_fish_id, ld in zip(plot_fish_counter.astype(int), [(1, 3), (2, 3), (3, 3), (4, 3)]):
        subfig_mse.draw_line([0, 1, 2, 3], [all_mses_add[single_fish_id], all_mses_wta[single_fish_id], all_mses_mot[single_fish_id], all_mses_lumi[single_fish_id]], lc='dimgray',
                              line_dashes=ld, lw=0.75)
    subfig_mse.draw_line([0, 1, 2, 3], [np.nanmean(all_mses_add), np.nanmean(all_mses_wta), np.nanmean(all_mses_mot), np.nanmean(all_mses_lumi)], lc='k', lw=1)

    subfig_mse.draw_line([0, 0, 1, 1], [0.11, 0.112, 0.112, 0.11], lc='k')
    _, pval = ttest_rel(all_mses_add, all_mses_wta)
    if pval < 0.001 / 3:
        subfig_mse.draw_text(0.5, 0.115, '***')
    elif pval < 0.01 / 3:
        subfig_mse.draw_text(0.5, 0.115, '**')
    elif pval < 0.05 / 3:
        subfig_mse.draw_text(0.5, 0.115, '*')
    else:
        subfig_mse.draw_text(0.5, 0.115, 'ns')
    effect_size = cohens_d(all_mses_add, all_mses_wta)
    print(f'ADD vs WTA, pval {pval}: Cohen D effect size {effect_size}')

    subfig_mse.draw_line([0, 0, 2, 2], [0.13, 0.132, 0.132, 0.13], lc='k')
    _, pval = ttest_rel(all_mses_add, all_mses_mot)
    if pval < 0.001 / 3:
        subfig_mse.draw_text(1, 0.135, '***')
    elif pval < 0.01 / 3:
        subfig_mse.draw_text(1, 0.135, '**')
    elif pval < 0.05 / 3:
        subfig_mse.draw_text(1, 0.135, '*')
    else:
        subfig_mse.draw_text(1, 0.135, 'ns')
    effect_size = cohens_d(all_mses_add, all_mses_mot)
    print(f'ADD vs MOT, pval {pval}: Cohen D effect size {effect_size}')

    subfig_mse.draw_line([0, 0, 3, 3], [0.15, 0.152, 0.152, 0.15], lc='k')
    _, pval = ttest_rel(all_mses_add, all_mses_lumi)
    if pval < 0.001 / 3:
        subfig_mse.draw_text(1.5, 0.155, '***')
    elif pval < 0.01 / 3:
        subfig_mse.draw_text(1.5, 0.155, '**')
    elif pval < 0.05 / 3:
        subfig_mse.draw_text(1.5, 0.155, '*')
    else:
        subfig_mse.draw_text(1.5, 0.155, 'ns')
    effect_size = cohens_d(all_mses_add, all_mses_lumi)
    print(f'ADD vs LUMI, pval {pval}: Cohen D effect size {effect_size}')

    return

def sub_plot_data_plus_model_all_fish(path_to_analysed_data, subfig):
    x_input, x_input_fine, model_in_left, model_in_right, model_in_left_fine, model_in_right_fine = get_model_inputs()

    analysed_df, reorder_stim_names = reorder_analysed_df(path_to_analysed_data)

    params_init = [-1, -1, 0.5, 0.5, 0, 0.6]

    data = []
    for stim_name in reorder_stim_names:
        coherence = int(stim_name.split('_')[0])
        lumi = int(stim_name.split('_')[1])

        stim_df = analysed_df.xs(stim_name, level='stimulus_name')
        n_fish = len(stim_df.index.unique('experiment_ID'))
        grouped_df = stim_df.groupby('window_time')

        mean = np.array(grouped_df['percentage_left'].mean())[grouped_df['percentage_left'].mean().index == 20][0]
        std = np.array(grouped_df['percentage_left'].std())[grouped_df['percentage_left'].std().index == 20][0]
        sem = std / np.sqrt(n_fish)

        if lumi == -1:
            subfig.draw_scatter(x=coherence, y=mean, yerr=sem, pc='tab:red', ec='tab:red', lw=0.5)
        elif lumi == 1:
            subfig.draw_scatter(x=coherence, y=mean, yerr=sem, pc='tab:blue', ec='tab:blue', lw=0.5)
        if lumi == 0:
            subfig.draw_scatter(x=coherence, y=mean, yerr=sem, pc='tab:gray', ec='tab:gray', lw=0.5)

        data = np.append(data, mean)

    input = np.array([model_in_left, model_in_right, np.ones(len(model_in_left)), np.ones(len(model_in_left))])
    input_fine = np.array([model_in_left_fine, model_in_right_fine, np.ones(len(model_in_left_fine)), np.ones(len(model_in_left_fine))])
    popt, pcov = curve_fit(additive_model_coen, input, data[::3], p0=params_init)
    mse = np.nanmean(np.square(additive_model_coen(input, *popt) - data[::3]))
    popt_2, pcov_2 = curve_fit(strongest_stim_model, input, data[::3], p0=params_init)
    mse_2 = np.nanmean(np.square(strongest_stim_model(input, *popt_2) - data[::3]))
    print(f'BEST MSE: {mse}, {mse_2}')
    print(f'Popt avg -Lumi {popt}')
    subfig.draw_line(x=x_input_fine, y=additive_model_coen(input_fine, *popt), lc='tab:red', ec='tab:red',)

    input = np.array([model_in_left, model_in_right, np.zeros(len(model_in_left)), np.zeros(len(model_in_left))])
    input_fine = np.array([model_in_left_fine, model_in_right_fine, np.zeros(len(model_in_left_fine)), np.zeros(len(model_in_left_fine))])
    popt, pcov = curve_fit(additive_model_coen, input, data[1::3], p0=params_init)
    mse = np.nanmean(np.square(additive_model_coen(input, *popt) - data[1::3]))
    popt_2, pcov_2 = curve_fit(strongest_stim_model, input, data[1::3], p0=params_init)
    mse_2 = np.nanmean(np.square(strongest_stim_model(input, *popt_2) - data[1::3]))
    print(f'BEST MSE: {mse}, {mse_2}')
    subfig.draw_line(x=x_input_fine, y=additive_model_coen(input_fine, *popt), lc='tab:gray', ec='tab:gray',)
    print(f'Popt avg no Lumi {popt}')

    input = np.array([model_in_left, model_in_right, np.ones(len(model_in_left)), np.zeros(len(model_in_left))])
    input_fine = np.array([model_in_left_fine, model_in_right_fine, np.ones(len(model_in_left_fine)), np.zeros(len(model_in_left_fine))])
    popt, pcov = curve_fit(additive_model_coen, input, data[2::3], p0=params_init)
    mse = np.nanmean(np.square(additive_model_coen(input, *popt) - data[2::3]))
    popt_2, pcov_2 = curve_fit(strongest_stim_model, input, data[2::3], p0=params_init)
    mse_2 = np.nanmean(np.square(strongest_stim_model(input, *popt_2) - data[2::3]))
    print(f'BEST MSE: {mse}, {mse_2}')
    subfig.draw_line(x=x_input_fine, y=additive_model_coen(input_fine, *popt), lc='tab:blue', ec='tab:blue',)
    print(f'Popt avg +Lumi {popt}')

    n_fish = len(analysed_df.index.unique('experiment_ID'))
    subfig.draw_text(-60, 0.7, f'{n_fish} fish')
    return

def sub_plot_ibi_orientation_overview(path_to_analysed_data, subfiga, subfigb, plot_fish_IDs):
    analysed_df, reorder_stim_names = reorder_analysed_df(path_to_analysed_data)
    n_fish = len(analysed_df.index.unique('experiment_ID'))

    ibi_data = np.zeros((n_fish, 4))
    ibi_data[:, 0] = np.nanmean([analysed_df.xs(f'25_0', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval'],
                                  analysed_df.xs(f'-25_0', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval']], axis=0)
    ibi_data[:, 1] = np.nanmean([analysed_df.xs(f'0_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval'],
                                analysed_df.xs(f'0_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval']], axis=0)
    ibi_data[:, 2] = np.nanmean([analysed_df.xs(f'25_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval'],
                                analysed_df.xs(f'-25_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval']], axis=0)
    ibi_data[:, 3] = np.nanmean([analysed_df.xs(f'25_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval'],
                                    analysed_df.xs(f'-25_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval']], axis=0)

    pl_data = np.zeros((n_fish, 4))
    pl_data[:, 0] = np.nanmean([analysed_df.xs(f'25_0', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left'],
                                  1 - analysed_df.xs(f'-25_0', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left']], axis=0)
    pl_data[:, 1] = np.nanmean([analysed_df.xs(f'0_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left'],
                                1 - analysed_df.xs(f'0_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left']], axis=0)
    pl_data[:, 2] = np.nanmean([analysed_df.xs(f'25_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left'],
                                1 - analysed_df.xs(f'-25_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left']], axis=0)
    pl_data[:, 3] = np.nanmean([analysed_df.xs(f'-25_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left'],
                                1 - analysed_df.xs(f'25_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left']], axis=0)

    sf_ibi_data = np.zeros((4, 4))
    sf_pl_data = np.zeros((4, 4))
    for fish, single_fish_ID in enumerate(plot_fish_IDs):
        fish_df = analysed_df.xs(single_fish_ID, level='experiment_ID')
        sf_ibi_data[fish, 0] = np.nanmean([fish_df.xs(f'25_0', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval'],
                                      fish_df.xs(f'-25_0', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval']], axis=0)[0]
        sf_ibi_data[fish, 1] = np.nanmean([fish_df.xs(f'0_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval'],
                                    fish_df.xs(f'0_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval']], axis=0)[0]
        sf_ibi_data[fish, 2] = np.nanmean([fish_df.xs(f'25_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval'],
                                    fish_df.xs(f'-25_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval']], axis=0)[0]
        sf_ibi_data[fish, 3] = np.nanmean([fish_df.xs(f'25_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval'],
                                        fish_df.xs(f'-25_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval']], axis=0)[0]

        sf_pl_data[fish, 0] = np.nanmean([fish_df.xs(f'25_0', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left'],
                                      1 - fish_df.xs(f'-25_0', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left']], axis=0)[0]
        sf_pl_data[fish, 1] = np.nanmean([fish_df.xs(f'0_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left'],
                                    1 - fish_df.xs(f'0_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left']], axis=0)[0]
        sf_pl_data[fish, 2] = np.nanmean([fish_df.xs(f'25_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left'],
                                    1 - fish_df.xs(f'-25_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left']], axis=0)[0]
        sf_pl_data[fish, 3] = np.nanmean([fish_df.xs(f'-25_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left'],
                                    1 - fish_df.xs(f'25_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left']], axis=0)[0]

    for fish in range(n_fish):
        subfiga.draw_line([0, 1, 2, 3], ibi_data[fish, :] - np.nanmean(ibi_data[fish, :]), lc='darkgray', lw=0.25)
        subfigb.draw_line([0, 1, 2, 3], pl_data[fish, :], lc='darkgray', lw=0.25)
    for fish, ld in zip(range(4), [(1, 3), (2, 3), (3, 3), (4, 3)]):
        subfiga.draw_line([0, 1, 2, 3], sf_ibi_data[fish, :] - np.nanmean(sf_ibi_data[fish, :]), lc='dimgray', line_dashes=ld, lw=0.75)
        subfigb.draw_line([0, 1, 2, 3], sf_pl_data[fish, :], lc='dimgray', line_dashes=ld, lw=0.75)
    subfiga.draw_line([0, 1, 2, 3], np.nanmedian(ibi_data - np.nanmean(ibi_data, axis=1).reshape(-1, 1), axis=0), lc='k', lw=1)
    subfigb.draw_line([0, 1, 2, 3], np.nanmedian(pl_data, axis=0), lc='k', lw=1)

    subfiga.draw_line([0, 0, 1, 1], [0.54, 0.55, 0.55, 0.54], lc='k')
    _, pval = ttest_rel(ibi_data[:, 0]- np.nanmean(ibi_data, axis=1), ibi_data[:, 1]- np.nanmean(ibi_data, axis=1))
    if pval < 0.001/2:
        subfiga.draw_text(0.5, 0.56, '***')
    elif pval < 0.01/2:
        subfiga.draw_text(0.5, 0.56, '**')
    elif pval < 0.05/2:
        subfiga.draw_text(0.5, 0.56, '*')
    else:
        subfiga.draw_text(0.5, 0.56, 'ns')
    effect_size = cohens_d(ibi_data[:, 0]- np.nanmean(ibi_data, axis=1), ibi_data[:, 1]- np.nanmean(ibi_data, axis=1))
    print(f'IBI0 vs IBI1, pval {pval}: Cohen D effect size {effect_size}')

    subfiga.draw_line([2, 2, 3, 3], [0.54, 0.55, 0.55, 0.54], lc='k')
    _, pval = ttest_rel(ibi_data[:, 2]- np.nanmean(ibi_data, axis=1), ibi_data[:, 3]- np.nanmean(ibi_data, axis=1))
    if pval < 0.001/2:
        subfiga.draw_text(2.5, 0.56, '***')
    elif pval < 0.01/2:
        subfiga.draw_text(2.5, 0.56, '**')
    elif pval < 0.05/2:
        subfiga.draw_text(2.5, 0.56, '*')
    else:
        subfiga.draw_text(2.5, 0.56, 'ns')
    effect_size = cohens_d(ibi_data[:, 2]- np.nanmean(ibi_data, axis=1), ibi_data[:, 3]- np.nanmean(ibi_data, axis=1))
    print(f'IBI2 vs IBI3, pval {pval}: Cohen D effect size {effect_size}')

    subfigb.draw_line([1, 1, 2, 2], [1.14, 1.15, 1.5, 1.14], lc='k')
    _, pval = ttest_rel(pl_data[:, 1], pl_data[:, 2])
    if pval < 0.001/3:
        subfigb.draw_text(1.5, 1.16, '***')
    elif pval < 0.01/3:
        subfigb.draw_text(1.5, 1.16, '**')
    elif pval < 0.05/3:
        subfigb.draw_text(1.5, 1.16, '*')
    else:
        subfigb.draw_text(1.5, 1.16, 'ns')
    effect_size = cohens_d(pl_data[:, 1], pl_data[:, 2])
    print(f'PL1 vs PL2, pval {pval}: Cohen D effect size {effect_size}')

    subfigb.draw_line([0, 0, 2, 2], [1.09, 1.1, 1.1, 1.09], lc='k')
    _, pval = ttest_rel(pl_data[:, 0], pl_data[:, 2])
    if pval < 0.001/3:
        subfigb.draw_text(0.5, 1.11, '***')
    elif pval < 0.01/3:
        subfigb.draw_text(0.5, 1.11, '**')
    elif pval < 0.05/3:
        subfigb.draw_text(0.5, 1.11, '*')
    else:
        subfigb.draw_text(0.5, 1.11, 'ns')
    effect_size = cohens_d(pl_data[:, 0], pl_data[:, 2])
    print(f'PL0 vs PL2, pval {pval}: Cohen D effect size {effect_size}')

    subfigb.draw_line([0, 0, 1, 1], [0.94, 0.95, 0.95, 0.94], lc='k')
    _, pval = ttest_rel(pl_data[:, 0], pl_data[:, 1])
    if pval < 0.001/3:
        subfigb.draw_text(0.5, 0.96, '***')
    elif pval < 0.01/3:
        subfigb.draw_text(0.5, 0.96, '**')
    elif pval < 0.05/3:
        subfigb.draw_text(0.5, 0.96, '*')
    else:
        subfigb.draw_text(0.5, 0.96, 'ns')
    effect_size = cohens_d(pl_data[:, 0], pl_data[:, 1])
    print(f'PL0 vs PL1, pval {pval}: Cohen D effect size {effect_size}')
    return


if __name__ == '__main__':
    path_to_analysed = Path(r'X:\ZT6 Multi-fish behavior setup\Freely swimming larvae\Katja\beh_simultaneous_titration_2\Analysis\data_analysed.hdf5')
    path_to_combined = Path(r'X:\ZT6 Multi-fish behavior setup\Freely swimming larvae\Katja\beh_simultaneous_titration_2\Analysis\data_combined.h5')
    path_to_raw_data = Path(r'X:\ZT6 Multi-fish behavior setup\Freely swimming larvae\Katja\beh_simultaneous_titration_2\2024-10-22_08-47-26_setup2_arena0\2024-10-22_08-47-26_setup2_arena0.hdf5')

    plot_fish_IDs = [107, 110, 88, 95]
    x_input, _, _, _, _, _ = get_model_inputs()

    fig = Figure(fig_width=9, fig_height=17)
    raw_orien_plot = fig.create_plot(xpos=4.75, ypos=15, plot_height=1, plot_width=3.75,
                                    xl='', yl='Orientation (deg)',
                                    yticks=[774, 824], yticklabels=['0', '50'], xmin=15.7, xmax=18.5, ymin=770, ymax=830,
                                     vlines=[16.17, 16.32, 16.845, 16.98, 17.7, 17.84], helper_lines_lc='tab:gray',
                                     helper_lines_dashes=(4, 4))

    orien_distr_plot = fig.create_plot(xpos=1, ypos=13, plot_height=1.25, plot_width=2.5,
                                       xl='Orientation change (deg/bout)', xmin=-110, xmax=110, xticks=[-100, -50, 0, 50, 100],
                                       ymin=0, ymax=0.05, vertical_bar_width=2, vlines=[-2, 2], helper_lines_lc='k',
                                       helper_lines_dashes=(1, 0))
    ibi_distr_plot = fig.create_plot(xpos=1, ypos=11, plot_height=1.25, plot_width=2.5,
                                    xl='Interbout interval (s)', xmin=0, xmax=3, xticks=[0, 1, 2, 3],
                                    ymin=0, ymax=2.1, vertical_bar_width=0.05, helper_lines_lc='k',
                                     helper_lines_dashes=(1, 0))

    add_model_hypothesis_plot = fig.create_plot(xpos=1.88, ypos=8.5, plot_height=1.5, plot_width=2.75,
                                                axis_off=True, xmin=-110, xmax=110, ymin=0, ymax=1,
                                                hlines=[0.5], vlines=[0], yl='Right swims (%)', xl='M coherence',
                                                helper_lines_lc='k', helper_lines_dashes=(1, 0))
    or_model_hypothesis_plot = fig.create_plot(xpos=6, ypos=8.5, plot_height=1.5, plot_width=2.75,
                                                axis_off=True, xmin=-110, xmax=110, ymin=0, ymax=1,
                                                hlines=[0.5], vlines=[0], helper_lines_lc='k', helper_lines_dashes=(1, 0),
                                                legend_xpos=7, legend_ypos=10.75)

    acc_over_coherence_plot = fig.create_plot(xpos=1.1, ypos=5.25, plot_height=3, plot_width=4.5, errorbar_area=True,
                                              xl='Coherence (%)', xmin=x_input[0] - 10, xmax=x_input[-1] + 10,
                                              xticks=[-100, -50, 0, 50, 100], yl='Rightward swims (%)', ymin=-0.05,
                                              ymax=1.05, yticks=[0.25, 0.50, 0.75], yticklabels=['25', '50', '75'],
                                              hlines=[0.5], vlines=[0], helper_lines_lc='k', helper_lines_dashes=(1, 0))

    single_fish_plot1 = fig.create_plot(xpos=6, ypos=6.5, plot_height=1.75, plot_width=2.75,
                                        errorbar_area=True, xmin=x_input[0] - 10, xmax=x_input[-1] + 10,
                                        xticks=[-100, -50, 0, 50, 100], xticklabels=['', '', '', '', ''],
                                        ymin=-0.05, ymax=1.05, yticks=[0.25, 0.50, 0.75], yticklabels=['', '', ''],
                                        hlines=[0.5], vlines=[0], helper_lines_lc='k', helper_lines_dashes=(1, 0))
    single_fish_plot2 = fig.create_plot(xpos=6, ypos=4.5, plot_height=1.75, plot_width=2.75,
                                        errorbar_area=True, xmin=x_input[0] - 10, xmax=x_input[-1] + 10,
                                        xticks=[-100, -50, 0, 50, 100], xticklabels=['', '', '', '', ''],
                                        ymin=-0.05, ymax=1.05, yticks=[0.25, 0.50, 0.75], yticklabels=['', '', ''],
                                        hlines=[0.5], vlines=[0], helper_lines_lc='k', helper_lines_dashes=(1, 0))
    single_fish_plot3 = fig.create_plot(xpos=6, ypos=2.5, plot_height=1.75, plot_width=2.75,
                                        errorbar_area=True, xmin=x_input[0] - 10, xmax=x_input[-1] + 10,
                                        xticks=[-100, -50, 0, 50, 100], xticklabels=['', '', '', '', ''],
                                        ymin=-0.05, ymax=1.05, yticks=[0.25, 0.50, 0.75], yticklabels=['', '', ''],
                                        hlines=[0.5], vlines=[0], helper_lines_lc='k', helper_lines_dashes=(1, 0))
    single_fish_plot4 = fig.create_plot(xpos=6, ypos=0.5, plot_height=1.75, plot_width=2.75,
                                        errorbar_area=True, xmin=x_input[0] - 10, xmax=x_input[-1] + 10,
                                        xticks=[-100, -50, 0, 50, 100], xticklabels=['', '', '', '', ''],
                                        ymin=-0.05, ymax=1.05,yticks=[0.25, 0.50, 0.75], yticklabels=['', '', ''],
                                        hlines=[0.5], vlines=[0], helper_lines_lc='k', helper_lines_dashes=(1, 0))
    single_fish_plots = [single_fish_plot1, single_fish_plot2, single_fish_plot3, single_fish_plot4]

    subfig_mse = fig.create_plot(xpos=1.1, ypos=2.75, plot_height=1.5, plot_width=4.5,
                                 xmin=-0.25, xmax=3.25, ymin=-0.025, ymax=0.175,
                                 yticks=[0, 0.05, 0.1, 0.15], yl='MSE', xticklabels=['ADD', 'WTA', 'MOT', 'LUMI'],
                                 xticks=[0, 1, 2, 3])

    ibi_overview_plot = fig.create_plot(xpos=3.75, ypos=0.5, plot_height=1.5, plot_width=1.75,
                                        xmin=-0.5, xmax=3.5, xticks=[0, 1, 2, 3],
                                        xticklabels=['M', 'L', 'M=L', 'M\u2260L'], yl='Relative IBI (s)',
                                        ymin=-0.4, ymax=0.56, yticks=[-0.4, -0.2, 0, 0.2, 0.4])
    orien_overview_plot = fig.create_plot(xpos=1.1, ypos=0.5, plot_height=1.5, plot_width=1.75,
                                        xmin=-0.5, xmax=3.5, xticks=[0, 1, 2, 3],
                                        xticklabels=['M', 'L', 'M=L', 'M\u2260L'], yl='Following swims (%)',
                                        ymin=0.25, ymax=1.11, yticks=[0.4, 0.6, 0.8, 1.0], yticklabels=['40', '60', '80', '100'])

    sub_fig_raw_trace(path_to_raw_data, raw_orien_plot)
    sub_fig_bout_distributions(path_to_combined, orien_distr_plot, ibi_distr_plot)
    sub_fig_model_hypotheses(add_model_hypothesis_plot, or_model_hypothesis_plot)
    sub_fig_data_plus_model_single_fish(path_to_analysed, plot_fish_IDs, single_fish_plots, subfig_mse)
    sub_plot_data_plus_model_all_fish(path_to_analysed, acc_over_coherence_plot)
    sub_plot_ibi_orientation_overview(path_to_analysed, ibi_overview_plot, orien_overview_plot, plot_fish_IDs)

    fig.save('C:/users/katja/Desktop/fig1.pdf')

