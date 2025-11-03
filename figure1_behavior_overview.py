import h5py
import numpy as np
from pathlib import Path
import pandas as pd
from multifeature_integration_paper.figure_helper import Figure
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel
from multifeature_integration_paper.useful_small_funcs import cohens_d

def additive_model_coen(input, vr, vl, ar, al, b, gamma):
    '''
    This function contains the additive model.
    :param input: Coherence levels inputs (between -100 and 100)
    :param vr: rightward-motion-scaling-factor
    :param vl: leftward-motion-scaling-factor
    :param ar: rightward-luminance-scaling-factor
    :param al: leftward-luminance-scaling-factor
    :param b: bias
    :param gamma: motion-gamma
    :return: pL: percentage-rightward output. Same length as input.
    '''
    VR, VL, AR, AL = input
    x = vr * VR ** gamma - vl * VL ** gamma + ar * AR - al * AL + b
    pL = 1 / (1 + np.exp(-x))
    return pL

def strongest_stim_model(input, vr, vl, ar, al, b, gamma):
    '''
    This function contains the strongest-stimulus or WTA model.
    :param input: Coherence levels inputs (between -100 and 100)
    :param vr: rightward-motion-scaling-factor
    :param vl: leftward-motion-scaling-factor
    :param ar: rightward-luminance-scaling-factor
    :param al: leftward-luminance-scaling-factor
    :param b: bias
    :param gamma: motion-gamma
    :return: pL: percentage-rightward output. Same length as input.
    '''
    VR, VL, AR, AL = input
    x = b * np.ones(input.shape[1])
    # Find the strongest stimulus for each input
    strongest_stim = np.argmax(np.array(np.array([-vr*VR**gamma, -vl*VL**gamma, ar*AR, al*AL])), axis=0)

    # only consider the stongest stimulus.
    x[strongest_stim == 0] += vr * VR[strongest_stim == 0] ** gamma
    x[strongest_stim == 1] -= vl * VL[strongest_stim == 1] ** gamma
    x[strongest_stim == 2] += ar * AR[strongest_stim == 2]
    x[strongest_stim == 3] -= al * AL[strongest_stim == 3]

    pL = 1 / (1 + np.exp(-x))
    return pL

def motion_only_model(input, vr, vl, ar, al, b, gamma):
    '''
    This function contains the only-motion model based on the additive model.
    :param input: Coherence levels inputs (between -100 and 100)
    :param vr: rightward-motion-scaling-factor
    :param vl: leftward-motion-scaling-factor
    :param ar: rightward-luminance-scaling-factor (is ignored in this function).
    :param al: leftward-luminance-scaling-factor (is ignored in this function).
    :param b: bias
    :param gamma: motion-gamma (is ignored in this function).
    :return: pL: percentage-rightward output. Same length as input.
    '''
    VR, VL, AR, AL = input
    x = vr * VR ** gamma - vl * VL ** gamma + b
    pL = 1 / (1 + np.exp(-x))
    return pL

def lumi_only_model(input, vr, vl, ar, al, b, gamma):
    '''
    This function contains the only-luminance model based on the additive model.
    :param input: Coherence levels inputs (between -100 and 100)
    :param vr: rightward-motion-scaling-factor (is ignored in this function).
    :param vl: leftward-motion-scaling-factor (is ignored in this function).
    :param ar: rightward-luminance-scaling-factor.
    :param al: leftward-luminance-scaling-factor.
    :param b: bias
    :param gamma: motion-gamma (is ignored in this function).
    :return: pL: percentage-rightward output. Same length as input.
    '''
    VR, VL, AR, AL = input
    x = ar * AR - al * AL + b
    pL = 1 / (1 + np.exp(-x))
    return pL

def sub_fig_raw_trace(path_to_raw_data, subfig):
    '''
    This function plots the orientation over time plot.
    Related to Fig. 1b
    :param path_to_raw_data: Path to the raw tracking data of one example fish.
    :param subfig: subfigure to plot the orientation over time plot.
    '''
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
    """
    This function plots the distribution of orientation change and interbout interval across all fish used in Fig. 1.
    Related to Fig. 1c
    :param path_to_combined_data: Path to the dataframe with on each row a bout.
    :param subfiga: Subfigure to plot the orientation change distribution.
    :param subfigb: Subfigure to plot the interbout interval distribution.
    :return:
    """
    combined_df = pd.read_hdf(path_to_combined_data, key='all_bout_data_pandas')

    # plot the orientation change across all fish and all stimuli.
    hist_values = np.histogram(combined_df['estimated_orientation_change'], bins=np.arange(-100, 102, 2), density=True)
    subfiga.draw_vertical_bars(hist_values[1][:-1]+1, hist_values[0], lc='tab:blue')
    subfiga.draw_text(-75, 0.035, 'left bouts')
    subfiga.draw_text(75, 0.035, 'right bouts')

    # plot the interbout interval across all fish and all stimuli.
    hist_values = np.histogram(combined_df['interbout_interval'], bins=np.arange(0, 4, 0.05), density=True)
    subfigb.draw_vertical_bars(hist_values[1][:-1]+0.025, hist_values[0], lc='tab:blue')
    return

def get_model_inputs():
    '''
    This function gives the model inputs at rough (-100, -50, -25, -10, 0, 10, 25, 50, 100)% coherence and interpolated fine scale (stepsize 0.02).
    :return: x_input/x_input_fine coherence levels as used on the x-axis in fig. 1g-h.
             model_in_left/model_in_left_fine leftward motion coherence level as proportion (between 0 and 1) instead of between -100 and 100 percentage.
             model_in_right/model_in_right_fine rightward motion coherence level as proportion (between 0 and 1) instead of between -100 and 100 percentage.
    '''
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
    '''
    Reorder the dataframe from alphabetic stimulus order to numerical stimulus order. Note that this resorting is hard-coded for the specific stimulus dataset used in Slangewal et. al.)
    :param path_to_analysed_data: path to the analysed dataframe
    :return: analysed_df and a list of reordered stimulus names.
    '''
    analysed_df = pd.read_hdf(path_to_analysed_data)

    # Sort full data
    sorted_stim_names = analysed_df.index.unique('stimulus_name').sort_values()

    # Stimuli are alphabetic by default, this order sets them back to -100 to 100 order.
    reorder_order = [0, 1, 2, 9, 10, 11, 6, 7, 8, 3, 4, 5, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 15, 16, 17]
    reorder_stim_names = sorted_stim_names[reorder_order]

    return analysed_df, reorder_stim_names

def sub_fig_model_hypotheses(subfiga, subfigb):
    '''
    This function plots the model hypothesis of the addtive and winner-take-all models.
    This is related to Fig. 1 e-f
    :param subfiga: Subfigure for the addtive model
    :param subfigb: Subfigure for the winner-take-all-model
    '''

    x_input, x_input_fine, model_in_left, model_in_right, model_in_left_fine, model_in_right_fine = get_model_inputs()

    # We use our initial parameter guess to plot the model hypotheses. Parameters are motion-right-scaling-factor, motion-left-scaling-factor, luminance-right-scaling-factor, luminance-left-scaling-factor, bias, motion-gamma.
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
    '''
    This function plots the percentage rightward swims over coherence for the four example fish and calculates the fit of each model (ADD, WTA, MOT, LUMI) for all individual fish.
    Related to Fig. 1h-i
    :param path_to_analysed_data: path to the analysed dataframe where each row contains the data (e.g. percentage-leftward-swims) of a 10s time-window.
    :param plot_fish_IDs: Array of Fish IDs of example fish. The length of the array should match the number of subfigs, if not, the smallest of the two determines how many examples are shown.
    :param subfigs: Array of subfigures that will show the percentage of rightward swims over coherence for the example fish.
    :param subfig_mse: Subfigure to show the mean-squared-error of the fit between indivual fish and each model (ADD, WTA, MOT, LUMI).
    '''

    # We get the input for the models at rough (-100, -50, -25, -10, 0, 10, 25, 50, 100)% coherence and interpolated fine scale.
    x_input, x_input_fine, model_in_left, model_in_right, model_in_left_fine, model_in_right_fine = get_model_inputs()

    # We reorder the data from alphabetical stimulus order to numerical stimulus order.
    analysed_df, reorder_stim_names = reorder_analysed_df(path_to_analysed_data)

    # Initial guess of the model parameters.
    params_init = [-1, -1, 0.5, 0.5, 0, 0.6]

    # Loop over the example fish to be plotted in Fig. 1h.
    for single_fish_plot_id, single_fish_ID in enumerate(plot_fish_IDs):
        data_single = []
        # For each stimulus extract the motion coherence as well as the luminance level and plot a scatter point accordingly.
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

        # Concatenate the three curves (lumi left, lumi off, lumi right) to use for model training and plot the fitted model curves.
        input_train = np.array([np.concatenate([model_in_left, model_in_left, model_in_left]),
                          np.concatenate([model_in_right, model_in_right, model_in_right]),
                          np.concatenate([np.zeros(len(model_in_left)), np.zeros(len(model_in_left)), np.ones(len(model_in_left))]),
                          np.concatenate([np.ones(len(model_in_left)), np.zeros(len(model_in_left)), np.zeros(len(model_in_left))])])

        popt, pcov = curve_fit(additive_model_coen, input_train, np.concatenate([data_single[::3], data_single[1::3], data_single[2::3]]), p0=params_init)

        input_fine = np.array([model_in_left_fine, model_in_right_fine, np.zeros(len(model_in_left_fine)), np.ones(len(model_in_left_fine))])
        subfigs[single_fish_plot_id].draw_line(x=x_input_fine, y=additive_model_coen(input_fine, *popt), lc='tab:red', ec='tab:red')

        input_fine = np.array([model_in_left_fine, model_in_right_fine, np.zeros(len(model_in_left_fine)), np.zeros(len(model_in_left_fine))])
        subfigs[single_fish_plot_id].draw_line(x=x_input_fine, y=additive_model_coen(input_fine, *popt), lc='tab:gray', ec='tab:gray')

        input_fine = np.array([model_in_left_fine, model_in_right_fine, np.ones(len(model_in_left_fine)), np.zeros(len(model_in_left_fine))])
        subfigs[single_fish_plot_id].draw_line(x=x_input_fine, y=additive_model_coen(input_fine, *popt), lc='tab:blue', ec='tab:blue')
        subfigs[single_fish_plot_id].draw_text(-120, 1, f'example fish {single_fish_plot_id+1}', textlabel_ha='left')

    # Preparing lists to store the MSE values.
    all_mses_add = []
    all_mses_wta = []
    all_mses_mot = []
    all_mses_lumi = []
    plot_fish_counter = []
    # Loop over each individual and fit all models (ADD, WTA, MOT, LUMI).
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

        # Concatenate the curves (lumi left, lumi off, lumi right) for model fitting.
        input_train = np.array([np.concatenate([model_in_left, model_in_left, model_in_left]),
                          np.concatenate([model_in_right, model_in_right, model_in_right]),
                          np.concatenate([np.zeros(len(model_in_left)), np.zeros(len(model_in_left)), np.ones(len(model_in_left))]),
                          np.concatenate([np.ones(len(model_in_left)), np.zeros(len(model_in_left)), np.zeros(len(model_in_left))])])

        # Fit the four models
        popt, pcov = curve_fit(additive_model_coen, input_train, np.concatenate([data_single[::3], data_single[1::3], data_single[2::3]]), p0=params_init)
        popt_2, pcov_2 = curve_fit(strongest_stim_model, input_train, np.concatenate([data_single[::3], data_single[1::3], data_single[2::3]]), p0=params_init)
        popt_3, pcov_3 = curve_fit(motion_only_model, input_train, np.concatenate([data_single[::3], data_single[1::3], data_single[2::3]]), p0=params_init)
        popt_4, pcov_4 = curve_fit(lumi_only_model, input_train, np.concatenate([data_single[::3], data_single[1::3], data_single[2::3]]), p0=params_init)

        # Calculate the mean-squared-error for lumi-off
        input = np.array([model_in_left, model_in_right, np.zeros(len(model_in_left)), np.zeros(len(model_in_left))])
        mse_o = np.nanmean(np.square(additive_model_coen(input, *popt)*100 - data_single[1::3]*100))
        mse_o_2 = np.nanmean(np.square(strongest_stim_model(input, *popt_2)*100 - data_single[1::3]*100))
        mse_o_3 = np.nanmean(np.square(motion_only_model(input, *popt_3)*100 - data_single[1::3]*100))
        mse_o_4 = np.nanmean(np.square(lumi_only_model(input, *popt_4)*100 - data_single[1::3]*100))

        # Calculate the mean-squared-error for lumi-left
        input = np.array([model_in_left, model_in_right, np.zeros(len(model_in_left)), np.ones(len(model_in_left))])
        mse_r = np.nanmean(np.square(additive_model_coen(input, *popt)*100 - data_single[::3]*100))
        mse_r_2 = np.nanmean(np.square(strongest_stim_model(input, *popt_2)*100 - data_single[::3]*100))
        mse_r_3 = np.nanmean(np.square(motion_only_model(input, *popt_3)*100 - data_single[::3]*100))
        mse_r_4 = np.nanmean(np.square(lumi_only_model(input, *popt_4)*100 - data_single[::3]*100))

        # Calculate the mean-squared-error for lumi-right
        input = np.array([model_in_left, model_in_right, np.ones(len(model_in_left)), np.zeros(len(model_in_left))])
        mse_l = np.nanmean(np.square(additive_model_coen(input, *popt)*100 - data_single[2::3]*100))
        mse_l_2 = np.nanmean(np.square(strongest_stim_model(input, *popt_2)*100 - data_single[2::3]*100))
        mse_l_3 = np.nanmean(np.square(motion_only_model(input, *popt_3)*100 - data_single[2::3]*100))
        mse_l_4 = np.nanmean(np.square(lumi_only_model(input, *popt_4)*100 - data_single[2::3]*100))

        # For our own sanity, always report the fish for whom the Additive model is not the best model.
        if np.nanmean([mse_r, mse_o, mse_l]) > np.nanmean([mse_r_2, mse_o_2, mse_l_2]) or np.nanmean([mse_r, mse_o, mse_l]) > np.nanmean([mse_r_3, mse_o_3, mse_l_3]) or np.nanmean([mse_r, mse_o, mse_l]) > np.nanmean([mse_r_4, mse_o_4, mse_l_4]):
            print('The additive model is not the best fit for fish ', single_fish_ID)
            print('MSE add, MSE wta, MSE mot, MSE lumi:')
            print(np.nanmean([mse_r, mse_o, mse_l]), np.nanmean([mse_r_2, mse_o_2, mse_l_2]), np.nanmean([mse_r_3, mse_o_3, mse_l_3]), np.nanmean([mse_r_4, mse_o_4, mse_l_4]))

        # Plot the MSE for the individual fish as a gray line.
        all_mses_add = np.append(all_mses_add, np.nanmean([mse_r, mse_o, mse_l]))
        all_mses_wta = np.append(all_mses_wta, np.nanmean([mse_r_2, mse_o_2, mse_l_2]))
        all_mses_mot = np.append(all_mses_mot, np.nanmean([mse_r_3, mse_o_3, mse_l_3]))
        all_mses_lumi = np.append(all_mses_lumi, np.nanmean([mse_r_4, mse_o_4, mse_l_4]))

        subfig_mse.draw_line([0, 1, 2, 3], [np.nanmean([mse_r, mse_o, mse_l]), np.nanmean([mse_r_2, mse_o_2, mse_l_2]), np.nanmean([mse_r_3, mse_o_3, mse_l_3]), np.nanmean([mse_r_4, mse_o_4, mse_l_4])], lc='darkgray', lw=0.25)

    # Add the model fits of the example fish as dashed gray lines with increasing dash-size (ld)
    for single_fish_id, ld in zip(plot_fish_counter.astype(int), [(1, 3), (2, 3), (3, 3), (4, 3)]):
        subfig_mse.draw_line([0, 1, 2, 3], [all_mses_add[single_fish_id], all_mses_wta[single_fish_id], all_mses_mot[single_fish_id], all_mses_lumi[single_fish_id]], lc='dimgray',
                              line_dashes=ld, lw=0.75)
    subfig_mse.draw_line([0, 1, 2, 3], [np.nanmean(all_mses_add), np.nanmean(all_mses_wta), np.nanmean(all_mses_mot), np.nanmean(all_mses_lumi)], lc='k', lw=1)

    # Statistically compare the Additive model to all others using a t-test and comparing the p-value to the Bonferonni corrected threshold.
    subfig_mse.draw_line([0, 0, 1, 1], [1100, 1120, 1120, 1100], lc='k')
    _, pval = ttest_rel(all_mses_add, all_mses_wta)
    if pval < 0.001 / 3:
        subfig_mse.draw_text(0.5, 1150, '***')
    elif pval < 0.01 / 3:
        subfig_mse.draw_text(0.5, 1150, '**')
    elif pval < 0.05 / 3:
        subfig_mse.draw_text(0.5, 1150, '*')
    else:
        subfig_mse.draw_text(0.5, 1150, 'ns')
    effect_size = cohens_d(all_mses_add, all_mses_wta)
    print(f'ADD vs WTA, pval {pval}: Cohen D effect size {effect_size}')

    subfig_mse.draw_line([0, 0, 2, 2], [1300, 1320, 1320, 1300], lc='k')
    _, pval = ttest_rel(all_mses_add, all_mses_mot)
    if pval < 0.001 / 3:
        subfig_mse.draw_text(1, 1350, '***')
    elif pval < 0.01 / 3:
        subfig_mse.draw_text(1, 1350, '**')
    elif pval < 0.05 / 3:
        subfig_mse.draw_text(1, 1350, '*')
    else:
        subfig_mse.draw_text(1, 1350, 'ns')
    effect_size = cohens_d(all_mses_add, all_mses_mot)
    print(f'ADD vs MOT, pval {pval}: Cohen D effect size {effect_size}')

    subfig_mse.draw_line([0, 0, 3, 3], [1500, 1520, 1520, 1500], lc='k')
    _, pval = ttest_rel(all_mses_add, all_mses_lumi)
    if pval < 0.001 / 3:
        subfig_mse.draw_text(1.5, 1550, '***')
    elif pval < 0.01 / 3:
        subfig_mse.draw_text(1.5, 1550, '**')
    elif pval < 0.05 / 3:
        subfig_mse.draw_text(1.5, 1550, '*')
    else:
        subfig_mse.draw_text(1.5, 1550, 'ns')
    effect_size = cohens_d(all_mses_add, all_mses_lumi)
    print(f'ADD vs LUMI, pval {pval}: Cohen D effect size {effect_size}')

    return

def sub_plot_data_plus_model_all_fish(path_to_analysed_data, subfig):
    '''
    This function plots the mean percentage rightward swims over coherence across all fish as well as the fit of the additive model.
    This is related to Fig. 1g
    :param path_to_analysed_data: path to the analysed dataframe where each row contains the data (e.g. percentage-leftward-swims) of a 10s time-window.
    :param subfig: subfigure to plot the percentage rightward swims over coherence.
    '''

    # We get the input to the additive model at rough-scale (-100, -50, -25, -10, 0, 10, 25, 50, 100)% coherence, and interpolated fine-scale.
    x_input, x_input_fine, model_in_left, model_in_right, model_in_left_fine, model_in_right_fine = get_model_inputs()

    # We need to reorder the stimuli from alphabetic to numerical order.
    analysed_df, reorder_stim_names = reorder_analysed_df(path_to_analysed_data)

    # Initial model parameter guesses
    params_init = [-1, -1, 0.5, 0.5, 0, 0.6]

    # For each stimuli read-out the motion coherence and luminance value. And accordingly add a scatter-point and sem-errorbar to the subfigure.
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

    # Concatenate the model input for the three curves (lumi left, lumi off, lumi right).
    input_train = np.array([np.concatenate([model_in_left, model_in_left, model_in_left]),
                            np.concatenate([model_in_right, model_in_right, model_in_right]),
                            np.concatenate([np.zeros(len(model_in_left)), np.zeros(len(model_in_left)),
                                            np.ones(len(model_in_left))]),
                            np.concatenate([np.ones(len(model_in_left)), np.zeros(len(model_in_left)),
                                            np.zeros(len(model_in_left))])])

    # Fit the models. Note for the figure we only need the additive_model. For our own sanity we check and print the performance of alternative models.
    popt, pcov = curve_fit(additive_model_coen, input_train, np.concatenate([data[::3], data[1::3], data[2::3]]), p0=params_init)
    popt_2, pcov_2 = curve_fit(strongest_stim_model, input_train, np.concatenate([data[::3], data[1::3], data[2::3]]), p0=params_init)
    popt_3, pcov_3 = curve_fit(motion_only_model, input_train, np.concatenate([data[::3], data[1::3], data[2::3]]), p0=params_init)
    popt_4, pcov_4 = curve_fit(lumi_only_model, input_train, np.concatenate([data[::3], data[1::3], data[2::3]]), p0=params_init)

    # Draw the lumi left curve
    input = np.array([model_in_left, model_in_right, np.zeros(len(model_in_left)), np.ones(len(model_in_left))])
    input_fine = np.array([model_in_left_fine, model_in_right_fine, np.zeros(len(model_in_left_fine)), np.ones(len(model_in_left_fine))])
    mse = np.nanmean(np.square(additive_model_coen(input, *popt)*100 - data[::3]*100))
    mse_2 = np.nanmean(np.square(strongest_stim_model(input, *popt_2)*100 - data[::3]*100))
    print(f'BEST MSE: {mse}, {mse_2}')
    print(f'Popt avg -Lumi {popt}')
    subfig.draw_line(x=x_input_fine, y=additive_model_coen(input_fine, *popt), lc='tab:red', ec='tab:red',)

    # Draw the lumi off curve
    input = np.array([model_in_left, model_in_right, np.zeros(len(model_in_left)), np.zeros(len(model_in_left))])
    input_fine = np.array([model_in_left_fine, model_in_right_fine, np.zeros(len(model_in_left_fine)), np.zeros(len(model_in_left_fine))])
    mse = np.nanmean(np.square(additive_model_coen(input, *popt)*100 - data[1::3]*100))
    mse_2 = np.nanmean(np.square(strongest_stim_model(input, *popt_2)*100 - data[1::3]*100))
    print(f'BEST MSE: {mse}, {mse_2}')
    subfig.draw_line(x=x_input_fine, y=additive_model_coen(input_fine, *popt), lc='tab:gray', ec='tab:gray',)
    print(f'Popt avg no Lumi {popt}')

    # Draw the lumi right curve
    input = np.array([model_in_left, model_in_right, np.ones(len(model_in_left)), np.zeros(len(model_in_left))])
    input_fine = np.array([model_in_left_fine, model_in_right_fine, np.ones(len(model_in_left_fine)), np.zeros(len(model_in_left_fine))])
    mse = np.nanmean(np.square(additive_model_coen(input, *popt)*100 - data[2::3]*100))
    mse_2 = np.nanmean(np.square(strongest_stim_model(input, *popt_2)*100 - data[2::3]*100))
    print(f'BEST MSE: {mse}, {mse_2}')
    subfig.draw_line(x=x_input_fine, y=additive_model_coen(input_fine, *popt), lc='tab:blue', ec='tab:blue',)
    print(f'Popt avg +Lumi {popt}')

    # Add the label with the number of fish.
    n_fish = len(analysed_df.index.unique('experiment_ID'))
    subfig.draw_text(-60, 0.7, f'{n_fish} fish')
    return

def sub_plot_ibi_orientation_overview(path_to_analysed_data, subfiga, subfigb, plot_fish_IDs):
    """
    This function plots the percentage following swims and relative interbout-interval per stimulus type. Stimulus types are only-motion (M), only-luminance (L), congruent (M=L) and conflicting (M!=L).
    This is related to figure 1j-k.
    :param path_to_analysed_data: path to the analysed dataframe where each row contains the data (e.g. average interbout_interval and percentage-leftward-swims) of a 10s time-window.
    :param subfiga: subfigure to plot the interbout-interval data
    :param subfigb: subfigure to plot the percentage following swims
    :param plot_fish_IDs: list of fish IDs that need to be highlighted by dashed lines (these should be the same fish as in Fig. 1h).
    """
    analysed_df, reorder_stim_names = reorder_analysed_df(path_to_analysed_data)
    n_fish = len(analysed_df.index.unique('experiment_ID'))

    # Note, the original stimulus name consist of 'coherence'_'luminance', where coherence is a value between -100 and 100 and luminance is either -1, 0, or 1. We hardcoded the use of 25% coherence for motion stimulus,
    # as we found that the behavior of the fish at this coherence level matches the behavior for luminance stimuli. We merge the interbout_intervals for left and rightward stimuli of the same value.
    ibi_data = np.zeros((n_fish, 4))
    ibi_data[:, 0] = np.nanmean([analysed_df.xs(f'25_0', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval'],
                                  analysed_df.xs(f'-25_0', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval']], axis=0)
    ibi_data[:, 1] = np.nanmean([analysed_df.xs(f'0_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval'],
                                analysed_df.xs(f'0_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval']], axis=0)
    ibi_data[:, 2] = np.nanmean([analysed_df.xs(f'25_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval'],
                                analysed_df.xs(f'-25_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval']], axis=0)
    ibi_data[:, 3] = np.nanmean([analysed_df.xs(f'25_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval'],
                                    analysed_df.xs(f'-25_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['interbout_interval']], axis=0)

    # We flip the percentage_left for rightward stimuli to be able to merge them with the leftward stimuli.
    pl_data = np.zeros((n_fish, 4))
    pl_data[:, 0] = np.nanmean([analysed_df.xs(f'25_0', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left'],
                                  1 - analysed_df.xs(f'-25_0', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left']], axis=0)
    pl_data[:, 1] = np.nanmean([analysed_df.xs(f'0_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left'],
                                1 - analysed_df.xs(f'0_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left']], axis=0)
    pl_data[:, 2] = np.nanmean([analysed_df.xs(f'25_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left'],
                                1 - analysed_df.xs(f'-25_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left']], axis=0)
    pl_data[:, 3] = np.nanmean([analysed_df.xs(f'-25_1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left'],
                                1 - analysed_df.xs(f'25_-1', level='stimulus_name').query('window_time > 10 and window_time <= 20')['percentage_left']], axis=0)

    # Here we select the data of the 4 example fish shown in Fig. 1h.
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

    # We plot each individual fish.
    for fish in range(n_fish):
        subfiga.draw_line([0, 1, 2, 3], ibi_data[fish, :] - np.nanmean(ibi_data[fish, :]), lc='darkgray', lw=0.25)
        subfigb.draw_line([0, 1, 2, 3], pl_data[fish, :], lc='darkgray', lw=0.25)
    # We plot the 4 example fish with dashed lines of increasing dash sizes (ld).
    for fish, ld in zip(range(4), [(1, 3), (2, 3), (3, 3), (4, 3)]):
        subfiga.draw_line([0, 1, 2, 3], sf_ibi_data[fish, :] - np.nanmean(sf_ibi_data[fish, :]), lc='dimgray', line_dashes=ld, lw=0.75)
        subfigb.draw_line([0, 1, 2, 3], sf_pl_data[fish, :], lc='dimgray', line_dashes=ld, lw=0.75)
    # We plot the median responses in black
    subfiga.draw_line([0, 1, 2, 3], np.nanmedian(ibi_data - np.nanmean(ibi_data, axis=1).reshape(-1, 1), axis=0), lc='k', lw=1)
    subfigb.draw_line([0, 1, 2, 3], np.nanmedian(pl_data, axis=0), lc='k', lw=1)

    # Finally we calculate the statistics by performing t-tests and comparing to a Bonferonni corrected alpha-value.
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
    # Provide the path to save the figure.
    fig_save_path = 'C:/users/katja/Desktop/fig_1.pdf'
    # Provide the path to the figure_1 folder.
    fig_1_folder_path = r'Z:\Bahl lab member directories\Katja\paper_data\figure_1'

    # Get the paths to the combined dataframes for figure 1. path_to_combined should contain the dataframe with one row per bout. path_to_analysed should contain the dataframe with one row per window (each contains data of a 10s window).
    path_to_analysed = Path(fr'{fig_1_folder_path}\data_analysed.hdf5')
    path_to_combined = Path(rf'{fig_1_folder_path}\data_combined.h5')

    # Get the path to one example fish, used to plot the orientation over time in Fig. 1b.
    path_to_raw_data = Path(rf'{fig_1_folder_path}\2024-10-22_08-47-26_setup2_arena0\2024-10-22_08-47-26_setup2_arena0.hdf5')

    # Set the fish IDs for 4 example fish in Fig. 1h (as well as the dashed lines in Fig. 1i-k).
    plot_fish_IDs = [107, 110, 88, 95]

    # We only need the size of the model inputs to know the x-limits of plot Fig1g-h.
    x_input, _, _, _, _, _ = get_model_inputs()

    # Here we define the figure and subpanel outlines (e.g. the limits, ticks and labels of the axes).
    fig = Figure(fig_width=9, fig_height=17)
    # Fig. 1b
    raw_orien_plot = fig.create_plot(xpos=4.75, ypos=15, plot_height=1, plot_width=3.75,
                                    xl='', yl='orientation (deg)',
                                    yticks=[774, 824], yticklabels=['0', '50'], xmin=15.7, xmax=18.5, ymin=770, ymax=830,
                                     vlines=[16.17, 16.32, 16.845, 16.98, 17.7, 17.84], helper_lines_lc='tab:gray',
                                     helper_lines_dashes=(4, 4))

    # Fig. 1c
    orien_distr_plot = fig.create_plot(xpos=1, ypos=13, plot_height=1.25, plot_width=2.5,
                                       xl='orientation change (deg/bout)', xmin=-110, xmax=110, xticks=[-100, -50, 0, 50, 100],
                                       ymin=0, ymax=0.05, vertical_bar_width=2, vlines=[-2, 2], helper_lines_lc='k',
                                       helper_lines_dashes=(1, 0))
    ibi_distr_plot = fig.create_plot(xpos=1, ypos=11, plot_height=1.25, plot_width=2.5,
                                    xl='interbout interval (s)', xmin=0, xmax=3, xticks=[0, 1, 2, 3],
                                    ymin=0, ymax=2.1, vertical_bar_width=0.05, helper_lines_lc='k',
                                     helper_lines_dashes=(1, 0))

    # Fig. 1e
    add_model_hypothesis_plot = fig.create_plot(xpos=1.88, ypos=8.5, plot_height=1.5, plot_width=2.75,
                                                axis_off=True, xmin=-110, xmax=110, ymin=0, ymax=1,
                                                hlines=[0.5], vlines=[0], yl='Right swims (%)', xl='M coherence',
                                                helper_lines_lc='k', helper_lines_dashes=(1, 0))
    # Fig. 1f
    or_model_hypothesis_plot = fig.create_plot(xpos=6, ypos=8.5, plot_height=1.5, plot_width=2.75,
                                                axis_off=True, xmin=-110, xmax=110, ymin=0, ymax=1,
                                                hlines=[0.5], vlines=[0], helper_lines_lc='k', helper_lines_dashes=(1, 0),
                                                legend_xpos=7, legend_ypos=10.75)

    # Fig. 1g
    acc_over_coherence_plot = fig.create_plot(xpos=1.1, ypos=5.25, plot_height=3, plot_width=4.5, errorbar_area=True,
                                              xl='Coherence (%)', xmin=x_input[0] - 10, xmax=x_input[-1] + 10,
                                              xticks=[-100, -50, 0, 50, 100], yl='rightward swims (%)', ymin=-0.05,
                                              ymax=1.05, yticks=[0.25, 0.50, 0.75], yticklabels=['25', '50', '75'],
                                              hlines=[0.5], vlines=[0], helper_lines_lc='k', helper_lines_dashes=(1, 0))

    # Fig. 1h
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

    # Fig. 1i
    subfig_mse = fig.create_plot(xpos=1.1, ypos=2.75, plot_height=1.5, plot_width=4.5,
                                 xmin=-0.25, xmax=3.25, ymin=-2.5, ymax=1750,
                                 yticks=[0, 500, 1000, 1500], yl='MSE (%\u00b2)', xticklabels=['ADD', 'WTA', 'MOT', 'LUMI'],
                                 xticks=[0, 1, 2, 3])

    # Fig. 1j
    orien_overview_plot = fig.create_plot(xpos=1.1, ypos=0.5, plot_height=1.5, plot_width=1.75,
                                        xmin=-0.5, xmax=3.5, xticks=[0, 1, 2, 3],
                                        xticklabels=['M', 'L', 'M=L', 'M\u2260L'], yl='following swims (%)',
                                        ymin=0.25, ymax=1.11, yticks=[0.4, 0.6, 0.8, 1.0], yticklabels=['40', '60', '80', '100'])
    # Fig. 1k
    ibi_overview_plot = fig.create_plot(xpos=3.75, ypos=0.5, plot_height=1.5, plot_width=1.75,
                                        xmin=-0.5, xmax=3.5, xticks=[0, 1, 2, 3],
                                        xticklabels=['M', 'L', 'M=L', 'M\u2260L'], yl='relative IBI (s)',
                                        ymin=-0.4, ymax=0.56, yticks=[-0.4, -0.2, 0, 0.2, 0.4])

    # Finally we plot all the data
    # Fig. 1b
    sub_fig_raw_trace(path_to_raw_data, raw_orien_plot)

    # Fig. 1c
    sub_fig_bout_distributions(path_to_combined, orien_distr_plot, ibi_distr_plot)

    # Fig. 1e-f
    sub_fig_model_hypotheses(add_model_hypothesis_plot, or_model_hypothesis_plot)

    # Fig. 1h-i
    sub_fig_data_plus_model_single_fish(path_to_analysed, plot_fish_IDs, single_fish_plots, subfig_mse)

    # Fig. 1g
    sub_plot_data_plus_model_all_fish(path_to_analysed, acc_over_coherence_plot)

    # Fig. 1j-k
    sub_plot_ibi_orientation_overview(path_to_analysed, ibi_overview_plot, orien_overview_plot, plot_fish_IDs)

    fig.save(fig_save_path)

    # Note that Fig. 1a and d only contain explanatory cartoons without actual data and therefore are not part of this code.

