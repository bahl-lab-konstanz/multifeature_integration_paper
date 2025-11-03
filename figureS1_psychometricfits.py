from pathlib import Path
import numpy as np
from multifeature_integration_paper.figure_helper import Figure
from scipy.optimize import curve_fit
import pandas as pd

def threshold_width_model(Vin, threshold, width):
    '''
    This function contains the threshold and width based psychometric curve.
    :param Vin: Coherence levels inputs (between -100 and 100)
    :param threshold: threshold - indicates how much the inflection point is shifted across the x-axis
    :param width: width - indicates the coding region that contains the 10–90% response rate.
    :return: pL: percentage-rightward output. Same length as input.
    '''
    y = (-2 * np.log10((1 / 0.8) - 1) * (Vin - threshold)) / width
    pL = 1 / (1 + np.exp(-y))
    return pL


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


def sup_fig_single_fish_analysis(path_to_analysed_data, subfigs_single_fish, subfig_swarm, plot_fish_IDs):
    '''
    This function fits the threshold and width based psychometric curve to each single fish and plots the parameters in the swarmplot (Fig. S1b).
    It also plots the accuracy over coherence for the given example fish.
    :param path_to_analysed_data: Path to the analysed dataframe.
    :param subfigs_single_fish: Subfigure to plot the single example fish accuracy over coherence plots.
    :param subfig_swarm: Subfigure to plot the swarmplot of the fitted model parameters (threshold and width for 3 difference luminance values: left, off, right).
    :param plot_fish_IDs: List of example fish.
    '''
    # Load the data and reorder the stimuli from alphabetic to numerical orde.r
    analysed_df, reorder_stim_names = reorder_analysed_df(path_to_analysed_data)

    # Get the interpolated model input.
    x_input_fine = np.arange(-100, 101, 2)
    model_in_abs = [-1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1.0]
    model_in_abs_fine = np.arange(-1, 1.01, 0.02)
    # Define the initial parameter guess.
    params_TW_init = [0, 0.4]

    # Initialize the lists to keep track of the two parameters for each luminanance level for each fish.
    single_fish_thresh_LL = []
    single_fish_thresh_LR = []
    single_fish_thresh_LO = []
    single_fish_width_LL = []
    single_fish_width_LR = []
    single_fish_width_LO = []
    single_fish_plot_count = 0
    # Loop over all fish.
    for single_fish_ID in analysed_df.index.unique('experiment_ID'):
        data_single = []
        # Loop over all stimuli.
        for stim_name in reorder_stim_names:
            coherence = int(stim_name.split('_')[0])
            lumi = int(stim_name.split('_')[1])

            # select the data for the current stimlus
            stim_df = analysed_df.xs(stim_name, level='stimulus_name')

            # check if the current fish has data for this stimulus and if so plot its data.
            if single_fish_ID in stim_df.index.unique('experiment_ID'):
                fish_df = stim_df.xs(single_fish_ID, level='experiment_ID')
                grouped_df_single = fish_df.groupby('window_time')
                # Get the accuracy of the current fish and stimulus
                mean = np.array(grouped_df_single['percentage_left'].mean())[
                    grouped_df_single['percentage_left'].mean().index == 20][0]

                # If the single fish is one of the randomly chosen example fish, plot its accuracy over coherence scatter point.
                if single_fish_ID in plot_fish_IDs:
                    if lumi == -1:
                        subfigs_single_fish[single_fish_plot_count].draw_scatter(x=coherence, y=mean, pc='tab:red',
                                                                                 lw=0.5, ec=None)
                    elif lumi == 1:
                        subfigs_single_fish[single_fish_plot_count].draw_scatter(x=coherence, y=mean,
                                                                                 pc='tab:blue', lw=0.5, ec=None)
                    if lumi == 0:
                        subfigs_single_fish[single_fish_plot_count].draw_scatter(x=coherence, y=mean,
                                                                                 pc='tab:gray', lw=0.5, ec=None)
                # Append the data to the list containing all stimulus values for this fish.
                data_single = np.append(data_single, mean)
            else:
                print(f'No data found for selected fish {single_fish_ID} stimulus {stim_name}')

        # Fit the threshold and width based psychometric curve to the single fish for lumi left. And store the fitted model parameters.
        popt, pcov = curve_fit(threshold_width_model, model_in_abs, data_single[::3], p0=params_TW_init)
        single_fish_thresh_LR = np.append(single_fish_thresh_LR, popt[0])
        single_fish_width_LR = np.append(single_fish_width_LR, popt[1])
        # If it is one of the example fish, plot the model curve.
        if single_fish_ID in plot_fish_IDs:
            print(single_fish_ID)
            print(f'TW model single red: {popt}')
            t1 = popt[0]
            w1 = popt[1]
            subfigs_single_fish[single_fish_plot_count].draw_line(x=x_input_fine,
                                                                  y=threshold_width_model(model_in_abs_fine, *popt),
                                                                  lc='tab:red')

        # Fit the threshold and width based psychometric curve to the single fish for lumi off. And store the fitted model parameters.
        popt, pcov = curve_fit(threshold_width_model, model_in_abs, data_single[1::3], p0=params_TW_init)
        single_fish_thresh_LO = np.append(single_fish_thresh_LO, popt[0])
        single_fish_width_LO = np.append(single_fish_width_LO, popt[1])
        # If it is one of the example fish, plot the model curve.
        if single_fish_ID in plot_fish_IDs:
            print(f'TW model single gray: {popt}')
            t2 = popt[0]
            w2 = popt[1]
            subfigs_single_fish[single_fish_plot_count].draw_line(x=x_input_fine,
                                                                  y=threshold_width_model(model_in_abs_fine, *popt),
                                                                  lc='tab:gray')

        # Fit the threshold and width based psychometric curve to the single fish for lumi right. And store the fitted model parameters.
        popt, pcov = curve_fit(threshold_width_model, model_in_abs, data_single[2::3], p0=params_TW_init)
        single_fish_thresh_LL = np.append(single_fish_thresh_LL, popt[0])
        single_fish_width_LL = np.append(single_fish_width_LL, popt[1])
        # If it is one of the example fish, plot the model curve.
        if single_fish_ID in plot_fish_IDs:
            print(f'TW model single blue: {popt}')
            t3 = popt[0]
            w3 = popt[1]
            subfigs_single_fish[single_fish_plot_count].draw_line(x=x_input_fine,
                                                                  y=threshold_width_model(model_in_abs_fine, *popt),
                                                                  lc='tab:blue')
            single_fish_plot_count += 1

            # If it is one of the example fish, add the parameter estimates as dashed lines to the swarmplot.
            subfig_swarm.draw_line(x=[0, 1, 2], y=[t3, t2, t1], lc='#404040', line_dashes=(3, 2))
            subfig_swarm.draw_line(x=[3, 4, 5], y=[w3, w2, w1], lc='#404040', line_dashes=(3, 2))

    # Draw the parameter estimates of each fish to the swarmplot.
    subfig_swarm.draw_swarmplot(
        [single_fish_thresh_LL, single_fish_thresh_LO, single_fish_thresh_LR, single_fish_width_LL,
         single_fish_width_LO, single_fish_width_LR],
        pc=['tab:blue', 'tab:gray', 'tab:red', 'tab:blue', 'tab:gray', 'tab:red'], ec=None)
    return


def sup_fig_acc_over_coh(path_to_analysed_data, subfig_acc_coh, subfig_swarm):
    '''
    This function plots the average accuracy over coherence data with the fitted threshold and width based psychometric curves.
    It also adds the parameters of the fit to the average data to the swarmplot in Fig. S1b.
    :param path_to_analysed_data: Path the analysed dataframe containing data of all fish.
    :param subfig_acc_coh: Subfigure to plot the average accuracy over coherence data and model fit.
    :param subfig_swarm: Subfigure to plot the parameters of the fit to the average data.
    '''

    # Load the data and reorder from alphabetic to numerical stimulus orde.r
    analysed_df, reorder_stim_names = reorder_analysed_df(path_to_analysed_data)
    # Loop over the stimuli and plot the accuracy for each coherence and luminance level.
    data = []
    for stim_name in reorder_stim_names:
        coherence = int(stim_name.split('_')[0])
        lumi = int(stim_name.split('_')[1])

        # Select the data for this specific stimulus
        stim_df = analysed_df.xs(stim_name, level='stimulus_name')
        n_fish = len(stim_df.index.unique('experiment_ID'))
        grouped_df = stim_df.groupby('window_time')

        # Get the mean and sem values.
        mean = np.array(grouped_df['percentage_left'].mean())[grouped_df['percentage_left'].mean().index == 20][0]
        std = np.array(grouped_df['percentage_left'].std())[grouped_df['percentage_left'].std().index == 20][0]
        sem = std / np.sqrt(n_fish)

        # Plot the scatter point of the accuracy with SEM error bar in the color depending on the luminance level.
        if lumi == -1:
            subfig_acc_coh.draw_scatter(x=coherence, y=mean, yerr=sem, pc='tab:red', lw=0.5, ec='tab:red')
        elif lumi == 1:
            subfig_acc_coh.draw_scatter(x=coherence, y=mean, yerr=sem, pc='tab:blue', lw=0.5, ec='tab:blue')
        if lumi == 0:
            subfig_acc_coh.draw_scatter(x=coherence, y=mean, yerr=sem, pc='tab:gray', lw=0.5, ec='tab:gray')

        data = np.append(data, mean)

    # Get the interpolated fine input to the threshold/width model.
    x_input_fine = np.arange(-100, 101, 2)
    model_in_abs = [-1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1.0]
    model_in_abs_fine = np.arange(-1, 1.01, 0.02)
    # Define the initial parameter guesses of the threshold and width.
    params_TW_init = [0, 0.4]

    # Fit the lumi left curve and plot the fitted model curve.
    popt, pcov = curve_fit(threshold_width_model, model_in_abs, data[::3], p0=params_TW_init)
    t1 = popt[0]
    w1 = popt[1]
    print(f'TW model red: {popt}')
    subfig_acc_coh.draw_line(x=x_input_fine, y=threshold_width_model(model_in_abs_fine, *popt), lc='tab:red')

    # Fit the lumi off curve and plot the fitted model curve.
    popt, pcov = curve_fit(threshold_width_model, model_in_abs, data[1::3], p0=params_TW_init)
    t2 = popt[0]
    w2 = popt[1]
    print(f'TW model gray: {popt}')
    subfig_acc_coh.draw_line(x=x_input_fine, y=threshold_width_model(model_in_abs_fine, *popt), lc='tab:gray')

    # Fit the lumi right curve and plot the fitted model curve.
    popt, pcov = curve_fit(threshold_width_model, model_in_abs, data[2::3], p0=params_TW_init)
    t3 = popt[0]
    w3 = popt[1]
    print(f'TW model blue: {popt}')
    subfig_acc_coh.draw_line(x=x_input_fine, y=threshold_width_model(model_in_abs_fine, *popt), lc='tab:blue')

    # Plot the model parameters (threshold and width) for the three lumi-values.
    subfig_swarm.draw_line(x=[0, 1, 2], y=[t3, t2, t1], lc='k')
    subfig_swarm.draw_line(x=[3, 4, 5], y=[w3, w2, w1], lc='k')
    return


if __name__ == '__main__':
    # Provide the path to save the figure.
    fig_save_path = 'C:/users/katja/Desktop/fig_S1.pdf'

    # Provide the path to the figure_1 folder.
    fig_1_folder_path = r'Z:\Bahl lab member directories\Katja\paper_data\figure_1'

    # Get the path to the folder containing the analysed dataframe related to the data in figure 1.
    path_to_analysed = f'{fig_1_folder_path}\data_analysed.hdf5'

    # Give 6 randomly chosen fish IDs
    plot_fish_IDs = [107, 90, 108, 130, 78, 97]

    # Here we define the figure and subpanel outlines (e.g. the limits, ticks and labels of the axes) beloning to figure S1.
    fig = Figure(fig_width=18, fig_height=12.5)

    # Fig. S1a
    acc_over_coherence_plot = fig.create_plot(xpos=1.5, ypos=8, plot_height=3.75, plot_width=6,
                                              errorbar_area=True,
                                              xl='coherence (%)', xmin=-110, xmax=110,
                                              xticks=[-100, -50, -25, 0, 25, 50, 100], yl='right swims (%)', ymin=0,
                                              ymax=1, yticks=[0, 0.25, 0.50, 0.75, 1],
                                              yticklabels=['0', '25', '50', '75', '100'], hlines=[0.5], vlines=[0],
                                              helper_lines_lc='k', legend_xpos=3.5, helper_lines_dashes=(2, 0), )

    # Fig. S1c
    single_fish_plot1 = fig.create_plot(xpos=1.5, ypos=4, plot_height=2.5, plot_width=4,
                                        errorbar_area=True, xmin=-110, xmax=110, yl='right swims (%)', ymin=0,
                                        ymax=1, yticks=[0, 0.25, 0.50, 0.75, 1],
                                        yticklabels=['0', '25', '50', '75', '100'], hlines=[0.5], vlines=[0],
                                        helper_lines_lc='k', legend_xpos=3.5, helper_lines_dashes=(2, 0))
    single_fish_plot2 = fig.create_plot(xpos=6.5, ypos=4, plot_height=2.5, plot_width=4,
                                        errorbar_area=True, xmin=-110, xmax=110, ymin=0,
                                        ymax=1, hlines=[0.5], vlines=[0], helper_lines_lc='k', legend_xpos=9.5,
                                        helper_lines_dashes=(2, 0))
    single_fish_plot3 = fig.create_plot(xpos=11.5, ypos=4, plot_height=2.5, plot_width=4,
                                        errorbar_area=True, xmin=-110, xmax=110, ymin=0,
                                        ymax=1, hlines=[0.5], vlines=[0], helper_lines_lc='k', legend_xpos=15.5,
                                        helper_lines_dashes=(2, 0))
    single_fish_plot4 = fig.create_plot(xpos=1.5, ypos=1, plot_height=2.5, plot_width=4,
                                        errorbar_area=True, xmin=-110, xmax=110,
                                        xticks=[-100, -50, -25, 0, 25, 50, 100], ymin=0,
                                        ymax=1, yticks=[0, 0.25, 0.50, 0.75, 1],
                                        yticklabels=['0', '25', '50', '75', '100'], hlines=[0.5], vlines=[0],
                                        helper_lines_lc='k', legend_xpos=3.5, helper_lines_dashes=(2, 0))
    single_fish_plot5 = fig.create_plot(xpos=6.5, ypos=1, plot_height=2.5, plot_width=4,
                                        errorbar_area=True, xl='coherence (%)', xmin=-110, xmax=110,
                                        xticks=[-100, -50, -25, 0, 25, 50, 100], ymin=0,
                                        ymax=1, hlines=[0.5], vlines=[0], helper_lines_lc='k', legend_xpos=9.5,
                                        helper_lines_dashes=(2, 0))
    single_fish_plot6 = fig.create_plot(xpos=11.5, ypos=1, plot_height=2.5, plot_width=4,
                                        errorbar_area=True, xmin=-110, xmax=110,
                                        xticks=[-100, -50, -25, 0, 25, 50, 100], ymin=0,
                                        ymax=1, hlines=[0.5], vlines=[0], helper_lines_lc='k', legend_xpos=15.5,
                                        helper_lines_dashes=(2, 0))
    single_fish_plots = [single_fish_plot1, single_fish_plot2, single_fish_plot3, single_fish_plot4, single_fish_plot5,
                         single_fish_plot6]

    # Fig. S1b
    fish_overview_plot = fig.create_plot(xpos=9, ypos=8, plot_height=3.75, plot_width=6, xmin=-1, xmax=6,
                                         xticks=np.arange(6), yl='parameter value', ymin=-3,
                                         ymax=3, yticks=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
                                         xticklabels=['', 'threshold', '', '', 'width', ''],
                                         hlines=[0.], helper_lines_lc='k', legend_xpos=13.5, helper_lines_dashes=(2, 0))

    # Plot the single fish decision curves and parameters (Fig. S1b-c)
    sup_fig_single_fish_analysis(path_to_analysed, single_fish_plots, fish_overview_plot, plot_fish_IDs)

    # Plot the average accuracy over coherence across all fish (Fig. S1a-b)
    sup_fig_acc_over_coh(path_to_analysed, acc_over_coherence_plot, fish_overview_plot)

    fig.save(fig_save_path)
