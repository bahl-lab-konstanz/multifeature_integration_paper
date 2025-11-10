import h5py
import numpy as np
import pandas as pd
import navis
from scipy.stats import linregress
from analysis_helpers.analysis.utils.figure_helper import Figure
from scipy.ndimage import convolve1d
from matplotlib import cm
from scipy.optimize import curve_fit
from matplotlib.colors import ListedColormap
from scipy.stats import ttest_rel
from multifeature_integration_paper.logic_regression_functions import logic_regression_left_motion, logic_regression_left_drive, logic_regression_left_bright, logic_regression_left_dark, logic_regression_left_diff, logic_regression_left_lumi, logic_regression_right_motion, logic_regression_right_drive, logic_regression_right_bright, logic_regression_right_dark, logic_regression_right_diff, logic_regression_right_lumi
from multifeature_integration_paper.logic_regression_functions import logic_regression_right_motion_wta, logic_regression_left_motion_wta, logic_regression_left_lumi_wta, logic_regression_right_lumi_wta, logic_regression_left_lumi_single, logic_regression_right_lumi_single
from multifeature_integration_paper.useful_small_funcs import rolling_end_window, create_combined_region_npy_mask, cohens_d

def get_stim_input(stim_len, stim_type, zero_coh=0.1):
    '''
    This function obtains the input to the models for each experiment stimulus.
    :param stim_len: Length of the stimulus in 0.1s steps (e.g. a 25s stimulus is written here as 250).
    :param stim_type: Stimulus type (Motion, Photo, Same or Oppo)
    :param zero_coh: To avoid divisions by zero we set the perceived motion at 0% coherence to a small value.
    :return: Four input arrays: motion_left, motion_right, luminance_left, luminance_right.
    '''
    # Get the specific parameters of when motion start and stops, luminance starts and stops. The strength factors measured from the data,
    # as well as the luminance factors pre/post stimulus and on the bright and dark side of the stimulus.
    mot_on, mot_off, lumi_on, lumi_off = [100, 400, 100, 400]
    mot_fac, lumi_fac = [1., 1.]
    lumi_pre, lumi_bright, lumi_dark = [0.3, 1., 0.]

    # Get the four inputs for the given stimulus type.
    if stim_type == 'Motion':
        left_input_mot = mot_fac * np.concatenate(
            (zero_coh * np.ones(mot_on), 1. * np.ones(mot_off - mot_on), zero_coh * np.ones(stim_len - mot_off)))
        right_input_mot = mot_fac * zero_coh * np.ones(stim_len)
        left_input_lumi = lumi_fac * lumi_pre * np.ones(stim_len)
        right_input_lumi = lumi_fac * lumi_pre * np.ones(stim_len)
    elif stim_type == 'Photo':
        left_input_mot = mot_fac * zero_coh * np.ones(stim_len)
        right_input_mot = mot_fac * zero_coh * np.ones(stim_len)
        left_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                     lumi_bright * np.ones(lumi_off - lumi_on),
                                                     lumi_pre * np.ones(stim_len - lumi_off)))
        right_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                      lumi_dark * np.ones(lumi_off - lumi_on),
                                                      lumi_pre * np.ones(stim_len - lumi_off)))
    elif stim_type == 'Same':
        left_input_mot = mot_fac * np.concatenate(
            (zero_coh * np.ones(mot_on), 1. * np.ones(mot_off - mot_on), zero_coh * np.ones(stim_len - mot_off)))
        right_input_mot = mot_fac * zero_coh * np.ones(stim_len)
        left_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                     lumi_bright * np.ones(lumi_off - lumi_on),
                                                     lumi_pre * np.ones(stim_len - lumi_off)))
        right_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                      lumi_dark * np.ones(lumi_off - lumi_on),
                                                      lumi_pre * np.ones(stim_len - lumi_off)))
    elif stim_type == 'Oppo':
        left_input_mot = mot_fac * np.concatenate(
            (zero_coh * np.ones(mot_on), 1. * np.ones(mot_off - mot_on), zero_coh * np.ones(stim_len - mot_off)))
        right_input_mot = mot_fac * zero_coh * np.ones(stim_len)
        left_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                     lumi_dark * np.ones(lumi_off - lumi_on),
                                                     lumi_pre * np.ones(stim_len - lumi_off)))
        right_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                      lumi_bright * np.ones(lumi_off - lumi_on),
                                                      lumi_pre * np.ones(stim_len - lumi_off)))

    return left_input_mot, right_input_mot, left_input_lumi, right_input_lumi

def wta_mot_lumi_change(model_input, subfigss, tau_mot=4.26, tau_ph_eye=12.66, tau_ph_rep=12.66, tau_drive=6.46,
                        w_attractor_pos=0.126, w_repulsor_pos=1.857, w_mot=2.783, tau_gcamp=24., kernel_length=150, #tau_gcamp: half decay time of 3.5s = tau of 2.4s
                        window_length=20):
    '''
    This function contains the WTA model with motion, luminance level and changes in luminance. A mutual inhibitory connection between conflicting motion and luminance direction nodes makes the model a WTA model.
    :param model_input: List of 5 input arrays (time, motion left, motion right, luminance left, luminance right).
    :param subfigss: list of subfigures to plot the model prediction into.
    :param tau_mot: Timeconstant of the motion integrator.
    :param tau_ph_eye: Timeconstant of the luminance level integrator.
    :param tau_ph_rep: Timeconstant of the integrator in the luminance change pathway.
    :param tau_drive: Timeconstant of the multifeature integrator.
    :param w_mot: Weight of the motion pathway
    :param w_attractor_pos: Weight of the luminance level pathway.
    :param w_repulsor_pos: Weight of the luminance change pathway.
    :param tau_gcamp: Timeconstant of the GCaMP kernel.
    :param kernel_length: length in 0.1s timesteps of the convolution kernels.
    :param window_length: length of the rolling window used to mimmick the data preprocessing.
    :param linregress: if True plot the model predictions using the regressor resolution of 0.1s timesteps instead of 0.5s.
    :return: GCaMP convolved activity predictions of each model node.
    '''
    # We add some baseline activity to both multifeature nodes to avoid division by zero. Since both nodes contain this baseline, it doesn't affect the percentage left swims.
    baseline = 1
    print(tau_mot,tau_ph_eye, tau_ph_rep, tau_drive, w_attractor_pos, w_repulsor_pos, w_mot, tau_gcamp, kernel_length,window_length)

    time, left_input_mot, right_input_mot, left_input_ph, right_input_ph = model_input

    # Integrate motion
    exp_kernel_mot = np.concatenate((np.zeros(kernel_length), 1/tau_mot * np.exp(-np.linspace(0, kernel_length, kernel_length+1) / tau_mot)))
    exp_kernel_mot = exp_kernel_mot / np.sum(exp_kernel_mot)
    left_integrated_mot = convolve1d(left_input_mot, exp_kernel_mot)
    right_integrated_mot = convolve1d(right_input_mot, exp_kernel_mot)

    # Integrate luminance for each eye
    exp_kernel_ph_eye = np.concatenate((np.zeros(kernel_length), 1/tau_ph_eye * np.exp(-np.linspace(0, kernel_length, kernel_length+1) / tau_ph_eye)))
    exp_kernel_ph_eye = exp_kernel_ph_eye / np.sum(exp_kernel_ph_eye)
    left_integrated_ph = convolve1d(left_input_ph, exp_kernel_ph_eye)
    right_integrated_ph = convolve1d(right_input_ph, exp_kernel_ph_eye)

    # WTA motion vs lumi, to normalize the model activity prior to mutual inhibition we place the pre/post stimulus activity to zero. The motion baseline is 0.1, the lumi baseline is 0.3.
    baseline_left_integrated_mot = (left_integrated_mot - 0.1) / 0.9
    baseline_right_integrated_mot = (right_integrated_mot - 0.1) / 0.9
    baseline_left_integrated_ph = (left_integrated_ph - 0.3) / 0.7
    baseline_right_integrated_ph = (right_integrated_ph - 0.3) / 0.7

    # Compute the mutual inhibition between leftward motion and rightward luminance as well as between rightward motion and leftward luminance.
    left_integrated_mot_bu = baseline_left_integrated_mot - np.clip(baseline_right_integrated_ph, 0, 1) * baseline_left_integrated_mot
    right_integrated_mot_bu = baseline_right_integrated_mot - np.clip(baseline_left_integrated_ph, 0, 1) * baseline_right_integrated_mot
    left_integrated_ph = baseline_left_integrated_ph - baseline_right_integrated_mot * np.clip(baseline_left_integrated_ph, 0, 1)
    right_integrated_ph = baseline_right_integrated_ph - baseline_left_integrated_mot * np.clip(baseline_right_integrated_ph, 0, 1)
    left_integrated_mot = left_integrated_mot_bu
    right_integrated_mot = right_integrated_mot_bu

    # After applying the mutual inhibtion we correct back the normalization to the motion baseline of 0.1, and the luminance baseline of 0.3.
    left_integrated_mot = 0.9*left_integrated_mot + 0.1
    right_integrated_mot = 0.9*right_integrated_mot + 0.1
    left_integrated_ph = 0.7*left_integrated_ph + 0.3
    right_integrated_ph = 0.7*right_integrated_ph + 0.3

    # Calculate repulsion force
    exp_kernel_ph_rep = np.concatenate((np.zeros(kernel_length), 1/tau_ph_rep * np.exp(-np.linspace(0, kernel_length, kernel_length+1) / tau_ph_rep)))
    exp_kernel_ph_rep = exp_kernel_ph_rep / np.sum(exp_kernel_ph_rep)
    left_integrated_rep = convolve1d(left_input_ph, exp_kernel_ph_rep)
    right_integrated_rep = convolve1d(right_input_ph, exp_kernel_ph_rep)

    dark_left = np.clip(left_integrated_rep - left_input_ph, 0, np.inf)
    dark_right = np.clip(right_integrated_rep - right_input_ph, 0, np.inf)
    bright_left = np.clip(left_input_ph - left_integrated_rep, 0, np.inf)
    bright_right = np.clip(right_input_ph - right_integrated_rep, 0, np.inf)
    repulsion_from_left = bright_left + dark_left
    repulsion_from_right = bright_right + dark_right

    # Linear weighted sum going ino the multifeature node (drive)
    drive_to_left = w_mot * left_integrated_mot + w_repulsor_pos * repulsion_from_right + baseline + w_attractor_pos * left_integrated_ph
    drive_to_right = w_mot * right_integrated_mot + w_repulsor_pos * repulsion_from_left + baseline + w_attractor_pos * right_integrated_ph

    # For integration of the drive
    exp_kernel_drive = np.concatenate((np.zeros(kernel_length), 1/tau_drive * np.exp(-np.linspace(0, kernel_length, kernel_length+1) / tau_drive)))
    exp_kernel_drive = exp_kernel_drive / np.sum(exp_kernel_drive)
    left_integrated_drive = convolve1d(drive_to_left, exp_kernel_drive)
    right_integrated_drive = convolve1d(drive_to_right, exp_kernel_drive)

    # Get the ratio of leftward swims
    swims_to_left_tot = ((left_integrated_drive - right_integrated_drive) / (left_integrated_drive + right_integrated_drive) + 1) / 2

    # Apply a rolling window to match the data preprocessing. And transform ratios to percentages.
    swims_to_left_tot_rw = rolling_end_window(swims_to_left_tot, window_length)
    swims_to_left_tot_rw = swims_to_left_tot_rw * 100
    swims_to_left_tot_rw[np.isinf(swims_to_left_tot_rw)] = 50
    swims_to_left_tot_rw[np.isnan(swims_to_left_tot_rw)] = 50

    # Tau GCaMP Migault et al. 2018 for H2B-6s
    exp_kernel_gcamp = np.concatenate((np.zeros(kernel_length), 1/tau_gcamp * np.exp(-np.linspace(0, kernel_length, kernel_length+1) / tau_gcamp)))
    exp_kernel_gcamp = exp_kernel_gcamp / np.sum(exp_kernel_gcamp)

    # If the subfigss is not None and linregress is False plot the model predictions into the subfigss.
    if subfigss is not None:
        # Loop over all model nodes.
        for subfigs, prediction_L, prediction_R, norm_A, baseline in zip(subfigss,
                                                       [convolve1d(left_integrated_mot, exp_kernel_gcamp),
                                                       convolve1d(left_integrated_ph, exp_kernel_gcamp),
                                                       convolve1d(dark_right, exp_kernel_gcamp),
                                                       convolve1d(bright_right, exp_kernel_gcamp),
                                                       convolve1d(drive_to_left, exp_kernel_gcamp),
                                                       convolve1d(repulsion_from_right, exp_kernel_gcamp)],
                                                       [convolve1d(right_integrated_mot, exp_kernel_gcamp),
                                                        convolve1d(right_integrated_ph, exp_kernel_gcamp),
                                                        convolve1d(dark_left, exp_kernel_gcamp),
                                                        convolve1d(bright_left, exp_kernel_gcamp),
                                                        convolve1d(drive_to_right, exp_kernel_gcamp),
                                                        convolve1d(repulsion_from_left, exp_kernel_gcamp)],
                                                        [0.45, 0.6, 0.6, 0.6, 0.6, 0.6],
                                                        [0.0, -0.2, 0.0, 0.0, 0.0, 0.0]):
            # Loop over all nine stimuli to plot the predictions of each node.
            # The predictions are ordered 'Motion only' - Timepoints 0-600, 'Lumi only' - Timepoints 650-1250, 'Congruent' - Timepoints 1300-1900, 'Conflicting' - Timepoints 1950-2550.
            # We flip the direction of the prediction for rightward subfigures, since the predictions are based on leftward model inputs.
            for subfig, start, direction in zip(subfigs, [1300, 1950, 650, 1950, 1300, 650, 0, 0, None],
                                                         ['left', 'right', 'left', 'left', 'right', 'right', 'left', 'right', None]):
                if direction == 'left':
                    norm_pred = (prediction_L[start:start + 600] - np.min([prediction_L, prediction_R])) / (np.max([prediction_L, prediction_R]) - np.min([prediction_L, prediction_R]))
                    subfig.draw_line(np.arange(0, 60, 0.1), baseline + norm_A * norm_pred, lc='k', line_dashes=(4, 2), lw=0.5)

                elif direction == 'right':
                    norm_pred = (prediction_R[start:start + 600] - np.min([prediction_L, prediction_R])) / (np.max([prediction_L, prediction_R]) - np.min([prediction_L, prediction_R]))
                    subfig.draw_line(np.arange(0, 60, 0.1), baseline + norm_A * norm_pred, lc='k', line_dashes=(4, 2), lw=0.5)

                else:
                    norm_pred = (prediction_L[50] - np.min([prediction_L, prediction_R])) / (np.max([prediction_L, prediction_R]) - np.min([prediction_L, prediction_R]))
                    subfig.draw_line(np.arange(0, 60, 0.1), baseline + norm_A * norm_pred * np.ones(600), lc='k', line_dashes=(4, 2), lw=0.5)

    return convolve1d(left_integrated_mot, exp_kernel_gcamp), convolve1d(left_integrated_ph, exp_kernel_gcamp), convolve1d(dark_right, exp_kernel_gcamp), \
           convolve1d(bright_right, exp_kernel_gcamp), convolve1d(drive_to_left, exp_kernel_gcamp), convolve1d(repulsion_from_right, exp_kernel_gcamp), \
           convolve1d(right_integrated_mot, exp_kernel_gcamp), convolve1d(right_integrated_ph, exp_kernel_gcamp), convolve1d(dark_left, exp_kernel_gcamp), \
           convolve1d(bright_left, exp_kernel_gcamp), convolve1d(drive_to_right, exp_kernel_gcamp), convolve1d(repulsion_from_left, exp_kernel_gcamp)

def avg_mot_lumi_change(model_input, subfigss, tau_mot=4.26, tau_ph_eye=12.66, tau_ph_rep=12.66, tau_drive=6.46,
                        w_attractor_pos=0.126, w_repulsor_pos=1.857, w_mot=2.783, tau_gcamp=24., kernel_length=150, #tau_gcamp: half decay time of 3.5s = tau of 2.4s
                        window_length=20, linregress=False):
    '''
    This function contains the additive model with motion, luminance level and changes in luminance.
    :param model_input: List of 5 input arrays (time, motion left, motion right, luminance left, luminance right).
    :param subfigss: list of subfigures to plot the model prediction into.
    :param tau_mot: Timeconstant of the motion integrator.
    :param tau_ph_eye: Timeconstant of the luminance level integrator.
    :param tau_ph_rep: Timeconstant of the integrator in the luminance change pathway.
    :param tau_drive: Timeconstant of the multifeature integrator.
    :param w_mot: Weight of the motion pathway
    :param w_attractor_pos: Weight of the luminance level pathway.
    :param w_repulsor_pos: Weight of the luminance change pathway.
    :param tau_gcamp: Timeconstant of the GCaMP kernel.
    :param kernel_length: length in 0.1s timesteps of the convolution kernels.
    :param window_length: length of the rolling window used to mimmick the data preprocessing.
    :param linregress: if True plot the model predictions using the regressor resolution of 0.1s timesteps instead of 0.5s.
    :return: GCaMP convolved activity predictions of each model node.
    '''
    # We add some baseline activity to both multifeature nodes to avoid division by zero. Since both nodes contain this baseline, it doesn't affect the percentage left swims.
    baseline = 1

    time, left_input_mot, right_input_mot, left_input_ph, right_input_ph = model_input

    # Integrate motion
    exp_kernel_mot = np.concatenate((np.zeros(kernel_length), 1/tau_mot * np.exp(-np.linspace(0, kernel_length, kernel_length+1) / tau_mot)))
    exp_kernel_mot = exp_kernel_mot / np.sum(exp_kernel_mot)
    left_integrated_mot = convolve1d(left_input_mot, exp_kernel_mot)
    right_integrated_mot = convolve1d(right_input_mot, exp_kernel_mot)

    # Integrate luminance for each eye
    exp_kernel_ph_eye = np.concatenate((np.zeros(kernel_length), 1/tau_ph_eye * np.exp(-np.linspace(0, kernel_length, kernel_length+1) / tau_ph_eye)))
    exp_kernel_ph_eye = exp_kernel_ph_eye / np.sum(exp_kernel_ph_eye)
    left_integrated_ph = convolve1d(left_input_ph, exp_kernel_ph_eye)
    right_integrated_ph = convolve1d(right_input_ph, exp_kernel_ph_eye)

    # Calculate repulsion force
    exp_kernel_ph_rep = np.concatenate((np.zeros(kernel_length), 1/tau_ph_rep * np.exp(-np.linspace(0, kernel_length, kernel_length+1) / tau_ph_rep)))
    exp_kernel_ph_rep = exp_kernel_ph_rep / np.sum(exp_kernel_ph_rep)
    left_integrated_rep = convolve1d(left_input_ph, exp_kernel_ph_rep)
    right_integrated_rep = convolve1d(right_input_ph, exp_kernel_ph_rep)

    dark_left = np.clip(left_integrated_rep - left_input_ph, 0, np.inf)
    dark_right = np.clip(right_integrated_rep - right_input_ph, 0, np.inf)
    bright_left = np.clip(left_input_ph - left_integrated_rep, 0, np.inf)
    bright_right = np.clip(right_input_ph - right_integrated_rep, 0, np.inf)
    repulsion_from_left = bright_left + dark_left
    repulsion_from_right = bright_right + dark_right

    # Linear weighted sum going ino the multifeature node (drive)
    drive_to_left = w_mot * left_integrated_mot + w_repulsor_pos * repulsion_from_right + baseline + w_attractor_pos * left_integrated_ph
    drive_to_right = w_mot * right_integrated_mot + w_repulsor_pos * repulsion_from_left + baseline + w_attractor_pos * right_integrated_ph

    # For integration of the drive
    exp_kernel_drive = np.concatenate((np.zeros(kernel_length), 1/tau_drive * np.exp(-np.linspace(0, kernel_length, kernel_length+1) / tau_drive)))
    exp_kernel_drive = exp_kernel_drive / np.sum(exp_kernel_drive)
    left_integrated_drive = convolve1d(drive_to_left, exp_kernel_drive)
    right_integrated_drive = convolve1d(drive_to_right, exp_kernel_drive)

    # Get the ratio of leftward swims
    swims_to_left_tot = ((left_integrated_drive - right_integrated_drive) / (left_integrated_drive + right_integrated_drive) + 1) / 2

    # Apply a rolling window to match the data preprocessing. And transform ratios to percentages.
    swims_to_left_tot_rw = rolling_end_window(swims_to_left_tot, window_length)
    swims_to_left_tot_rw = swims_to_left_tot_rw * 100
    swims_to_left_tot_rw[np.isinf(swims_to_left_tot_rw)] = 50
    swims_to_left_tot_rw[np.isnan(swims_to_left_tot_rw)] = 50

    # Tau GCaMP Migault et al. 2018 for H2B-6s
    exp_kernel_gcamp = np.concatenate((np.zeros(kernel_length), 1/tau_gcamp * np.exp(-np.linspace(0, kernel_length, kernel_length+1) / tau_gcamp)))
    exp_kernel_gcamp = exp_kernel_gcamp / np.sum(exp_kernel_gcamp)

    # If the subfigss is not None and linregress is False plot the model predictions into the subfigss.
    if subfigss is not None and linregress is False:
        # Loop over all model nodes.
        for subfigs, prediction_L, prediction_R, norm_A, baseline in zip(subfigss,
                                                       [convolve1d(left_integrated_mot, exp_kernel_gcamp),
                                                       convolve1d(left_integrated_ph, exp_kernel_gcamp),
                                                       convolve1d(dark_right, exp_kernel_gcamp),
                                                       convolve1d(bright_right, exp_kernel_gcamp),
                                                       convolve1d(drive_to_left, exp_kernel_gcamp),
                                                       convolve1d(repulsion_from_right, exp_kernel_gcamp)],
                                                       [convolve1d(right_integrated_mot, exp_kernel_gcamp),
                                                        convolve1d(right_integrated_ph, exp_kernel_gcamp),
                                                        convolve1d(dark_left, exp_kernel_gcamp),
                                                        convolve1d(bright_left, exp_kernel_gcamp),
                                                        convolve1d(drive_to_right, exp_kernel_gcamp),
                                                        convolve1d(repulsion_from_left, exp_kernel_gcamp)],
                                                        [0.45, 0.6, 0.6, 0.6, 0.6, 0.6],
                                                        [0.0, -0.2, 0.0, 0.0, 0.0, 0.0]):
            # Loop over all nine stimuli to plot the predictions of each node.
            # The predictions are ordered 'Motion only' - Timepoints 0-600, 'Lumi only' - Timepoints 650-1250, 'Congruent' - Timepoints 1300-1900, 'Conflicting' - Timepoints 1950-2550.
            # We flip the direction of the prediction for rightward subfigures, since the predictions are based on leftward model inputs.
            for subfig, start, direction in zip(subfigs, [1300, 1950, 650, 1950, 1300, 650, 0, 0, None],
                                                         ['left', 'right', 'left', 'left', 'right', 'right', 'left', 'right', None]):
                if direction == 'left':
                    norm_pred = (prediction_L[start:start + 600] - np.min([prediction_L, prediction_R])) / (np.max([prediction_L, prediction_R]) - np.min([prediction_L, prediction_R]))
                    subfig.draw_line(np.arange(0, 60, 0.1), baseline + norm_A * norm_pred, lc='k', line_dashes=(4, 2), lw=0.5)

                elif direction == 'right':
                    norm_pred = (prediction_R[start:start + 600] - np.min([prediction_L, prediction_R])) / (np.max([prediction_L, prediction_R]) - np.min([prediction_L, prediction_R]))
                    subfig.draw_line(np.arange(0, 60, 0.1), baseline + norm_A * norm_pred, lc='k', line_dashes=(4, 2), lw=0.5)

                else:
                    norm_pred = (prediction_L[50] - np.min([prediction_L, prediction_R])) / (np.max([prediction_L, prediction_R]) - np.min([prediction_L, prediction_R]))
                    subfig.draw_line(np.arange(0, 60, 0.1), baseline + norm_A * norm_pred * np.ones(600), lc='k', line_dashes=(4, 2), lw=0.5)

    # If subfigss is not None and linregress is True, the predictions are in the correct order and are directly looped over.
    elif subfigss is not None:
        for subfigs, prediction_L, norm_A, baseline in zip(subfigss,
                                           [convolve1d(left_integrated_mot, exp_kernel_gcamp),
                                            convolve1d(left_integrated_ph, exp_kernel_gcamp),
                                            convolve1d(dark_right, exp_kernel_gcamp),
                                            convolve1d(bright_right, exp_kernel_gcamp),
                                            convolve1d(drive_to_left, exp_kernel_gcamp),
                                            convolve1d(repulsion_from_right, exp_kernel_gcamp)],
                                            [0.45, 0.6, 0.6, 0.6, 0.6, 0.6],
                                            [0.0, -0.2, 0.0, 0.0, 0.0, 0.0]):
            for subfig, start in zip(subfigs, [0, 120, 240, 360, 480, 600, 720, 840, 960]):
                norm_pred = (prediction_L[start:start + 120] - np.min(prediction_L)) / (np.max(prediction_L) - np.min(prediction_L))
                subfig.draw_line(np.arange(0, 60, 0.5), baseline + norm_A * norm_pred, lc='k', line_dashes=(4, 2), lw=0.5)

    return convolve1d(left_integrated_mot, exp_kernel_gcamp), convolve1d(left_integrated_ph, exp_kernel_gcamp), convolve1d(dark_right, exp_kernel_gcamp), \
           convolve1d(bright_right, exp_kernel_gcamp), convolve1d(drive_to_left, exp_kernel_gcamp), convolve1d(repulsion_from_right, exp_kernel_gcamp), \
           convolve1d(right_integrated_mot, exp_kernel_gcamp), convolve1d(right_integrated_ph, exp_kernel_gcamp), convolve1d(dark_left, exp_kernel_gcamp), \
           convolve1d(bright_left, exp_kernel_gcamp), convolve1d(drive_to_right, exp_kernel_gcamp), convolve1d(repulsion_from_left, exp_kernel_gcamp)

def get_stim_input_regression(stim_len, stim_type, zero_coh=0.1):
    '''
    This function creates the model inputs for each given stimulus.
    :param stim_len: Length of the stimulus in 0.1s steps (a stimulus of 25s should be written here as 250).
    :param stim_type: Name of the stimulus, chosen from: Motion_L, Motion_R, Photo_L, Photo_R, Same_L, Same_R, Oppo_L, Oppo_R, No_Stim.
    :param zero_coh: The baseline level of perceived motion for 0% coherence. This baseline helps to avoid divisions by zero.
    :return: list of 4 model input arrays: motion_left, motion_right, luminance_left, luminance_right.
    '''

    # Define the stimulus specific parameters: when motion turns on and off, when luminance turns on and off. Motion and luminance factors (set to 1)
    # and the luminance level pre/post stimulus, and at the bright and dark side of the splitview.
    mot_on, mot_off, lumi_on, lumi_off = [20, 80, 20, 80]
    mot_fac, lumi_fac = [1., 1.]
    lumi_pre, lumi_bright, lumi_dark = [0.3, 1., 0.]

    # Create the model inputs for each stimulus type.
    if stim_type == 'Motion_L':
        left_input_mot = mot_fac * np.concatenate(
            (zero_coh * np.ones(mot_on), 1. * np.ones(mot_off - mot_on), zero_coh * np.ones(stim_len - mot_off)))
        right_input_mot = mot_fac * zero_coh * np.ones(stim_len)
        left_input_lumi = lumi_fac * lumi_pre * np.ones(stim_len)
        right_input_lumi = lumi_fac * lumi_pre * np.ones(stim_len)
    elif stim_type == 'Photo_L':
        left_input_mot = mot_fac * zero_coh * np.ones(stim_len)
        right_input_mot = mot_fac * zero_coh * np.ones(stim_len)
        left_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                     lumi_bright * np.ones(lumi_off - lumi_on),
                                                     lumi_pre * np.ones(stim_len - lumi_off)))
        right_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                      lumi_dark * np.ones(lumi_off - lumi_on),
                                                      lumi_pre * np.ones(stim_len - lumi_off)))
    elif stim_type == 'Same_L':
        left_input_mot = mot_fac * np.concatenate(
            (zero_coh * np.ones(mot_on), 1. * np.ones(mot_off - mot_on), zero_coh * np.ones(stim_len - mot_off)))
        right_input_mot = mot_fac * zero_coh * np.ones(stim_len)
        left_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                     lumi_bright * np.ones(lumi_off - lumi_on),
                                                     lumi_pre * np.ones(stim_len - lumi_off)))
        right_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                      lumi_dark * np.ones(lumi_off - lumi_on),
                                                      lumi_pre * np.ones(stim_len - lumi_off)))
    elif stim_type == 'Oppo_L':
        left_input_mot = mot_fac * np.concatenate(
            (zero_coh * np.ones(mot_on), 1. * np.ones(mot_off - mot_on), zero_coh * np.ones(stim_len - mot_off)))
        right_input_mot = mot_fac * zero_coh * np.ones(stim_len)
        left_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                     lumi_dark * np.ones(lumi_off - lumi_on),
                                                     lumi_pre * np.ones(stim_len - lumi_off)))
        right_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                      lumi_bright * np.ones(lumi_off - lumi_on),
                                                      lumi_pre * np.ones(stim_len - lumi_off)))

    elif stim_type == 'Motion_R':
        right_input_mot = mot_fac * np.concatenate(
            (zero_coh * np.ones(mot_on), 1. * np.ones(mot_off - mot_on), zero_coh * np.ones(stim_len - mot_off)))
        left_input_mot = mot_fac * zero_coh * np.ones(stim_len)
        right_input_lumi = lumi_fac * lumi_pre * np.ones(stim_len)
        left_input_lumi = lumi_fac * lumi_pre * np.ones(stim_len)
    elif stim_type == 'Photo_R':
        right_input_mot = mot_fac * zero_coh * np.ones(stim_len)
        left_input_mot = mot_fac * zero_coh * np.ones(stim_len)
        right_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                     lumi_bright * np.ones(lumi_off - lumi_on),
                                                     lumi_pre * np.ones(stim_len - lumi_off)))
        left_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                      lumi_dark * np.ones(lumi_off - lumi_on),
                                                      lumi_pre * np.ones(stim_len - lumi_off)))
    elif stim_type == 'Same_R':
        right_input_mot = mot_fac * np.concatenate(
            (zero_coh * np.ones(mot_on), 1. * np.ones(mot_off - mot_on), zero_coh * np.ones(stim_len - mot_off)))
        left_input_mot = mot_fac * zero_coh * np.ones(stim_len)
        right_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                     lumi_bright * np.ones(lumi_off - lumi_on),
                                                     lumi_pre * np.ones(stim_len - lumi_off)))
        left_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                      lumi_dark * np.ones(lumi_off - lumi_on),
                                                      lumi_pre * np.ones(stim_len - lumi_off)))
    elif stim_type == 'Oppo_R':
        right_input_mot = mot_fac * np.concatenate(
            (zero_coh * np.ones(mot_on), 1. * np.ones(mot_off - mot_on), zero_coh * np.ones(stim_len - mot_off)))
        left_input_mot = mot_fac * zero_coh * np.ones(stim_len)
        right_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                     lumi_dark * np.ones(lumi_off - lumi_on),
                                                     lumi_pre * np.ones(stim_len - lumi_off)))
        left_input_lumi = lumi_fac * np.concatenate((lumi_pre * np.ones(lumi_on),
                                                      lumi_bright * np.ones(lumi_off - lumi_on),
                                                      lumi_pre * np.ones(stim_len - lumi_off)))
    elif stim_type == 'No_Stim':
        right_input_mot = mot_fac * zero_coh * np.ones(stim_len)
        left_input_mot = mot_fac * zero_coh * np.ones(stim_len)
        right_input_lumi = lumi_fac * lumi_pre * np.ones(stim_len)
        left_input_lumi = lumi_fac * lumi_pre * np.ones(stim_len)

    return left_input_mot, right_input_mot, left_input_lumi, right_input_lumi

def create_lumiint_traces_subplots(fig, x_l=7.5, y_t=9., x_ss=1, y_ss=2):
    '''
    This function creates a list with the subfigures for the activity traces of luminance integrators during the luminance integration experiment.
    :param fig: The figure to plot the subfigures in.
    :param x_l: left-most x coordinate of the subfigures.
    :param y_t: Top-most y coordinate of the subfigures.
    :param x_ss: small-step size x between the stimuli panels.
    :return: List of four subfigures for each contrast level.
    '''
    lumi_traces_plota = fig.create_plot(xpos=x_l, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plotb = fig.create_plot(xpos=x_l+x_ss, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plotc = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plotd = fig.create_plot(xpos=x_l, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plote = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])

    lumi_traces_plots = [lumi_traces_plota, lumi_traces_plotb, lumi_traces_plotc, lumi_traces_plotd, lumi_traces_plote]

    subfigs_traces = [lumi_traces_plots, ]

    return subfigs_traces


def create_traces_single_subplots(fig, x_l=7.5, y_t=10., x_ss=0.75, y_ss=0.75, ymax_extra=0):
    '''
    This function creates a list of the subfigures for the activity traces of one functional type.
    :param fig: The figure to plot the subfigures in.
    :param x_l: left-most x coordinate of the subfigures.
    :param y_t: Top-most y coordinate of the subfigures.
    :param x_ss: small-step size x between the stimuli panels.
    :param y_ss: small-step size y between the stimuli panels.
    :param ymax_extra: Option to increase the y-axis by ymax_extra amount
    :return: List of nine subfigures for each functional type.
    '''
    traces_plota = fig.create_plot(xpos=x_l, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.55, ymax=1.2 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_plotb = fig.create_plot(xpos=x_l+x_ss, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.55, ymax=1.2 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_plotc = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.55, ymax=1.2 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_plotd = fig.create_plot(xpos=x_l, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.55, ymax=1.2 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_plote = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.55, ymax=1.2 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_plotf = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.55, ymax=1.2 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_plotg = fig.create_plot(xpos=x_l, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.55, ymax=1.2 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_ploth = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.55, ymax=1.2 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_ploti = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.55, ymax=1.2 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_plots = [traces_plota, traces_plotb, traces_plotc, traces_plotd,
                           traces_plote, traces_plotf, traces_plotg, traces_ploth,
                           traces_ploti]

    return traces_plots

def create_traces_subplots(fig, x_l=7.5, y_t=10., x_ss=1, x_bs=5.40, y_ss=1, y_bs=3.5, wta=False, ymax_extra=0):
    '''
    This function creates a list of lists with the subfigures for the activity traces of each functional type.
    :param fig: The figure to plot the subfigures in.
    :param x_l: left-most x coordinate of the subfigures.
    :param y_t: Top-most y coordinate of the subfigures.
    :param x_ss: small-step size x between the stimuli panels.
    :param x_bs: big-step size x between the functional types.
    :param y_ss: small-step size y between the stimuli panels.
    :param y_bs: big-step size y between the functional types.
    :param wta: If TRUE only subfigures for motion and luminance integrators are made. If FALSE, subfigures for all six functional types are made.
    :param ymax_extra: Option to increase the y-axis by ymax_extra amount
    :return: List of two/six lists of nine subfigures for each functional type.
    '''

    # Create the nine subfigures for motion integrators.
    motion_traces_plota = fig.create_plot(xpos=x_l, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_plotb = fig.create_plot(xpos=x_l+x_ss, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_plotc = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_plotd = fig.create_plot(xpos=x_l, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_plote = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_plotf = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_plotg = fig.create_plot(xpos=x_l, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_ploth = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_ploti = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_plots = [motion_traces_plota, motion_traces_plotb, motion_traces_plotc, motion_traces_plotd,
                           motion_traces_plote, motion_traces_plotf, motion_traces_plotg, motion_traces_ploth,
                           motion_traces_ploti]

    # Create the nine subfigures for luminance integrators.
    lumi_traces_plota = fig.create_plot(xpos=x_l+x_bs, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plotb = fig.create_plot(xpos=x_l+x_bs+x_ss, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plotc = fig.create_plot(xpos=x_l+x_bs+x_ss+x_ss, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plotd = fig.create_plot(xpos=x_l+x_bs, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plote = fig.create_plot(xpos=x_l+x_bs+x_ss, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plotf = fig.create_plot(xpos=x_l+x_bs+x_ss+x_ss, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plotg = fig.create_plot(xpos=x_l+x_bs, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_ploth = fig.create_plot(xpos=x_l+x_bs+x_ss, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_ploti = fig.create_plot(xpos=x_l+x_bs+x_ss+x_ss, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plots = [lumi_traces_plota, lumi_traces_plotb, lumi_traces_plotc, lumi_traces_plotd,
                        lumi_traces_plote, lumi_traces_plotf, lumi_traces_plotg, lumi_traces_ploth,
                        lumi_traces_ploti]

    if not wta:
        # Create the nine subfigures for luminance decrease detectors.
        dark_traces_plota = fig.create_plot(xpos=x_l, ypos=y_t-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_plotb = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_plotc = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_plotd = fig.create_plot(xpos=x_l, ypos=y_t-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_plote = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_plotf = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_plotg = fig.create_plot(xpos=x_l, ypos=y_t-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_ploth = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_ploti = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_plots = [dark_traces_plota, dark_traces_plotb, dark_traces_plotc, dark_traces_plotd,
                               dark_traces_plote, dark_traces_plotf, dark_traces_plotg, dark_traces_ploth,
                               dark_traces_ploti]

        # Create the nine subfigures for luminance increase detectors.
        bright_traces_plota = fig.create_plot(xpos=x_l+x_bs, ypos=y_t-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_plotb = fig.create_plot(xpos=x_l+x_bs+x_ss, ypos=y_t-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_plotc = fig.create_plot(xpos=x_l+x_bs+x_ss+x_ss, ypos=y_t-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_plotd = fig.create_plot(xpos=x_l+x_bs, ypos=y_t-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_plote = fig.create_plot(xpos=x_l+x_bs+x_ss, ypos=y_t-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_plotf = fig.create_plot(xpos=x_l+x_bs+x_ss+x_ss, ypos=y_t-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_plotg = fig.create_plot(xpos=x_l+x_bs, ypos=y_t-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_ploth = fig.create_plot(xpos=x_l+x_bs+x_ss, ypos=y_t-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_ploti = fig.create_plot(xpos=x_l+x_bs+x_ss+x_ss, ypos=y_t-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_plots = [bright_traces_plota, bright_traces_plotb, bright_traces_plotc, bright_traces_plotd,
                             bright_traces_plote, bright_traces_plotf, bright_traces_plotg, bright_traces_ploth,
                             bright_traces_ploti]

        # Create the nine subfigures for multifeature integrators.
        drive_traces_plota = fig.create_plot(xpos=x_l+x_bs, ypos=y_t-y_bs-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_plotb = fig.create_plot(xpos=x_l+x_bs+x_ss, ypos=y_t-y_bs-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_plotc = fig.create_plot(xpos=x_l+x_bs+x_ss+x_ss, ypos=y_t-y_bs-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_plotd = fig.create_plot(xpos=x_l+x_bs, ypos=y_t-y_bs-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_plote = fig.create_plot(xpos=x_l+x_bs+x_ss, ypos=y_t-y_bs-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_plotf = fig.create_plot(xpos=x_l+x_bs+x_ss+x_ss, ypos=y_t-y_bs-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_plotg = fig.create_plot(xpos=x_l+x_bs, ypos=y_t-y_bs-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_ploth = fig.create_plot(xpos=x_l+x_bs+x_ss, ypos=y_t-y_bs-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_ploti = fig.create_plot(xpos=x_l+x_bs+x_ss+x_ss, ypos=y_t-y_bs-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_plots = [drive_traces_plota, drive_traces_plotb, drive_traces_plotc, drive_traces_plotd,
                               drive_traces_plote, drive_traces_plotf, drive_traces_plotg, drive_traces_ploth,
                               drive_traces_ploti]

        # Create the nine subfigures for luminance change detectors.
        diff_traces_plota = fig.create_plot(xpos=x_l, ypos=y_t-y_bs-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_plotb = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_bs-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_plotc = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t-y_bs-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_plotd = fig.create_plot(xpos=x_l, ypos=y_t-y_bs-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_plote = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_bs-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_plotf = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t-y_bs-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_plotg = fig.create_plot(xpos=x_l, ypos=y_t-y_bs-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_ploth = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_bs-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_ploti = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t-y_bs-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_plots = [diff_traces_plota, diff_traces_plotb, diff_traces_plotc, diff_traces_plotd,
                             diff_traces_plote, diff_traces_plotf, diff_traces_plotg, diff_traces_ploth,
                             diff_traces_ploti]

        subfigs_traces = [motion_traces_plots, lumi_traces_plots, dark_traces_plots, bright_traces_plots, drive_traces_plots, diff_traces_plots]
    else:
        subfigs_traces = [motion_traces_plots, lumi_traces_plots]

    return subfigs_traces

def create_locs_subplots(fig, x_l=10.5, y_t=8.5, wta=False):
    '''
    This function creates the list of subfigures to plot the location of functional types.
    :param fig: Figure to add the subpanel to.
    :param x_l: Left-most x coordinate of the subfigures.
    :param y_t: Top-most y coordinate of the subfigures.
    :param wta: If True, only 2 subfigures are made (for motion and lumi WTA neurons) else 6 subfigures are made (for all six functional types).
    :return: list of subfigures.
    '''
    # Define the big-step sizes between the subfigures.
    x_bs = 5.4
    y_bs = 3.5

    # Create the motion and luminance integrator subfigures.
    motion_locs_plot = fig.create_plot(xpos=x_l, ypos=y_t, plot_height=2, plot_width=2, axis_off=True,
                                          xmin=30, xmax=800, ymin=850, ymax=80,
                                       legend_xpos=x_l-2.75, legend_ypos=y_t+2.75)
    lumi_locs_plot = fig.create_plot(xpos=x_l+x_bs, ypos=y_t, plot_height=2, plot_width=2, axis_off=True,
                                          xmin=30, xmax=800, ymin=850, ymax=80,
                                       legend_xpos=x_l+x_bs-2.75, legend_ypos=y_t+2.75)

    if not wta:
        # Create the other four subfigures (lumi decrease/increase/change detectors and multifeature integrators).
        dark_locs_plot = fig.create_plot(xpos=x_l, ypos=y_t - y_bs, plot_height=2, plot_width=2, axis_off=True,
                                         xmin=30, xmax=800, ymin=850, ymax=80,
                                         legend_xpos=x_l - 2.75, legend_ypos=y_t - y_bs + 2.75)
        bright_locs_plot = fig.create_plot(xpos=x_l + x_bs, ypos=y_t - y_bs, plot_height=2, plot_width=2, axis_off=True,
                                           xmin=30, xmax=800, ymin=850, ymax=80,
                                           legend_xpos=x_l + x_bs - 2.75, legend_ypos=y_t - y_bs + 2.75)
        drive_locs_plot = fig.create_plot(xpos=x_l+x_bs, ypos=y_t-y_bs-y_bs, plot_height=2, plot_width=2, axis_off=True,
                                              xmin=30, xmax=800, ymin=850, ymax=80,
                                           legend_xpos=x_l+x_bs-2.75, legend_ypos=y_t-y_bs-y_bs+2.75)
        diff_locs_plot = fig.create_plot(xpos=x_l, ypos=y_t-y_bs-y_bs, plot_height=2, plot_width=2, axis_off=True,
                                              xmin=30, xmax=800, ymin=850, ymax=80,
                                           legend_xpos=x_l-2.75, legend_ypos=y_t-y_bs-y_bs+2.75)

        subfigs_locs = [motion_locs_plot, lumi_locs_plot, dark_locs_plot, bright_locs_plot, drive_locs_plot, diff_locs_plot]
    else:
        subfigs_locs = [motion_locs_plot, lumi_locs_plot,]

    return subfigs_locs

def sub_plot_traces(traces_df, subfigss, subfigs_loc, subfig_loc_comb,
                    thresh_resp=0.2, thresh_min=0.1, thresh_peaks_diff=1.25, thresh_peaks=1.5, thresh_below=0.9):
    '''

    :param traces_df: dataframe containing all functional activity with on each row a neuron.
    :param subfigss: List of subfigures to plot the activity traces per functional type.
    :param subfigs_loc: List of subfigures to plot the location of neurons of each functional type.
    :param subfig_loc_comb: Subfigure to plot the location of neurons of all functional types (labeled by color).
    :param thresh_resp: minimum dF/F activity required to be considered part of the functional types.
    :param thresh_min: During non-responses activity cannot go thresh_min above the pre/post stimulus activity.
    :param thresh_peaks_diff: The change detector peak activity of the stronger contrast needs to be thresh_peaks_diff times higher than the peak activity during the weak contrast stimulus.
    :param thresh_peaks: The increase/decrease detector peak activity of the stronger contrast needs to be thresh_peaks_diff times higher than the peak activity during the weak contrast stimulus.
    :param thresh_below: The decrease in activity of the luminance integrators needs to go thresh_below times lower than the pre/post stimulus activity.
    '''
    # Load the data for each functional type.
    print('Finding medium threshold cells')
    motion_left_df = logic_regression_left_motion(traces_df, thresh_resp=thresh_resp, thresh_min=thresh_min, shuffle_stim_idx=False)
    drive_left_df = logic_regression_left_drive(traces_df, thresh_resp=thresh_resp, shuffle_stim_idx=False)
    lumi_left_df = logic_regression_left_lumi(traces_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min, shuffle_stim_idx=False)
    diff_left_df = logic_regression_left_diff(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks_diff, shuffle_stim_idx=False)
    bright_left_df = logic_regression_left_bright(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
    dark_left_df = logic_regression_left_dark(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
    motion_right_df = logic_regression_right_motion(traces_df, thresh_resp=thresh_resp, thresh_min=thresh_min, shuffle_stim_idx=False)
    drive_right_df = logic_regression_right_drive(traces_df, thresh_resp=thresh_resp, shuffle_stim_idx=False)
    lumi_right_df = logic_regression_right_lumi(traces_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min, shuffle_stim_idx=False)
    diff_right_df = logic_regression_right_diff(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks_diff, shuffle_stim_idx=False)
    bright_right_df = logic_regression_right_bright(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
    dark_right_df = logic_regression_right_dark(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)

    print(f'There are {len(traces_df)} total neurons')
    print(f'There are {len(motion_left_df)} motion neurons, {len(drive_left_df)} drive neurons and {len(np.intersect1d(motion_left_df.index, drive_left_df.index))} overlapping neurons. ')
    print(f'There are {len(lumi_left_df)} lumi neurons, {len(drive_left_df)} drive neurons and {len(np.intersect1d(lumi_left_df.index, drive_left_df.index))} overlapping neurons. ')
    print(f'There are {len(lumi_left_df)} lumi neurons, {len(motion_left_df)} motion neurons and {len(np.intersect1d(lumi_left_df.index, motion_left_df.index))} overlapping neurons. ')
    print(f'There are {len(bright_left_df)} bright neurons, {len(diff_left_df)} diff neurons and {len(np.intersect1d(bright_left_df.index, diff_left_df.index))} overlapping neurons. ')
    print(f'There are {len(dark_left_df)} dark neurons, {len(diff_left_df)} diff neurons and {len(np.intersect1d(dark_left_df.index, diff_left_df.index))} overlapping neurons. ')
    print(f'There are {len(dark_left_df)} dark neurons, {len(bright_left_df)} bright neurons and {len(np.intersect1d(dark_left_df.index, bright_left_df.index))} overlapping neurons. ')

    # Loop over each functional type, combine the left and right dataframe.
    for dfL, dfR, subfigs, color, fillcolor in zip([motion_left_df, lumi_left_df, dark_left_df, bright_left_df, drive_left_df, diff_left_df],
                                                    [motion_right_df, lumi_right_df, dark_right_df, bright_right_df, drive_right_df, diff_right_df], subfigss,
                                             ['#359B73', '#E69F00',  '#9F0162', '#F748A5', '#2271B2', '#D55E00'],
                                             ['#8DCDB4', '#F7D280', '#CC7CAD', '#F7A4D0', '#93BADA', '#EEAE7C']):
        # Loop over each stimulus. The right stimuli are flipped to merge the left and right dataframes.
        for subfig, stimL, stimR in zip(subfigs, ['lumi_left_dots_left',  'lumi_left_dots_right',  'lumi_left_dots_off',
                                          'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                                          'lumi_off_dots_left',   'lumi_off_dots_right',   'lumi_off_dots_off'],
                                        ['lumi_right_dots_right', 'lumi_right_dots_left', 'lumi_right_dots_off',
                                         'lumi_left_dots_right', 'lumi_left_dots_left', 'lumi_left_dots_off',
                                         'lumi_off_dots_right', 'lumi_off_dots_left', 'lumi_off_dots_off']
                                        ):
            # Plot the median activity trace across all neurons with the quartile range.
            subfig.draw_line(np.arange(0, 60, 0.5), [np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)],
                             yerr_neg = np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 25) for i in range(120)]),
                             yerr_pos = np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 75) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]),
                             lc=color, lw=1, eafc=fillcolor, eaalpha=1.0, ealw=1, eaec=fillcolor)

        subfigs[6].draw_line([-4, -4], [0, 0.5], lc='k')
        subfigs[6].draw_text(-20, 0., '0.5 dF/F\u2080', textlabel_rotation=90, textlabel_va="bottom")

    subfigss[4][8].draw_line([40, 60], [-0.34, -0.34], lc='k')
    subfigss[4][8].draw_text(50, -0.6, '20s')

    # Loop over all functional types to plot the location.
    for dfL, dfR, subfig, color, intens_color, label in zip(
            [motion_left_df, lumi_left_df, dark_left_df, bright_left_df, drive_left_df, diff_left_df],
            [motion_right_df, lumi_right_df, dark_right_df, bright_right_df, drive_right_df, diff_right_df], subfigs_loc,
            ['#8DCDB4', '#F7D280', '#CC7CAD', '#F7A4D0', '#93BADA', '#EEAE7C'],
            ['#359B73', '#E69F00', '#9F0162', '#F748A5', '#2271B2', '#D55E00'],
            ['Motion integrators', 'Luminance integrators', 'Luminance decrease detectors',
             'Luminance increase detectors', 'Multifeature integrators', 'Luminance change detectors']    ):

        # The left neurons are plotted as solid circles, the right neurons are plotted as open circles. Here split with a seperate figure for each functional type.
        subfig.draw_scatter(dfL['ZB_x'].astype(float)*0.798, dfL['ZB_y'].astype(float)*0.798, label=label, pc=intens_color, ec=intens_color, ps=0.75, elw=0.25, alpha=0.75)
        subfig.draw_scatter(dfL['ZB_z'].astype(float)*2 + 515, dfL['ZB_y'].astype(float)*0.798, pc=intens_color, ec=intens_color, ps=0.75, elw=0.25, alpha=0.75)
        subfig.draw_scatter(dfR['ZB_x'].astype(float)*0.798, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=intens_color, ps=0.75, elw=0.25, alpha=0.75)
        subfig.draw_scatter(dfR['ZB_z'].astype(float)*2 + 515, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=intens_color, ps=0.75, elw=0.25, alpha=0.75)

        # Draw red boxes as outlines to later fit the ZBRAIN cartoon outline.
        subfig.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
        subfig.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')

        # The left neurons are plotted as solid circles, the right neurons are plotted as open circles. Here all functional types go together in a single location plot (xy and zy view).
        subfig_loc_comb.draw_scatter(dfL['ZB_x'].astype(float)*0.798, dfL['ZB_y'].astype(float)*0.798, pc=intens_color, ec=intens_color, elw=0.25, ps=0.75, alpha=0.75)
        subfig_loc_comb.draw_scatter(dfL['ZB_z'].astype(float)*2 + 515, dfL['ZB_y'].astype(float)*0.798, pc=intens_color, ec=intens_color, elw=0.25, ps=0.75, alpha=0.75)
        subfig_loc_comb.draw_scatter(dfR['ZB_x'].astype(float)*0.798, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=intens_color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc_comb.draw_scatter(dfR['ZB_z'].astype(float)*2 + 515, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=intens_color, ps=0.75, elw=0.25, alpha=0.75)

    # Draw red boxes as outlines to later fit the ZBRAIN cartoon outline.
    subfig_loc_comb.draw_text(387.5, 50, 'selected cells')
    subfig_loc_comb.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
    subfig_loc_comb.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')
    subfigs_loc[4].draw_line([420, 520], [780, 780], lc='k')
    subfigs_loc[4].draw_text(470, 820, '100\u00b5m')

    return

def linear_regression(df, regressors, rval_thresh=0.85):
    '''
    This function performs the linear regression between the data and model-based regressors.
    :param df: traces_df containing all functional activity with on each row a neuron.
    :param regressors: The model based regressors.
    :param rval_thresh: The threshold based on which a cell is considered a good fit with the regressors (between -1 and 1).
    :return: all_dfs containing the dfs of all functional types with all fitting neurons and all_unique_dfs containing the dfs of all functional types with only the neurons that are not part of other functional types.
    '''

    # Define the stimulus names.
    stims = ['lumi_left_dots_left',  'lumi_left_dots_right',  'lumi_left_dots_off',
             'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
             'lumi_off_dots_left',   'lumi_off_dots_right',   'lumi_off_dots_off']

    # Loop through all the cells and perform linear regression.
    print(f'Regressing through {len(df)} cells')
    good_cells = [[]] * len(regressors)
    cell_type = [[]] * len(df)
    keep_cell = np.ones(len(df))
    for cell in range(len(df)):
        if cell % 1000 == 0:
            print(cell)
        # Concatenate the average traces to all stimuli.
        trace = []
        for stim in stims:
            trace = np.append(trace, np.array([df[f'{stim}_avg_trace_{i}'][cell] for i in range(120)]))

        # Normalize the functional trace.
        norm_trace = (trace - np.nanpercentile(trace, 5)) / (np.nanpercentile(trace, 95) - np.nanpercentile(trace, 5))

        # Loop over all regressors and perform the linear regression.
        all_coefs = [[]] * len(regressors)
        valid_cell = np.zeros(len(regressors))
        for r, regressor in enumerate(regressors):
            try:
                slope, intercept, r_value, p_value, std_err = linregress(regressor[~np.isnan(norm_trace)],
                                                                         norm_trace[~np.isnan(norm_trace)])
            except:
                print('linregress failed. ')
                valid_cell[r] = 1
                continue

            # Check if the regression score is good enough and keep track of the good cells.
            if r_value > rval_thresh:
                all_coefs[r] = r_value
            else:
                all_coefs[r] = -1
            if r_value > rval_thresh:
                good_cells[r] = np.append(good_cells[r], 1)
            else:
                good_cells[r] = np.append(good_cells[r], 0)

        # Sanity check, if all of the linear regressions failed or didn't pass the regression threshold once - the neuron gets cell-type -1 (no functional type).
        if np.sum(valid_cell) == len(regressors) or np.max(all_coefs) == -1:
            cell_type[cell] = -1
            if np.sum(valid_cell) == len(regressors):
                keep_cell[cell] = 0
        else:
            cell_type[cell] = np.argmax(all_coefs)

    # Combine the good cells into the final dataframes keeping track of all neurons per type and all unique neurons (that are only part of one functional type) per type.
    print('Found all regressions, saving into dfs. ')
    all_dfs = [[]] * len(regressors)
    all_unique_dfs = [[]] * len(regressors)
    for r in range(len(regressors)):
        df_kc = df[keep_cell.astype(bool)]
        all_dfs[r] = df_kc[np.logical_and(good_cells[r] >= 1, df_kc['ZB_z'].astype(float) > 0)]
        all_unique_dfs[r] = df_kc[np.logical_and(np.array(cell_type)[keep_cell.astype(bool)] == r, df_kc['ZB_z'].astype(float) > 0)]

    return all_dfs, all_unique_dfs


def sub_plot_linear_regression_traces(traces_df, subfigss, subfig_loc, subfigoverlap, model_params=[5.73/5, 2.88/5, 14.54/5, 7.61/5, 0.214, 1.922, 2.88]):
    '''
    This function plots the linear regression based traces, and neuron locations, as well as the number of neurons found.
    This is related to figure S4a-b.
    :param traces_df: The dataframe containing all functional traces. Each row is a neuron.
    :param subfigss: List of subfigures that will show the functional traces of the linear regression based types.
    :param subfig_loc: Subfigure showing the location of linear regression based neurons.
    :param subfigoverlap: Subfigure containing the number of neurons for each functional type found by different analysis methods.
    :param model_params: List of model parameters. Note that the linear regressors are made with timestep 0.1s, instead of 0.5s, therefore the timeconstants (first 4 parameters) are divided by 5.
    :return:
    '''

    # Define the length of the stimulus as well as the stimulus types.
    stim_len_timepoints = [120]
    stim_names = ['Same_L', 'Oppo_R', 'Photo_L', 'Oppo_L', 'Same_R', 'Photo_R', 'Motion_L', 'Motion_R', 'No_Stim']

    # Loop over the stimuli to obtain and concatenate the model input
    for i in range(len(stim_len_timepoints) * len(stim_names)):
        folder_id = i % len(stim_len_timepoints)
        stim_id = int(i / len(stim_len_timepoints))
        # Load the model input for this stimulus
        left_mot_input, right_mot_input, left_lumi_input, right_lumi_input = get_stim_input_regression(stim_len_timepoints[folder_id],
                                                                                                       stim_names[stim_id])

        time = np.linspace(0, (stim_len_timepoints[folder_id] - 1) / 2, stim_len_timepoints[folder_id])

        # Concatenate the total model input.
        if i == 0:
            model_input = [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]
        else:
            model_input = np.hstack((model_input, [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]))

    # Create the model-based regressors (=model node activity convolved with a GCaMP kernel).
    regressor_left_mot, regressor_left_lumi, regressor_left_dark, regressor_left_bright, regressor_left_drive, regressor_left_diff,\
        regressor_right_mot, regressor_right_lumi, regressor_right_dark, regressor_right_bright, regressor_right_drive, regressor_right_diff = avg_mot_lumi_change(model_input, None, *model_params, tau_gcamp=24./5, kernel_length=int(150/5), window_length=int(20/5))

    # Normalize the regressors.
    regressor_left_mot = (regressor_left_mot - np.nanmin(regressor_left_mot)) / (np.nanmax(regressor_left_mot) - np.nanmin(regressor_left_mot))
    regressor_left_drive = (regressor_left_drive - np.nanmin(regressor_left_drive)) / (np.nanmax(regressor_left_drive) - np.nanmin(regressor_left_drive))
    regressor_left_lumi = (regressor_left_lumi - np.nanmin(regressor_left_lumi)) / (np.nanmax(regressor_left_lumi) - np.nanmin(regressor_left_lumi))
    regressor_left_diff = (regressor_left_diff - np.nanmin(regressor_left_diff)) / (np.nanmax(regressor_left_diff) - np.nanmin(regressor_left_diff))
    regressor_left_bright = (regressor_left_bright - np.nanmin(regressor_left_bright)) / (np.nanmax(regressor_left_bright) - np.nanmin(regressor_left_bright))
    regressor_left_dark = (regressor_left_dark - np.nanmin(regressor_left_dark)) / (np.nanmax(regressor_left_dark) - np.nanmin(regressor_left_dark))
    regressor_right_mot = (regressor_right_mot - np.nanmin(regressor_right_mot)) / (np.nanmax(regressor_right_mot) - np.nanmin(regressor_right_mot))
    regressor_right_drive = (regressor_right_drive - np.nanmin(regressor_right_drive)) / (np.nanmax(regressor_right_drive) - np.nanmin(regressor_right_drive))
    regressor_right_lumi = (regressor_right_lumi - np.nanmin(regressor_right_lumi)) / (np.nanmax(regressor_right_lumi) - np.nanmin(regressor_right_lumi))
    regressor_right_diff = (regressor_right_diff - np.nanmin(regressor_right_diff)) / (np.nanmax(regressor_right_diff) - np.nanmin(regressor_right_diff))
    regressor_right_bright = (regressor_right_bright - np.nanmin(regressor_right_bright)) / (np.nanmax(regressor_right_bright) - np.nanmin(regressor_right_bright))
    regressor_right_dark = (regressor_right_dark - np.nanmin(regressor_right_dark)) / (np.nanmax(regressor_right_dark) - np.nanmin(regressor_right_dark))

    # Combining the regressors into integrators and detectors.
    regressors_integrators = [regressor_left_mot, regressor_left_drive, regressor_left_lumi,
                  regressor_right_mot, regressor_right_drive, regressor_right_lumi, ]
    regressors_change = [ regressor_left_diff, regressor_left_bright, regressor_left_dark,
                  regressor_right_diff, regressor_right_bright, regressor_right_dark]

    print('Finding linear regression based cells')  # For the integrators
    (motion_left_med_df, drive_left_med_df, lumi_left_med_df, motion_right_med_df, drive_right_med_df, lumi_right_med_df), \
        (motion_left_umed_df, drive_left_umed_df, lumi_left_umed_df, motion_right_umed_df, drive_right_umed_df, lumi_right_umed_df), \
        = linear_regression(traces_df, regressors_integrators, rval_thresh=0.8)
    # Finding linear regression based cells for the detectors. Since transient activity is more difficult to detect by linear regression, we lowered the threshold as compared to the integrators with persistent activity.
    (diff_left_med_df, bright_left_med_df,dark_left_med_df, diff_right_med_df, bright_right_med_df, dark_right_med_df), \
        (diff_left_umed_df, bright_left_umed_df,dark_left_umed_df, diff_right_umed_df, bright_right_umed_df, dark_right_umed_df) \
        = linear_regression(traces_df, regressors_change, rval_thresh=0.6)

    print(f'There are {len(traces_df)} total neurons. ')

    # Compute the amount of neurons per functional type, the unique numbers as well as the overlap.
    num_motion_drive_overlap = len(np.intersect1d(motion_left_med_df.index, drive_left_med_df.index)) + len(np.intersect1d(motion_right_med_df.index, drive_right_med_df.index))
    num_lumi_drive_overlap = len(np.intersect1d(lumi_left_med_df.index, drive_left_med_df.index)) + len(np.intersect1d(lumi_right_med_df.index, drive_right_med_df.index))
    num_motion_lumi_overlap = len(np.intersect1d(lumi_left_med_df.index, motion_left_med_df.index)) + len(np.intersect1d(lumi_right_med_df.index, motion_right_med_df.index))
    num_motion_drive_lumi_overlap = len(np.intersect1d(np.intersect1d(motion_left_med_df.index, drive_left_med_df.index), lumi_left_med_df.index)) + \
                                    len(np.intersect1d(np.intersect1d(motion_right_med_df.index, drive_right_med_df.index), lumi_right_med_df.index))
    num_bright_diff_overlap = len(np.intersect1d(bright_left_med_df.index, diff_left_med_df.index)) + len(np.intersect1d(bright_right_med_df.index, diff_right_med_df.index))
    num_dark_diff_overlap = len(np.intersect1d(dark_left_med_df.index, diff_left_med_df.index)) + len(np.intersect1d(dark_right_med_df.index, diff_right_med_df.index))
    num_bright_dark_overlap = len(np.intersect1d(dark_left_med_df.index, bright_left_med_df.index)) + len(np.intersect1d(dark_right_med_df.index, bright_right_med_df.index))
    num_diff_bright_dark_overlap = len(np.intersect1d(np.intersect1d(bright_left_med_df.index, diff_left_med_df.index), dark_left_med_df.index)) + \
                                   len(np.intersect1d(np.intersect1d(bright_right_med_df.index, diff_right_med_df.index), dark_right_med_df.index))
    num_motion = len(motion_left_med_df) + len(motion_right_med_df) - num_motion_drive_overlap - num_motion_lumi_overlap - num_motion_drive_lumi_overlap
    num_drive = len(drive_left_med_df) + len(drive_right_med_df)  - num_motion_drive_overlap - num_lumi_drive_overlap - num_motion_drive_lumi_overlap
    num_lumi = len(lumi_left_med_df) + len(lumi_right_med_df)  - num_motion_lumi_overlap - num_lumi_drive_overlap - num_motion_drive_lumi_overlap
    num_diff = len(diff_left_med_df) + len(diff_right_med_df)  - num_bright_diff_overlap - num_dark_diff_overlap - num_diff_bright_dark_overlap
    num_bright = len(bright_left_med_df) + len(bright_right_med_df)  - num_bright_diff_overlap - num_bright_dark_overlap - num_diff_bright_dark_overlap
    num_dark = len(dark_left_med_df) + len(dark_right_med_df)  - num_dark_diff_overlap - num_bright_dark_overlap - num_diff_bright_dark_overlap

    print(f'{len(motion_left_med_df) + len(motion_right_med_df)} total motion neurons')
    print(f'{len(drive_left_med_df) + len(drive_right_med_df)} total drive neurons')
    print(f'{len(lumi_left_med_df) + len(lumi_right_med_df)} total lumi neurons')
    print(f'{len(diff_left_med_df) + len(diff_right_med_df)} total diff neurons')
    print(f'{len(bright_left_med_df) + len(bright_right_med_df)} total bright neurons')
    print(f'{len(dark_left_med_df) + len(dark_right_med_df)} total dark neurons')
    print(f'{num_motion_drive_overlap} motion drive overlap')
    print(f'{num_motion_lumi_overlap} motion lumi overlap')
    print(f'{num_lumi_drive_overlap} lumi drive overlap')
    print(f'{num_bright_diff_overlap} bright diff overlap')
    print(f'{num_dark_diff_overlap} dark diff overlap')
    print(f'{num_bright_dark_overlap} bright dark overlap')
    print(f'{num_motion} unique motion neurons')
    print(f'{num_lumi} unique lumi neurons')
    print(f'{num_drive} unique drive neurons')
    print(f'{num_diff} unique diff neurons')
    print(f'{num_bright} unique bright neurons')
    print(f'{num_dark} unique dark neurons')

    # Plot the number of numbers for each type in Fig. S4b as vertical bar plots. The overlapping sections will be gray. We add the mixed color later in Affinity.
    # When the subfigss are None, it means the function is called with the control traces df and we will add only the numbers (to columns 9 and 10).
    if subfigss is None and subfig_loc is None:
        subfigoverlap.draw_vertical_bars([9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, ],
                                         [num_lumi, num_motion_lumi_overlap, num_motion, num_motion_drive_overlap,
                                          num_drive, num_lumi_drive_overlap, num_motion_drive_lumi_overlap,
                                          num_diff, num_bright_diff_overlap, num_bright, num_bright_dark_overlap,
                                          num_dark, num_dark_diff_overlap, num_diff_bright_dark_overlap],
                                         vertical_bar_bottom=[0, num_lumi, num_lumi + num_motion_lumi_overlap,
                                                              num_lumi + num_motion_lumi_overlap + num_motion,
                                                              num_lumi + num_motion_lumi_overlap + num_motion + num_motion_drive_overlap,
                                                              num_lumi + num_motion_lumi_overlap + num_motion + num_motion_drive_overlap + num_drive,
                                                              num_lumi + num_motion_lumi_overlap + num_motion + num_motion_drive_overlap + num_drive + num_lumi_drive_overlap,
                                                              0, num_diff, num_diff + num_bright_diff_overlap,
                                                              num_diff + num_bright_diff_overlap + num_bright,
                                                              num_diff + num_bright_diff_overlap + num_bright + num_bright_dark_overlap,
                                                              num_diff + num_bright_diff_overlap + num_bright + num_bright_dark_overlap + num_dark,
                                                              num_diff + num_bright_diff_overlap + num_bright + num_bright_dark_overlap + num_dark + num_dark_diff_overlap],
                                         lc=['#E69F00', '#808080', '#359B73', '#808080', '#2271B2', '#808080',
                                             '#404040',
                                             '#D55E00', '#808080', '#F748A5', '#808080', '#9F0162', '#808080',
                                             '#404040'])
        subfigoverlap.draw_text(9.5, 600, 'lin. reg.\ncontrol')

    # When the subfigss are not None, it means the function is called with the real traces df and we will add the numbers to columns 0 and 1.
    else:
        subfigoverlap.draw_vertical_bars([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                                         [num_lumi, num_motion_lumi_overlap, num_motion, num_motion_drive_overlap, num_drive, num_lumi_drive_overlap, num_motion_drive_lumi_overlap,
                                          num_diff, num_bright_diff_overlap, num_bright, num_bright_dark_overlap, num_dark, num_dark_diff_overlap, num_diff_bright_dark_overlap],
                                           vertical_bar_bottom=[0, num_lumi, num_lumi + num_motion_lumi_overlap, num_lumi + num_motion_lumi_overlap + num_motion, num_lumi + num_motion_lumi_overlap + num_motion + num_motion_drive_overlap, num_lumi + num_motion_lumi_overlap + num_motion + num_motion_drive_overlap + num_drive, num_lumi + num_motion_lumi_overlap + num_motion + num_motion_drive_overlap + num_drive + num_lumi_drive_overlap,
                                                                0, num_diff, num_diff + num_bright_diff_overlap, num_diff + num_bright_diff_overlap + num_bright, num_diff + num_bright_diff_overlap + num_bright + num_bright_dark_overlap, num_diff + num_bright_diff_overlap + num_bright + num_bright_dark_overlap + num_dark, num_diff + num_bright_diff_overlap + num_bright + num_bright_dark_overlap + num_dark + num_dark_diff_overlap],
                                           lc=['#E69F00', '#808080', '#359B73', '#808080', '#2271B2', '#808080', '#404040',
                                               '#D55E00', '#808080', '#F748A5', '#808080', '#9F0162', '#808080', '#404040'])
        subfigoverlap.draw_text(0.5, 1200, 'linear\nregression')

    # If subfigss is not None, it means this function was called using the real traces and we will plot the linear regression based traces in Fig S4a.
    if subfigss is not None:
        # Loop over all functional types, combine left and right.
        for dfL, dfR, subfigs, color, fillcolor in zip([motion_left_med_df, lumi_left_med_df, dark_left_med_df, bright_left_med_df, drive_left_med_df, diff_left_med_df],
                                                       [motion_right_med_df, lumi_right_med_df, dark_right_med_df, bright_right_med_df, drive_right_med_df, diff_right_med_df],
                                                       subfigss,
                                                         ['#359B73', '#E69F00',  '#9F0162', '#F748A5', '#2271B2', '#D55E00'],
                                                         ['#8DCDB4', '#F7D280', '#CC7CAD', '#F7A4D0', '#93BADA', '#EEAE7C']):
            # Loop over all stimuli. Flip the rightward stimuli to be able to merge left and right.
            for subfig, stimL, stimR in zip(subfigs, ['lumi_left_dots_left',  'lumi_left_dots_right',  'lumi_left_dots_off',
                                                  'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                                                  'lumi_off_dots_left',   'lumi_off_dots_right',   'lumi_off_dots_off'],
                                            ['lumi_right_dots_right', 'lumi_right_dots_left', 'lumi_right_dots_off',
                                             'lumi_left_dots_right', 'lumi_left_dots_left', 'lumi_left_dots_off',
                                             'lumi_off_dots_right', 'lumi_off_dots_left', 'lumi_off_dots_off']
                                            ):
                # Draw the median traces with the quartile range.
                subfig.draw_line(np.arange(0, 60, 0.5), [np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)],
                                 yerr_neg = np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 25) for i in range(120)]),
                                 yerr_pos = np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 75) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]),
                                 lc=color, lw=1, eafc=fillcolor, eaalpha=1.0, ealw=1, eaec=fillcolor)

            subfigs[6].draw_line([-4, -4], [0, 0.5], lc='k')

        subfigss[4][8].draw_line([40, 60], [-0.34, -0.34], lc='k')
        subfigss[4][8].draw_text(50, -0.6, '20s')

        # Add the model prediction to the traces.
        avg_mot_lumi_change(model_input, subfigss, *model_params, tau_gcamp=24. / 5, kernel_length=int(150 / 5),
                            window_length=int(20 / 5), linregress=True)

    # If subfig_loc is not None, it means this function was called using the real traces and we will plot the linear regression based locations in Fig S4a.
    if subfig_loc is not None:
        # Loop over all functional types.
        for dfL, dfR, color in zip(
                [motion_left_umed_df, lumi_left_umed_df, dark_left_umed_df, bright_left_umed_df, drive_left_umed_df, diff_left_umed_df],
                [motion_right_umed_df, lumi_right_umed_df, dark_right_umed_df, bright_right_umed_df, drive_right_umed_df, diff_right_umed_df],
                ['#359B73', '#E69F00', '#9F0162', '#F748A5', '#2271B2', '#D55E00']):
            # Plot the left neurons as solid circles and the right neurons as open circles.
            subfig_loc.draw_scatter(dfL['ZB_x'].astype(float)*0.798, dfL['ZB_y'].astype(float)*0.798, pc=color, ec=color, ps=0.75, elw=0.25, alpha=0.75)
            subfig_loc.draw_scatter(dfL['ZB_z'].astype(float)*2 + 515, dfL['ZB_y'].astype(float)*0.798, pc=color, ec=color, ps=0.75, elw=0.25, alpha=0.75)
            subfig_loc.draw_scatter(dfR['ZB_x'].astype(float)*0.798, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
            subfig_loc.draw_scatter(dfR['ZB_z'].astype(float)*2 + 515, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
            # Draw red boxes as outlines to later fit the ZBRAIN cartoon outline.
            subfig_loc.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
            subfig_loc.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')

        subfig_loc.draw_line([420, 520], [780, 780], lc='k')
        subfig_loc.draw_text(470, 820, '100\u00b5m')

    return

def sub_plot_control_and_wta_traces(traces_df, traces_control_df, subfigss, subfigss_wta, subfig_loc_ctrl, subfig_loc_wta,
                                    thresh_resp=0.2, thresh_min=0.1, thresh_peaks_diff=1.25, thresh_peaks=1.5, thresh_below=0.9):
    '''
    This function plots the control and wta traces and locations.
    This is related to figure S4c-d.
    :param traces_df: The dataframe containing all functional traces. Each row is a neuron.
    :param traces_control_df: The dataframe containing all functional control traces. Each row is a neuron.
    :param subfigss: List of subfigures which will show the traces of the control data.
    :param subfigss_wta: List of subfigures which will show the traces of the WTA neurons.
    :param subfig_loc_ctrl: Subfigure containing the location of the neurons detected in the control.
    :param subfig_loc_wta: Subfigure containing the location fo the WTA neurons.
    :param thresh_resp: minimum dF/F activity required to be considered part of the functional types.
    :param thresh_min: During non-responses activity cannot go thresh_min above the pre/post stimulus activity.
    :param thresh_peaks_diff: The change detector peak activity of the stronger contrast needs to be thresh_peaks_diff times higher than the peak activity during the weak contrast stimulus.
    :param thresh_peaks: The increase/decrease detector peak activity of the stronger contrast needs to be thresh_peaks_diff times higher than the peak activity during the weak contrast stimulus.
    :param thresh_below: The decrease in activity of the luminance integrators needs to go thresh_below times lower than the pre/post stimulus activity.
    '''

    # Loading the control traces split by functional type.
    motion_left_df = logic_regression_left_motion(traces_control_df, thresh_resp=thresh_resp, thresh_min=thresh_min, shuffle_stim_idx=False)
    drive_left_df = logic_regression_left_drive(traces_control_df, thresh_resp=thresh_resp, shuffle_stim_idx=False)
    lumi_left_df = logic_regression_left_lumi(traces_control_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min, shuffle_stim_idx=False)
    diff_left_df = logic_regression_left_diff(traces_control_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks_diff, shuffle_stim_idx=False)
    bright_left_df = logic_regression_left_bright(traces_control_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
    dark_left_df = logic_regression_left_dark(traces_control_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
    motion_right_df = logic_regression_right_motion(traces_control_df, thresh_resp=thresh_resp, thresh_min=thresh_min, shuffle_stim_idx=False)
    drive_right_df = logic_regression_right_drive(traces_control_df, thresh_resp=thresh_resp, shuffle_stim_idx=False)
    lumi_right_df = logic_regression_right_lumi(traces_control_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min, shuffle_stim_idx=False)
    diff_right_df = logic_regression_right_diff(traces_control_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks_diff, shuffle_stim_idx=False)
    bright_right_df = logic_regression_right_bright(traces_control_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
    dark_right_df = logic_regression_right_dark(traces_control_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)

    # Loop over all functional types, combine left and right, and plot the traces.
    for dfL, dfR, subfigs, color, fillcolor in zip([motion_left_df, lumi_left_df, dark_left_df, bright_left_df, drive_left_df, diff_left_df],
                                                   [motion_right_df, lumi_right_df, dark_right_df, bright_right_df, drive_right_df, diff_right_df],
                                                   subfigss,
                                                     ['#359B73', '#E69F00',  '#9F0162', '#F748A5', '#2271B2', '#D55E00'],
                                                     ['#8DCDB4', '#F7D280', '#CC7CAD', '#F7A4D0', '#93BADA', '#EEAE7C']):
        # Loop over all stimuli. We flip the rightward stimuli to be able to merge left and right.
        for subfig, stimL, stimR in zip(subfigs, ['lumi_left_dots_left',  'lumi_left_dots_right',  'lumi_left_dots_off',
                                              'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                                              'lumi_off_dots_left',   'lumi_off_dots_right',   'lumi_off_dots_off'],
                                        ['lumi_right_dots_right', 'lumi_right_dots_left', 'lumi_right_dots_off',
                                         'lumi_left_dots_right', 'lumi_left_dots_left', 'lumi_left_dots_off',
                                         'lumi_off_dots_right', 'lumi_off_dots_left', 'lumi_off_dots_off']
                                        ):
            # Draw the median traces with the quartile range of all functional control types.
            subfig.draw_line(np.arange(0, 60, 0.5), [np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)],
                             yerr_neg = np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 25) for i in range(120)]),
                             yerr_pos = np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 75) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]),
                             lc=color, lw=1, eafc=fillcolor, eaalpha=1.0, ealw=1, eaec=fillcolor)

        subfigs[6].draw_line([-4, -4], [0, 0.5], lc='k')

    subfigss[4][8].draw_line([40, 60], [-0.34, -0.34], lc='k')
    subfigss[4][8].draw_text(50, -0.6, '20s')

    # Loop over all functional types to plot the location of the neurons.
    for dfL, dfR, color in zip(
            [motion_left_df, lumi_left_df, dark_left_df, bright_left_df, drive_left_df, diff_left_df],
            [motion_right_df, lumi_right_df, dark_right_df, bright_right_df, drive_right_df, diff_right_df],
            ['#359B73', '#E69F00', '#9F0162', '#F748A5', '#2271B2', '#D55E00']):
        # Draw the left neurons as solid circles and the right neurons as open circles.
        subfig_loc_ctrl.draw_scatter(dfL['ZB_x'].astype(float)*0.798, dfL['ZB_y'].astype(float)*0.798, pc=color, ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc_ctrl.draw_scatter(dfL['ZB_z'].astype(float)*2 + 515, dfL['ZB_y'].astype(float)*0.798, pc=color, ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc_ctrl.draw_scatter(dfR['ZB_x'].astype(float)*0.798, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc_ctrl.draw_scatter(dfR['ZB_z'].astype(float)*2 + 515, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)

        # Draw red boxes as outlines to later fit the ZBRAIN cartoon outline.
        subfig_loc_ctrl.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
        subfig_loc_ctrl.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')

    subfig_loc_ctrl.draw_line([420, 520], [780, 780], lc='k')
    subfig_loc_ctrl.draw_text(470, 820, '100\u00b5m')

    # Load the WTA motion and luminance dataframes based on real data.
    motion_left_df_wta = logic_regression_left_motion_wta(traces_df, thresh_resp=thresh_resp, thresh_min=thresh_min, shuffle_stim_idx=False)
    lumi_left_df_wta = logic_regression_left_lumi_wta(traces_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min, shuffle_stim_idx=False)
    motion_right_df_wta = logic_regression_right_motion_wta(traces_df, thresh_resp=thresh_resp, thresh_min=thresh_min, shuffle_stim_idx=False)
    lumi_right_df_wta = logic_regression_right_lumi_wta(traces_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min, shuffle_stim_idx=False)

    print('We found WTA neurons: ')
    print(f'{len(motion_left_df_wta)} Motion left')
    print(f'{len(motion_right_df_wta)} Motion right')
    print(f'{len(lumi_left_df_wta)} Lumi left')
    print(f'{len(lumi_right_df_wta)} Lumi right')

    # Loop over the WTA functional types, combine left and right, and plot the traces.
    for dfL, dfR, subfigs, color, fillcolor in zip([motion_left_df_wta, lumi_left_df_wta,],
                                                   [motion_right_df_wta, lumi_right_df_wta,],
                                                   subfigss_wta,
                                                     ['#359B73', '#E69F00', ],
                                                     ['#8DCDB4', '#F7D280', ]):

        # Loop over all stimuli. We flip the rightward stimuli to be able to merge left and right.
        for subfig, stimL, stimR in zip(subfigs, ['lumi_left_dots_left',  'lumi_left_dots_right',  'lumi_left_dots_off',
                                              'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                                              'lumi_off_dots_left',   'lumi_off_dots_right',   'lumi_off_dots_off'],
                                        ['lumi_right_dots_right', 'lumi_right_dots_left', 'lumi_right_dots_off',
                                         'lumi_left_dots_right', 'lumi_left_dots_left', 'lumi_left_dots_off',
                                         'lumi_off_dots_right', 'lumi_off_dots_left', 'lumi_off_dots_off']
                                        ):
            # Draw the median traces with the quartile range of all functional WTA types.
            subfig.draw_line(np.arange(0, 60, 0.5), [np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)],
                             yerr_neg=np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 25) for i in range(120)]),
                             yerr_pos=np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 75) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]),
                             lc=color, lw=1, eafc=fillcolor, eaalpha=1.0, ealw=1, eaec=fillcolor)

        subfigs[6].draw_line([-4, -4], [0, 0.5], lc='k')

    subfigss[4][8].draw_line([40, 60], [-0.34, -0.34], lc='k')
    subfigss[4][8].draw_text(50, -0.6, '20s')

    # Loop over all functional WTA types to plot the location of the neurons.
    for dfL, dfR, color in zip(
            [motion_left_df_wta, lumi_left_df_wta,],
            [motion_right_df_wta, lumi_right_df_wta,],
            ['#359B73', '#E69F00', '#9F0162', '#F748A5', '#2271B2', '#D55E00']):
        # Draw the left neurons as solid circles and the right neurons as open circles.
        subfig_loc_wta.draw_scatter(dfL['ZB_x'].astype(float)*0.798, dfL['ZB_y'].astype(float)*0.798, pc=color, ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc_wta.draw_scatter(dfL['ZB_z'].astype(float)*2 + 515, dfL['ZB_y'].astype(float)*0.798, pc=color, ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc_wta.draw_scatter(dfR['ZB_x'].astype(float)*0.798, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc_wta.draw_scatter(dfR['ZB_z'].astype(float)*2 + 515, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
        # Draw red boxes as outlines to later fit the ZBRAIN cartoon outline.
        subfig_loc_wta.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
        subfig_loc_wta.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')

    subfig_loc_wta.draw_line([420, 520], [780, 780], lc='k')
    subfig_loc_wta.draw_text(470, 820, '100\u00b5m')

    return

def sub_plot_add_model_prediction_to_traces(subfigss, model_params=[5.73, 2.88, 14.54, 7.61, 0.214, 19.22, 2.88], wta=False):
    '''
    This function adds the model prediction to the traces plots.
    :param subfigss: Subfigure to add the model predictions to.
    :param model_params: List of model parameters. Note for the model based regressors time is sampled at 0.1s instead of 0.5s, so the timeconstants (first 4 model params) need to be divided by 5.
    :param wta: If True the WTA version of the model is used for the predictions. If False the default avg_mot_lumi_change model is used.
    '''

    # Define the length and stimulus types for the model predictions.
    stim_len_timepoints = [600] # 600 means 60s.
    stim_names = ['Motion', 'Photo', 'Same', 'Oppo']

    # Loop over the stimuli to concatenate the model input.
    for i in range(len(stim_len_timepoints) * len(stim_names)):
        folder_id = i % len(stim_len_timepoints)
        stim_id = int(i / len(stim_len_timepoints))

        # Load the model input for this stimulus
        left_mot_input, right_mot_input, left_lumi_input, right_lumi_input = get_stim_input(stim_len_timepoints[folder_id],
                                                                                            stim_names[stim_id])

        time = np.linspace(0, (stim_len_timepoints[folder_id] - 1) / 10, stim_len_timepoints[folder_id])

        # Concatenate the model input to the total model input.
        if i == 0:
            model_input = [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]
        else:
            model_input = np.hstack((model_input, [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]))

        # Add 5s baseline between stimuli to allow the model to return to baseline.
        if i < len(stim_len_timepoints) * len(stim_names) - 1:
            model_input = np.hstack((model_input, [np.zeros(50), 0.1 * np.ones(50), 0.1 * np.ones(50), 0.3 * np.ones(50),
                                      0.3 * np.ones(50)]))

    # Call the requested model function to draw the model prediction.
    if wta:
        wta_mot_lumi_change(model_input, subfigss, *model_params)
    else:
        avg_mot_lumi_change(model_input, subfigss, *model_params)

    return

def sub_plot_control(traces_df, traces_control_df, subfig_s_loc_comb, subfigoverlap,
                     thresh_resp=0.2, thresh_min=0.1, thresh_peaks_diff=1.25, thresh_peaks=1.5, thresh_below=0.9):
    '''
    This function plots the control locations, as well as the number of neurons per type for both the real as the control traces.
    This is related to figure 3g and S4b.
    :param traces_df: The dataframe containing all functional traces. Each row is a neuron.
    :param traces_control_df: The dataframe containing all functional control traces. Each row is a neuron.
    :param subfig_s_loc_comb: The subfigure showing the location of all functional types for the control traces (Fig. 3g).
    :param subfigoverlap: The subfigure showing the number of neurons per type for each analysis strategy (Fig. S4b)
    :param thresh_resp: minimum dF/F activity required to be considered part of the functional types.
    :param thresh_min: During non-responses activity cannot go thresh_min above the pre/post stimulus activity.
    :param thresh_peaks_diff: The change detector peak activity of the stronger contrast needs to be thresh_peaks_diff times higher than the peak activity during the weak contrast stimulus.
    :param thresh_peaks: The increase/decrease detector peak activity of the stronger contrast needs to be thresh_peaks_diff times higher than the peak activity during the weak contrast stimulus.
    :param thresh_below: The decrease in activity of the luminance integrators needs to go thresh_below times lower than the pre/post stimulus activity.
    '''

    # Load the dataframes of each functional type for the real traces.
    print('Loading functional dataframes. ')
    motion_left_df = logic_regression_left_motion(traces_df, thresh_resp=thresh_resp, thresh_min=thresh_min, shuffle_stim_idx=False)
    drive_left_df = logic_regression_left_drive(traces_df, thresh_resp=thresh_resp, shuffle_stim_idx=False)
    lumi_left_df = logic_regression_left_lumi(traces_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min, shuffle_stim_idx=False)
    diff_left_df = logic_regression_left_diff(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks_diff, shuffle_stim_idx=False)
    bright_left_df = logic_regression_left_bright(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
    dark_left_df = logic_regression_left_dark(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
    motion_right_df = logic_regression_right_motion(traces_df, thresh_resp=thresh_resp, thresh_min=thresh_min, shuffle_stim_idx=False)
    drive_right_df = logic_regression_right_drive(traces_df, thresh_resp=thresh_resp, shuffle_stim_idx=False)
    lumi_right_df = logic_regression_right_lumi(traces_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min, shuffle_stim_idx=False)
    diff_right_df = logic_regression_right_diff(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks_diff, shuffle_stim_idx=False)
    bright_right_df = logic_regression_right_bright(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
    dark_right_df = logic_regression_right_dark(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)

    # Compute the number of neurons that overlap between functional types.
    num_motion_drive_overlap = len(np.intersect1d(motion_left_df.index, drive_left_df.index)) + len(np.intersect1d(motion_right_df.index, drive_right_df.index))
    num_lumi_drive_overlap = len(np.intersect1d(lumi_left_df.index, drive_left_df.index)) + len(np.intersect1d(lumi_right_df.index, drive_right_df.index))
    num_motion_lumi_overlap = len(np.intersect1d(lumi_left_df.index, motion_left_df.index)) + len(np.intersect1d(lumi_right_df.index, motion_right_df.index))
    num_motion_drive_lumi_overlap = len(np.intersect1d(np.intersect1d(motion_left_df.index, drive_left_df.index), lumi_left_df.index)) + \
                                    len(np.intersect1d(np.intersect1d(motion_right_df.index, drive_right_df.index), lumi_right_df.index))
    num_bright_diff_overlap = len(np.intersect1d(bright_left_df.index, diff_left_df.index)) + len(np.intersect1d(bright_right_df.index, diff_right_df.index))
    num_dark_diff_overlap = len(np.intersect1d(dark_left_df.index, diff_left_df.index)) + len(np.intersect1d(dark_right_df.index, diff_right_df.index))
    num_bright_dark_overlap = len(np.intersect1d(dark_left_df.index, bright_left_df.index)) + len(np.intersect1d(dark_right_df.index, bright_right_df.index))
    num_diff_bright_dark_overlap = len(np.intersect1d(np.intersect1d(bright_left_df.index, diff_left_df.index), dark_left_df.index)) + \
                                   len(np.intersect1d(np.intersect1d(bright_right_df.index, diff_right_df.index), dark_right_df.index))
    # Compute the number of neurons that are unique for each functional type.
    num_motion = len(motion_left_df) + len(motion_right_df) - num_motion_drive_overlap - num_motion_lumi_overlap - num_motion_drive_lumi_overlap
    num_drive = len(drive_left_df) + len(drive_right_df) - num_motion_drive_overlap - num_lumi_drive_overlap - num_motion_drive_lumi_overlap
    num_lumi = len(lumi_left_df) + len(lumi_right_df) - num_motion_lumi_overlap - num_lumi_drive_overlap - num_motion_drive_lumi_overlap
    num_diff = len(diff_left_df) + len(diff_right_df) - num_bright_diff_overlap - num_dark_diff_overlap - num_diff_bright_dark_overlap
    num_bright = len(bright_left_df) + len(bright_right_df) - num_bright_diff_overlap - num_bright_dark_overlap - num_diff_bright_dark_overlap
    num_dark = len(dark_left_df) + len(dark_right_df) - num_dark_diff_overlap - num_bright_dark_overlap - num_diff_bright_dark_overlap

    print(f'{len(motion_left_df) + len(motion_right_df)} total motion neurons')
    print(f'{len(drive_left_df) + len(drive_right_df)} total drive neurons')
    print(f'{len(lumi_left_df) + len(lumi_right_df)} total lumi neurons')
    print(f'{len(diff_left_df) + len(diff_right_df)} total diff neurons')
    print(f'{len(bright_left_df) + len(bright_right_df)} total bright neurons')
    print(f'{len(dark_left_df) + len(dark_right_df)} total dark neurons')
    print(f'{num_motion_drive_overlap} motion drive overlap')
    print(f'{num_motion_lumi_overlap} motion lumi overlap')
    print(f'{num_lumi_drive_overlap} lumi drive overlap')
    print(f'{num_bright_diff_overlap} bright diff overlap')
    print(f'{num_dark_diff_overlap} dark diff overlap')
    print(f'{num_bright_dark_overlap} bright dark overlap')
    print(f'{num_motion} unique motion neurons')
    print(f'{num_lumi} unique lumi neurons')
    print(f'{num_drive} unique drive neurons')
    print(f'{num_diff} unique diff neurons')
    print(f'{num_bright} unique bright neurons')
    print(f'{num_dark} unique dark neurons')

    # Draw the number of neurons per functional type as a vertical barplot (both unique and overlapping ones) in Fig. S4b. The overlapping sections will be gray bars, the mixed colors are added later in Affinity.
    subfigoverlap.draw_vertical_bars(
        [3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, ],
        [num_lumi, num_motion_lumi_overlap, num_motion, num_motion_drive_overlap, num_drive, num_lumi_drive_overlap, num_motion_drive_lumi_overlap,
         num_diff, num_bright_diff_overlap, num_bright, num_bright_dark_overlap, num_dark, num_dark_diff_overlap, num_diff_bright_dark_overlap],
        vertical_bar_bottom=[0, num_lumi, num_lumi + num_motion_lumi_overlap,
                             num_lumi + num_motion_lumi_overlap + num_motion,
                             num_lumi + num_motion_lumi_overlap + num_motion + num_motion_drive_overlap,
                             num_lumi + num_motion_lumi_overlap + num_motion + num_motion_drive_overlap + num_drive,
                             num_lumi + num_motion_lumi_overlap + num_motion + num_motion_drive_overlap + num_drive + num_lumi_drive_overlap,
                             0, num_diff, num_diff + num_bright_diff_overlap,
                             num_diff + num_bright_diff_overlap + num_bright,
                             num_diff + num_bright_diff_overlap + num_bright + num_bright_dark_overlap,
                             num_diff + num_bright_diff_overlap + num_bright + num_bright_dark_overlap + num_dark,
                             num_diff + num_bright_diff_overlap + num_bright + num_bright_dark_overlap + num_dark + num_dark_diff_overlap],
        lc=['#E69F00', '#808080', '#359B73', '#808080', '#2271B2', '#808080', '#404040', '#D55E00', '#808080',
            '#F748A5', '#808080', '#9F0162', '#808080', '#404040'])
    subfigoverlap.draw_text(3.5, 1000, 'logical\nstatements')

    # Loading the functional dataframes of the control traces.
    print('Loading functional control dataframes. ')
    motion_left_df_s = logic_regression_left_motion(traces_control_df, thresh_resp=thresh_resp, thresh_min=thresh_min, shuffle_stim_idx=False)
    drive_left_df_s = logic_regression_left_drive(traces_control_df, thresh_resp=thresh_resp, shuffle_stim_idx=False)
    lumi_left_df_s = logic_regression_left_lumi(traces_control_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min, shuffle_stim_idx=False)
    diff_left_df_s = logic_regression_left_diff(traces_control_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks_diff, shuffle_stim_idx=False)
    bright_left_df_s = logic_regression_left_bright(traces_control_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
    dark_left_df_s = logic_regression_left_dark(traces_control_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
    motion_right_df_s = logic_regression_right_motion(traces_control_df, thresh_resp=thresh_resp, thresh_min=thresh_min, shuffle_stim_idx=False)
    drive_right_df_s = logic_regression_right_drive(traces_control_df, thresh_resp=thresh_resp, shuffle_stim_idx=False)
    lumi_right_df_s = logic_regression_right_lumi(traces_control_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min, shuffle_stim_idx=False)
    diff_right_df_s = logic_regression_right_diff(traces_control_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks_diff, shuffle_stim_idx=False)
    bright_right_df_s = logic_regression_right_bright(traces_control_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
    dark_right_df_s = logic_regression_right_dark(traces_control_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)

    # Loop over all control dataframes and plot the location of the neurons in Fig. 3g.
    for dfL, dfR, color, intens_color in zip(
            [motion_left_df_s, lumi_left_df_s, dark_left_df_s, bright_left_df_s, drive_left_df_s, diff_left_df_s],
            [motion_right_df_s, lumi_right_df_s, dark_right_df_s, bright_right_df_s, drive_right_df_s, diff_right_df_s],
            ['#8DCDB4', '#F7D280', '#CC7CAD', '#F7A4D0', '#93BADA', '#EEAE7C'],
            ['#359B73', '#E69F00', '#9F0162', '#F748A5', '#2271B2', '#D55E00']):

        subfig_s_loc_comb.draw_scatter(dfL['ZB_x'].astype(float)*0.798, dfL['ZB_y'].astype(float)*0.798, pc=intens_color, ec=intens_color, elw=0.25, ps=0.5, alpha=0.75)
        subfig_s_loc_comb.draw_scatter(dfL['ZB_z'].astype(float)*2 + 515, dfL['ZB_y'].astype(float)*0.798, pc=intens_color, ec=intens_color, elw=0.25, ps=0.5, alpha=0.75)
        subfig_s_loc_comb.draw_scatter(dfR['ZB_x'].astype(float)*0.798, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=intens_color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_s_loc_comb.draw_scatter(dfR['ZB_z'].astype(float)*2 + 515, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=intens_color, ps=0.75, elw=0.25, alpha=0.75)

    subfig_s_loc_comb.draw_line([420, 520], [780, 780], lc='k')
    subfig_s_loc_comb.draw_text(470, 820, '100\u00b5m')
    subfig_s_loc_comb.draw_text(387.5, 50, 'no stim control')
    # Draw red boxes as outlines to later fit the ZBRAIN cartoon outline.
    subfig_s_loc_comb.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
    subfig_s_loc_comb.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')

    # Compute the number of neurons that overlap between functional types of the control dataframes.
    num_motion_drive_overlap = len(np.intersect1d(motion_left_df_s.index, drive_left_df_s.index)) + len(np.intersect1d(motion_right_df_s.index, drive_right_df_s.index))
    num_lumi_drive_overlap = len(np.intersect1d(lumi_left_df_s.index, drive_left_df_s.index)) + len(np.intersect1d(lumi_right_df_s.index, drive_right_df_s.index))
    num_motion_lumi_overlap = len(np.intersect1d(lumi_left_df_s.index, motion_left_df_s.index)) + len(np.intersect1d(lumi_right_df_s.index, motion_right_df_s.index))
    num_motion_drive_lumi_overlap = len(np.intersect1d(np.intersect1d(motion_left_df_s.index, drive_left_df_s.index), lumi_left_df_s.index)) + \
                                    len(np.intersect1d(np.intersect1d(motion_right_df_s.index, drive_right_df_s.index), lumi_right_df_s.index))
    num_bright_diff_overlap = len(np.intersect1d(bright_left_df_s.index, diff_left_df_s.index)) + len(np.intersect1d(bright_right_df_s.index, diff_right_df_s.index))
    num_dark_diff_overlap = len(np.intersect1d(dark_left_df_s.index, diff_left_df_s.index)) + len(np.intersect1d(dark_right_df_s.index, diff_right_df_s.index))
    num_bright_dark_overlap = len(np.intersect1d(dark_left_df_s.index, bright_left_df_s.index)) + len(np.intersect1d(dark_right_df_s.index, bright_right_df_s.index))
    num_diff_bright_dark_overlap = len(np.intersect1d(np.intersect1d(bright_left_df_s.index, diff_left_df_s.index), dark_left_df_s.index)) + \
                                   len(np.intersect1d(np.intersect1d(bright_right_df_s.index, diff_right_df_s.index), dark_right_df_s.index))

    # Compute the number of neurons that are unique for each functional type in the control dataframes.
    num_motion = len(motion_left_df_s) + len(motion_right_df_s) - num_motion_drive_overlap - num_motion_lumi_overlap - num_motion_drive_lumi_overlap
    num_drive = len(drive_left_df_s) + len(drive_right_df_s) - num_motion_drive_overlap - num_lumi_drive_overlap - num_motion_drive_lumi_overlap
    num_lumi = len(lumi_left_df_s) + len(lumi_right_df_s) - num_motion_lumi_overlap - num_lumi_drive_overlap - num_motion_drive_lumi_overlap
    num_diff = len(diff_left_df_s) + len(diff_right_df_s) - num_bright_diff_overlap - num_dark_diff_overlap - num_diff_bright_dark_overlap
    num_bright = len(bright_left_df_s) + len(bright_right_df_s) - num_bright_diff_overlap - num_bright_dark_overlap - num_diff_bright_dark_overlap
    num_dark = len(dark_left_df_s) + len(dark_right_df_s) - num_dark_diff_overlap - num_bright_dark_overlap - num_diff_bright_dark_overlap

    # Draw the number of neurons per functional type for the control dataframe as a vertical barplot (both unique and overlapping ones) in Fig. S4b. The overlapping sections will be gray bars, the mixed colors are added later in Affinity.
    subfigoverlap.draw_vertical_bars([6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, ],
        [num_lumi, num_motion_lumi_overlap, num_motion, num_motion_drive_overlap, num_drive, num_lumi_drive_overlap, num_motion_drive_lumi_overlap,
         num_diff, num_bright_diff_overlap, num_bright, num_bright_dark_overlap, num_dark, num_dark_diff_overlap, num_diff_bright_dark_overlap],
        vertical_bar_bottom=[0, num_lumi, num_lumi + num_motion_lumi_overlap,
                             num_lumi + num_motion_lumi_overlap + num_motion,
                             num_lumi + num_motion_lumi_overlap + num_motion + num_motion_drive_overlap,
                             num_lumi + num_motion_lumi_overlap + num_motion + num_motion_drive_overlap + num_drive,
                             num_lumi + num_motion_lumi_overlap + num_motion + num_motion_drive_overlap + num_drive + num_lumi_drive_overlap,
                             0, num_diff, num_diff + num_bright_diff_overlap,
                             num_diff + num_bright_diff_overlap + num_bright,
                             num_diff + num_bright_diff_overlap + num_bright + num_bright_dark_overlap,
                             num_diff + num_bright_diff_overlap + num_bright + num_bright_dark_overlap + num_dark,
                             num_diff + num_bright_diff_overlap + num_bright + num_bright_dark_overlap + num_dark + num_dark_diff_overlap],
        lc=['#E69F00', '#808080', '#359B73', '#808080', '#2271B2', '#808080', '#404040', '#D55E00', '#808080',
            '#F748A5', '#808080', '#9F0162', '#808080', '#404040'])
    subfigoverlap.draw_text(6.5, 800, 'no stim\ncontrol')
    return

def sub_plot_example_traces(path_to_trace_data, path_to_volume, neurons_to_plot, subfiga, subfigb, subfigc):
    '''
    This function plots the example traces and example imaging stack.
    This is related to figure 3c.
    :param path_to_trace_data: Path to the preprocessed dataframe containing the example functional data.
    :param path_to_volume: Path to the preprocessed dataframe containing the example imaging stack.
    :param neurons_to_plot: List of neuron IDs to plot the functional traces and location of.
    :param subfiga: Subfigure showing the example imaging stack.
    :param subfigb: Subfigure showing the example functional plane and segmented cell outlines.
    :param subfigc: Subfigure showing the example traces of the neurons specified in neurons_to_plot.
    '''
    # Load the example stack ('average_stack_green_channel').
    preproc_hdf5 = h5py.File(path_to_volume, "r")
    avg_im = np.array(preproc_hdf5['average_stack_green_channel'])
    # Loop over 5 planes and plot them with an x and y offset.
    for z, offset in zip(range(5), [0, 50, 250, 300, 350]):
        # For visualization (to avoid a very dark image) we clip the lowest and highest 5 percentiles.
        subfiga.draw_image(np.clip(avg_im[4-z, :, :], np.nanpercentile(avg_im[4-z, :, :], 5), np.nanpercentile(avg_im[4-z, :, :], 95)), colormap='gray',
                           extent=(offset, 800+offset, 800+offset, offset), image_origin='upper')
        subfiga.draw_line([offset, 800+offset, 800+offset, offset, offset], [offset, offset, 800+offset, 800+offset, offset], lc='w', lw=0.5)
    # Add three dots on each side to highlight its an imaging stack.
    subfiga.draw_scatter([100, 150, 200, 900, 950, 1000], [900, 950, 1000, 100, 150, 200], ec='k', pc='k', ps=1)
    preproc_hdf5.close()

    # Load the functional data.
    preproc_hdf5 = h5py.File(path_to_trace_data, "r")
    # Draw the average image.
    avg_im = np.array(preproc_hdf5['average_stack_green_channel']).reshape(799, 799)
    subfigb.draw_image(np.clip(avg_im, np.nanpercentile(avg_im, 5), np.nanpercentile(avg_im, 95)), colormap='gray', extent=(0, 800, 800, 0), image_origin='upper')
    # Loop over all neurons are plot the cellpose segmentation outline.
    for i in range(2103):
        unit_contour = np.array(preproc_hdf5['z_plane0000']['cellpose_segmentation']['unit_contours'][f'{10000+i}'])
        if i in neurons_to_plot:
            continue
        else:
            subfigb.draw_line(unit_contour[:, 0], unit_contour[:, 1], lc='tab:blue', lw=0.2)
    # Redraw the outline of the selected neurons in a different color.
    for i in neurons_to_plot:
        unit_contour = np.array(preproc_hdf5['z_plane0000']['cellpose_segmentation']['unit_contours'][f'{10000+i}'])
        subfigb.draw_line(unit_contour[:, 0], unit_contour[:, 1], lc='#00FAFF', lw=0.4)

    # Add numbers to indicate which neuron relates to which example traces.
    subfigb.draw_line([600, 732], [770, 770], lc='w', lw=1)
    subfigb.draw_text(666, 720, '50 \u00b5m', textcolor='w')
    subfigb.draw_text(160, 270, '1', textcolor='#00FAFF')
    subfigb.draw_text(330, 170, '2', textcolor='#00FAFF')
    subfigb.draw_text(420, 150, '3', textcolor='#00FAFF')
    subfigb.draw_text(440, 240, '4', textcolor='#00FAFF')
    subfigb.draw_text(270, 400, '5', textcolor='#00FAFF')
    subfigb.draw_text(330, 470, '6', textcolor='#00FAFF')
    subfigb.draw_text(400, 400, '7', textcolor='#00FAFF')
    subfigb.draw_text(490, 380, '8', textcolor='#00FAFF')
    subfigb.draw_text(490, 480, '9', textcolor='#00FAFF')
    subfigb.draw_text(400, -50, '~1000 cells per plane')

    # Load the example traces and stimulus information (to know which stimulus was shown when).
    traces = np.array(preproc_hdf5['z_plane0000']['cellpose_segmentation']['F'])[:, :-50]  # Skipping the last 50 frames because the last stimulus was incomplete.
    stim_starts = np.array(preproc_hdf5['z_plane0000']['stimulus_information'][:, 0])
    stim_ends = np.array(preproc_hdf5['z_plane0000']['stimulus_information'][:, 1])
    stim_types = np.array(preproc_hdf5['z_plane0000']['stimulus_information'][:, 2])
    im_times = np.array(preproc_hdf5['z_plane0000']['imaging_information'][:, 0])[:-50]
    preproc_hdf5.close()

    # Loop over the example neurons and plot the normalized traces
    for i, n in enumerate(neurons_to_plot):
        norm_trace = (traces[n, :] - np.max(traces[n, :])) / (np.max(traces[n, :]) - np.min(traces[n, :]))
        subfigc.draw_line(im_times, norm_trace + i * 1.1)
    # Loop over the trials and add a letter to indicate which stimulus was shown.
    for stim_start, stim_end, stim_type in zip(stim_starts, stim_ends, stim_types):
        if stim_type == 0:
            st = 'a'
        elif stim_type == 1:
            st = 'b'
        elif stim_type == 2:
            st = 'c'
        elif stim_type == 3:
            st = 'd'
        elif stim_type == 4:
            st = 'e'
        elif stim_type == 5:
            st = 'f'
        elif stim_type == 6:
            st = 'g'
        elif stim_type == 7:
            st = 'h'
        elif stim_type == 8:
            st = 'i'
        if stim_type in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            subfigc.draw_text(stim_start + 25, 9.5, f'{st}')
    subfigc.draw_line([2320, 2320], [-0.9, 0.1], lc='k')
    subfigc.draw_text(2200, -0.9, 'normalized\nfluorescence', textlabel_rotation=90, textlabel_va='bottom')
    subfigc.draw_line([4840, 5140], [-1.5, -1.5], lc='k')
    subfigc.draw_text(4990, -2, '5 min')
    return

def sub_plot_total_cells(traces_df, subfig):
    '''
    This function plots the total imaged cells as a heatmap.
    This is related to fig 3d
    :param traces_df: Dataframe containing all functional traces. Each row contains a neuron.
    :param subfig: Subfigure to plot the heatmap of all neurons.
    '''

    # Get the x, y, z location of all neurons.
    x = traces_df['ZB_x'][traces_df['ZB_z'] > 0].astype(float)
    y = traces_df['ZB_y'][traces_df['ZB_z'] > 0].astype(float)
    z = traces_df['ZB_z'][traces_df['ZB_z'] > 0].astype(float)

    # Create the xy and zy heatmaps
    heatmapxy, xedges, yedges = np.histogram2d(x * 0.798, y * 0.798, bins=[np.arange(35, 475, 5), np.arange(85, 845, 5)])
    heatmapzy, zedges, yedges = np.histogram2d(z * 2, y * 0.798, bins=[np.arange(35, 285, 5), np.arange(85, 845, 5)])
    extentxy = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
    extentzy = [zedges[0]+515, zedges[-1]+515, yedges[-1], yedges[0]]

    # Create the colormaps and make sure they are comparable between the xy and yz plot.
    cmapxy = cm.get_cmap('Blues', heatmapxy.max())
    newcolorsxy = cmapxy(np.linspace(0.25, 1, int(heatmapxy.max())))
    white = np.array([1, 1, 1, 1])
    newcolorsxy[0, :] = white
    bluewhite_cmapxy = ListedColormap(newcolorsxy)
    cmapzy = cm.get_cmap('Blues', heatmapzy.max())
    newcolorszy = cmapzy(np.linspace(0.25, 1, int(heatmapzy.max())))
    white = np.array([1, 1, 1, 1])
    newcolorszy[0, :] = white
    bluewhite_cmapzy = ListedColormap(newcolorszy)

    # Draw the heatmaps
    subfig.draw_image(heatmapxy.T, colormap=bluewhite_cmapxy, extent=extentxy, image_origin='upper')
    subfig.draw_image(heatmapzy.T, colormap=bluewhite_cmapzy, extent=extentzy, image_origin='upper')
    print(np.array(heatmapxy.T).max())
    print(np.array(heatmapxy.T).min())
    print(np.array(heatmapzy.T).max())
    print(np.array(heatmapzy.T).min())
    subfig.draw_line([420, 520], [780, 780], lc='k')
    subfig.draw_text(470, 820, '100\u00b5m')
    subfig.draw_text(560, 900, '~140000 cells total')
    # Draw red boxes as outlines to later fit the ZBRAIN cartoon outline.
    subfig.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
    subfig.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')

    return


def exponential_func(t, a, tau, b):
    '''
    This function contains the exponential function fitted to the luminance integration activity.
    :param t: Time array.
    :param a: Scaling factor.
    :param tau: Timeconstant.
    :param b: Offset bias
    :return: Array with outcome of the exponential function.
    '''
    return a * (1-np.exp(-t/tau)) + b

def subplot_lumi_integrator_check(traces_df, subfigs_traces, loc_plot, tau_scatter,
                                  thresh_resp=0.2, thresh_min=0.1, thresh_below=0.9):
    '''
    This function plots the functional activity and location of tectal luminance integrator neurons for 3 different contrast levels. It also shows the fitted timeconstant per fish.
    This is related to Figure S3b-d.
    :param traces_df: Dataframe with all functional traces from the luminance integrator experiment. Each row contains one neuron.
    :param subfigs_traces: Subfigure to show the 3 functional traces for strong, medium and weak contrast.
    :param loc_plot: Subfigure to show the location of all tectal luminance integrators.
    :param tau_scatter: Subfigure to show the fitted time-constant across contrast levels per fish.
    :param thresh_resp: minimum dF/F activity required to be considered part of the functional types.
    :param thresh_min: During non-responses activity cannot go thresh_min above the pre/post stimulus activity.
    :param thresh_below: The decrease in activity of the luminance integrators needs to go thresh_below times lower than the pre/post stimulus activity.
    '''
    # We select all luminance integrator neurons among the tectal neurons.
    lumi_left_df = logic_regression_left_lumi_single(traces_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min)
    lumi_right_df = logic_regression_right_lumi_single(traces_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min)

    # We plot the functional activity and location of the tectal luminance integrator neurons.
    dfL = lumi_left_df
    dfR = lumi_right_df
    subfigs = subfigs_traces[0]
    color = '#E69F00'
    fillcolor = '#F7D280'

    # The left neurons are plotted as solid circles, the right neurons are plotted as open circles. Here split with a seperate figure for each functional type.
    loc_plot.draw_scatter(dfL['ZB_x'].astype(float) * 0.798, dfL['ZB_y'].astype(float) * 0.798, pc=color, ec=color, ps=0.75, elw=0.25, alpha=0.75)
    loc_plot.draw_scatter(dfL['ZB_z'].astype(float) * 2 + 515, dfL['ZB_y'].astype(float) * 0.798, pc=color, ec=color, ps=0.75, elw=0.25, alpha=0.75)
    loc_plot.draw_scatter(dfR['ZB_x'].astype(float) * 0.798, dfR['ZB_y'].astype(float) * 0.798, pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
    loc_plot.draw_scatter(dfR['ZB_z'].astype(float) * 2 + 515, dfR['ZB_y'].astype(float) * 0.798, pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)

    for subfig, stimL, stimR in zip(subfigs, ['lumi_left_strong_dots_off',  'lumi_left_medium_dots_off',  'lumi_left_weak_dots_off',
                                              'lumi_off_dots_left', 'lumi_off_dots_off'],
                                    ['lumi_right_strong_dots_off', 'lumi_right_medium_dots_off', 'lumi_right_weak_dots_off',
                                     'lumi_off_dots_right', 'lumi_off_dots_off']
                                    ):

        # Get the functional activity and the exponential fit.
        median_line = np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)])
        popt, pcov = curve_fit(exponential_func, np.arange(10, 40, 0.5), median_line[20:80], p0=[1.0, 1.0, 0.0], bounds=[(0, 0, -10), (100, 60, 10)])
        a_fit, tau_fit, b_fit = popt
        print(a_fit, tau_fit, b_fit)

        # Plot the activity (median and quartile range) and the exponential fit.
        subfig.draw_line(np.arange(0, 60, 0.5), [np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)],
                         yerr_neg=np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 25) for i in range(120)]),
                         yerr_pos=np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 75) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]),
                         lc=color, lw=1, eafc=fillcolor, eaalpha=1.0, ealw=1, eaec=fillcolor)
        subfig.draw_line(np.arange(10, 40, 0.5), exponential_func(np.arange(10, 40, 0.5), *popt), lc='k', lw=0.3)

    # Add the scale bars to the functional activity
    subfigs[0].draw_line([-4, -4], [0, 0.5], lc='k')
    subfigs[4].draw_line([40, 60], [-0.34, -0.34], lc='k')
    subfigs[4].draw_text(50, -0.6, '20s')

    # Draw red boxes as outlines to later fit the ZBRAIN cartoon outline.
    loc_plot.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
    loc_plot.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')
    # Draw the scale bar.
    loc_plot.draw_line([420, 520], [780, 780], lc='k')
    loc_plot.draw_text(470, 820, '100\u00b5m')

    # Loop over each fit and get the timeconstants for the strong, medium and weak contrast levels.
    taus_per_fish = np.zeros((len(traces_df['datetime'].unique()), 3))
    for fish_id, date_time in enumerate(traces_df['datetime'].unique()):
        print(f'Fish {date_time}')
        # Select the data for a single fish.
        dfL = lumi_left_df[traces_df['datetime'] == date_time]
        dfR = lumi_right_df[traces_df['datetime'] == date_time]
        print(f'N numbers: {len(dfL)} {len(dfR)}')
        # Loop over the three stimuli.
        for stimL, stimR, stim_x_idx in zip(['lumi_left_strong_dots_off', 'lumi_left_medium_dots_off',
                                                     'lumi_left_weak_dots_off'],
                                                    ['lumi_right_strong_dots_off', 'lumi_right_medium_dots_off',
                                                     'lumi_right_weak_dots_off'],
                                                    [0, 1, 2, ]
                                                    ):
            # Get the median response of the single fish.
            median_line = np.array([np.nanpercentile(
                np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)),
                50) for i in range(120)])
            # Fit the time constant.
            popt, pcov = curve_fit(exponential_func, np.arange(10, 40, 0.5), median_line[20:80], p0=[1.0, 1.0, 0.0], bounds=[(0, 0, -10), (100, 60, 10)])
            a_fit, tau_fit, b_fit = popt
            print(a_fit, tau_fit, b_fit)
            # Store the data and draw the scatter point.
            taus_per_fish[fish_id, stim_x_idx] = tau_fit
            tau_scatter.draw_scatter([stim_x_idx, ], [tau_fit], pc='#E69F00', ec=None)

        tau_scatter.draw_line([0, 1, 2, ], taus_per_fish[fish_id, :], lc='gray', lw=0.3)

    # We check whether there is a signficant increase in the timeconstant of medium vs strong contrasts. The pval is Bonferonni corrected for 2 tests.
    tau_scatter.draw_line([0, 0, 1, 1], [4.7, 4.8, 4.8, 4.7], lc='k')
    _, pval = ttest_rel(taus_per_fish[:, 1], taus_per_fish[:, 0], alternative='greater')
    if pval < 0.001/2:
        tau_scatter.draw_text(0.5, 5, '***')
    elif pval < 0.01/2:
        tau_scatter.draw_text(0.5, 5, '**')
    elif pval < 0.05/2:
        tau_scatter.draw_text(0.5, 5, '*')
    else:
        tau_scatter.draw_text(0.5, 5, 'ns')
    effect_size = cohens_d(taus_per_fish[:, 1], taus_per_fish[:, 0])
    print(f'Strong lumi contrast vs medium lumi contrast, pval {pval}: Cohen D effect size {effect_size}')

    # We check whether there is a signficant increase in the timeconstant of weak vs medium contrasts. The pval is Bonferonni corrected for 2 tests.
    tau_scatter.draw_line([1, 1, 2, 2], [5.2, 5.3, 5.3, 5.2], lc='k')
    _, pval = ttest_rel(taus_per_fish[:, 2], taus_per_fish[:, 1], alternative='greater')
    if pval < 0.001/2:
        tau_scatter.draw_text(1.5, 5.5, '***')
    elif pval < 0.01/2:
        tau_scatter.draw_text(1.5, 5.5, '**')
    elif pval < 0.05/2:
        tau_scatter.draw_text(1.5, 5.5, '*')
    else:
        tau_scatter.draw_text(1.5, 5.5, 'ns')
    effect_size = cohens_d(taus_per_fish[:, 2], taus_per_fish[:, 1])
    print(f'Medium lumi contrast vs Weak lumi contrast, pval {pval}: Cohen D effect size {effect_size}')


    return

def sub_plot_brain_region_overview(regions, regions_short_names, regions_obj_path, example_brain_xy_overview_plots, example_brain_yz_overview_plots):
    '''
    This function plots the brain cartoons with each analysed region highlighted.
    This is related to Figure S3a.
    :param regions: List of mapzebrain region names to plot.
    :param regions_short_names: List of the short names of the same regions in regions. These short names are used as labels.
    :param regions_obj_path: Path to the folder containing all region obj files.
    :param example_brain_xy_overview_plots: List of Subfigures to plot the xy-view of each region in regions.
    :param example_brain_yz_overview_plots: List of Subfigures to plot the yz-view of each region in regions.
    '''
    total_brain_regions = [
        navis.read_mesh(fr'{regions_obj_path}\prosencephalon_(forebrain).obj', units='microns', output='volume'),
        navis.read_mesh(fr'{regions_obj_path}\mesencephalon_(midbrain).obj', units='microns', output='volume'),
        navis.read_mesh(fr'{regions_obj_path}\rhombencephalon_(hindbrain).obj', units='microns', output='volume')]
    for r in range(len(regions)):
        print(regions[r])
        brain_regions = [navis.read_mesh(fr'{regions_obj_path}\{regions[r]}.obj', units='microns', output='volume'), ]
        # Plot the full brain with highlighted reference brain regions as guide.
        example_brain_xy_overview_plots[r].draw_navis_neuron(None, total_brain_regions, navis_view=('x', '-y'), lw=0.5, rasterized=True)
        example_brain_xy_overview_plots[r].draw_navis_neuron(None, brain_regions, navis_color='gray', navis_view=('x', '-y'), lw=0.5, rasterized=True)
        example_brain_yz_overview_plots[r].draw_navis_neuron(None, total_brain_regions, navis_view=('z', '-y'), lw=0.5, rasterized=True)
        example_brain_yz_overview_plots[r].draw_navis_neuron(None, brain_regions, navis_color='gray', navis_view=('z', '-y'), lw=0.5, rasterized=True)
        example_brain_xy_overview_plots[r].draw_text(0, 0, regions_short_names[r+1], textlabel_ha='left')
    return

def sub_plot_n_neurons_per_region(traces_df, perc_neurons_plot_bottom, perc_neurons_plot_top, regions, regions_short_names,
                                  regions_path, thresh_resp=0.2, thresh_min=0.1, thresh_peaks_diff=1.25, thresh_peaks=1.5, thresh_below=0.9):
    '''
    This figure plots the total number of functional type neurons. And the percentage of functional type neurons per region per fish.
    This is related to Figure 3h and S3a
    :param traces_df: Dataframe with all functional traces. Each row contains one neuron.
    :param perc_neurons_plot_bottom:  subfigure with the percentage of functional type neurons per region per fish. Up from 2 % (the y-axis is split to accommodate for high percentages in several regions).
    :param perc_neurons_plot_top: subfigure with the percentage of functional type neurons per region per fish. Up to 2 % (the y-axis is split to accommodate for high percentages in several regions).
    :param regions: List of mapzebrain region names to plot.
    :param regions_short_names: List of the short names of the same regions in regions. These short names are used as labels.
    :param regions_path: Path to the dataframe containing all mapzebrain region.
    :param thresh_resp: minimum dF/F activity required to be considered part of the functional types.
    :param thresh_min: During non-responses activity cannot go thresh_min above the pre/post stimulus activity.
    :param thresh_peaks_diff: The change detector peak activity of the stronger contrast needs to be thresh_peaks_diff times higher than the peak activity during the weak contrast stimulus.
    :param thresh_peaks: The increase/decrease detector peak activity of the stronger contrast needs to be thresh_peaks_diff times higher than the peak activity during the weak contrast stimulus.
    :param thresh_below: The decrease in activity of the luminance integrators needs to go thresh_below times lower than the pre/post stimulus activity.
    '''

    # Load the regions
    region_masks = create_combined_region_npy_mask(regions_path, regions=regions)

    # Loop over all fish to find the percentage of functional type neurons per region per fish.
    avg_over_fish = np.nan * np.ones((len(np.unique(traces_df['fish_idx'])), len(regions), 6))
    for f_idx, fish in enumerate(np.unique(traces_df['fish_idx'])):
        # Select the data of a single fish.
        fish_df = traces_df[traces_df['fish_idx'] == fish]
        # Get the total number of regions per region.
        mask_fish = np.zeros((621, 1406, 138))
        mask_fish[fish_df['ZB_x'].astype(int), fish_df['ZB_y'].astype(int), fish_df['ZB_z'].astype(int)] = 1
        neurons_per_region = np.histogram(region_masks[mask_fish.astype(bool)], bins=np.arange(0, len(regions)+2))[0]
        # Select only the regions which have at least 100 neurons
        good_regions = np.where(neurons_per_region > 100)[0]
        print(f'Good regions: {good_regions}')

        # Split the fish_df by functional type.
        print('Getting neurons')
        motion_left_df = logic_regression_left_motion(fish_df, thresh_resp=thresh_resp, thresh_min=thresh_min, shuffle_stim_idx=False)
        drive_left_df = logic_regression_left_drive(fish_df, thresh_resp=thresh_resp, shuffle_stim_idx=False)
        lumi_left_df = logic_regression_left_lumi(fish_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min, shuffle_stim_idx=False)
        diff_left_df = logic_regression_left_diff(fish_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks_diff, shuffle_stim_idx=False)
        bright_left_df = logic_regression_left_bright(fish_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
        dark_left_df = logic_regression_left_dark(fish_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
        motion_right_df = logic_regression_right_motion(fish_df, thresh_resp=thresh_resp, thresh_min=thresh_min, shuffle_stim_idx=False)
        drive_right_df = logic_regression_right_drive(fish_df, thresh_resp=thresh_resp, shuffle_stim_idx=False)
        lumi_right_df = logic_regression_right_lumi(fish_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min, shuffle_stim_idx=False)
        diff_right_df = logic_regression_right_diff(fish_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks_diff, shuffle_stim_idx=False)
        bright_right_df = logic_regression_right_bright(fish_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
        dark_right_df = logic_regression_right_dark(fish_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
        print('Found all neurons. ')

        # Loop over all functional types and combine the left and right version.
        for count, (left_df, right_df, type_color) in enumerate(zip([motion_left_df, drive_left_df, lumi_left_df, diff_left_df, bright_left_df, dark_left_df],
                                                                    [motion_right_df, drive_right_df, lumi_right_df, diff_right_df, bright_right_df, dark_right_df],
                                                                    ['#359B73', '#2271B2', '#E69F00', '#D55E00', '#F748A5', '#9F0162',])):
            # Label by the neurons by region based on their location.
            mask_left = np.zeros((621, 1406, 138))
            mask_left[left_df['ZB_x'].astype(int), left_df['ZB_y'].astype(int), left_df['ZB_z'].astype(int)] = 1
            mask_right = np.zeros((621, 1406, 138))
            mask_right[right_df['ZB_x'].astype(int), right_df['ZB_y'].astype(int), right_df['ZB_z'].astype(int)] = 1
            overlap = np.append(region_masks[mask_left.astype(bool)], region_masks[mask_right.astype(bool)])

            # Loop over all regions and plot the percentage of functional type neurons if the region is good (=has at least 100 total neurons).
            for r in range(len(regions_short_names)):
                if r == 0 or r not in good_regions:
                    continue
                percentage_neurons = np.sum(overlap == r) / neurons_per_region[r] * 100
                avg_over_fish[f_idx, r-1, count] = percentage_neurons
                # The y-axis is split at 2%, check if the scatter point should go in the top or bottom plot.
                if percentage_neurons > 2:
                    perc_neurons_plot_top.draw_scatter(r + len(regions_short_names) * count, percentage_neurons, pc=type_color, ec=None)
                else:
                    perc_neurons_plot_bottom.draw_scatter(r + len(regions_short_names) * count, percentage_neurons, pc=type_color, ec=None)

    # Draw the black median lines across all regions.
    for count in range(6):
        perc_neurons_plot_bottom.draw_scatter(np.arange(1, len(regions)+1) + count * len(regions_short_names), np.nanmedian(avg_over_fish[:, :, count], axis=0), pt='_', pc='k')
    return

def create_model_based_traces(tau_gcamp=4.8, kernel_length=30, noise_level=1.0):
    '''
    This function creates synthetic data for four responsive neurons. Each neuron has a different model-based response: Left lumi change, Right lumi change, Left lumi, and Right lumi.
    :param tau_gcamp: Timeconstant used for the convolution with a GCaMP kernel.
    :param kernel_length: Length of the GCaMP kernel.
    :param noise_level: Noise-level used as std in the added gaussian noise.
    :return: Average activity trace across 8 trials of a single synthetic neuron.
    '''
    # Initialize the stimulus parameters used to get the model predicted activity traces.
    stim_len_timepoints = [120]
    stim_names = ['Same_L', 'Oppo_R', 'Photo_L', 'Oppo_L', 'Same_R', 'Photo_R', 'Motion_L', 'Motion_R', 'No_Stim']
    # Loop over the stimuli to combine the total inputs used to get the model predicted activity traces.
    for i in range(len(stim_len_timepoints) * len(stim_names)):
        folder_id = i % len(stim_len_timepoints)
        stim_id = int(i / len(stim_len_timepoints))

        left_mot_input, right_mot_input, left_lumi_input, right_lumi_input = get_stim_input_regression(
            stim_len_timepoints[folder_id],
            stim_names[stim_id])

        time = np.linspace(0, (stim_len_timepoints[folder_id] - 1) / 2, stim_len_timepoints[folder_id])

        if i == 0:
            model_input = [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]
        else:
            model_input = np.hstack(
                (model_input, [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]))

    model_params = [5.678052396390699 / 5, 3.765203515714735 / 5, 16.105457978010474 / 5, 7.487431450829603 / 5,
                    0.2136764584807561, 2.0003470409289816, 2.850268651613628]

    # Get the activity of the relevant model nodes.
    _, input_left_lumi, _, _, _, input_left_diff, _, input_right_lumi, _, _, _, input_right_diff = avg_mot_lumi_change(
        model_input, None, *model_params, tau_gcamp=24. / 5, kernel_length=int(150 / 5), window_length=int(20 / 5))

    # Normalize the model activity
    input_left_lumi = (input_left_lumi - np.nanmin(input_left_lumi)) / (
                np.nanmax(input_left_lumi) - np.nanmin(input_left_lumi))
    input_left_diff = (input_left_diff - np.nanmin(input_left_diff)) / (
            np.nanmax(input_left_diff) - np.nanmin(input_left_diff))
    input_right_lumi = (input_right_lumi - np.nanmin(input_right_lumi)) / (
                np.nanmax(input_right_lumi) - np.nanmin(input_right_lumi))
    input_right_diff = (input_right_diff - np.nanmin(input_right_diff)) / (
            np.nanmax(input_right_diff) - np.nanmin(input_right_diff))

    # Initialize the arrays that will contain the activity traces of 8 trials and (9 stimuli x 120 timepoints) = 1080 timepoints.
    gcamp_input_left_lumi = np.zeros((8, 1080))
    gcamp_input_right_lumi = np.zeros((8, 1080))
    gcamp_input_left_diff = np.zeros((8, 1080))
    gcamp_input_right_diff = np.zeros((8, 1080))
    # Loop over all trials
    for trial in range(8):
        # Add some random (Poisson based) additional spikes to the synthetic data
        extra_fires = np.random.poisson()
        for extra_fire in range(extra_fires):
            input_left_lumi[np.random.randint(0, 120)] = np.random.uniform(0, 1)
        extra_fires = np.random.poisson()
        for extra_fire in range(extra_fires):
            input_right_lumi[np.random.randint(0, 120)] = np.random.uniform(0, 1)
        extra_fires = np.random.poisson()
        for extra_fire in range(extra_fires):
            input_left_diff[np.random.randint(0, 120)] = np.random.uniform(0, 1)
        extra_fires = np.random.poisson()
        for extra_fire in range(extra_fires):
            input_right_diff[np.random.randint(0, 120)] = np.random.uniform(0, 1)

        # Convolve the synthetic data with a GCaMP kernel and add gaussian noise (std depends on the noise-level)
        exp_kernel_gcamp = np.concatenate((np.zeros(kernel_length), 1 / tau_gcamp * np.exp(
            -np.linspace(0, kernel_length, kernel_length + 1) / tau_gcamp)))
        exp_kernel_gcamp = exp_kernel_gcamp / np.sum(exp_kernel_gcamp)

        gcamp_input_left_lumi[trial, :] = convolve1d(input_left_lumi, exp_kernel_gcamp) + np.random.normal(0, noise_level, 1080)
        gcamp_input_right_lumi[trial, :] = convolve1d(input_right_lumi, exp_kernel_gcamp) + np.random.normal(0, noise_level, 1080)
        gcamp_input_left_diff[trial, :] = convolve1d(input_left_diff, exp_kernel_gcamp) + np.random.normal(0, noise_level, 1080)
        gcamp_input_right_diff[trial, :] = convolve1d(input_right_diff, exp_kernel_gcamp) + np.random.normal(0, noise_level, 1080)

    return gcamp_input_left_diff.mean(axis=0), gcamp_input_right_diff.mean(axis=0), gcamp_input_left_lumi.mean(axis=0), gcamp_input_right_lumi.mean(axis=0)


def create_off_traces(tau_gcamp=4.8, kernel_length=30, noise_level=1.0):
    '''
    This function creates synthetic data for a non-responsive just noisy neuron.
    :param tau_gcamp: Timeconstant used for the convolution with a GCaMP kernel.
    :param kernel_length: Length of the GCaMP kernel.
    :param noise_level: Noise-level used as std in the added gaussian noise.
    :return: Average activity trace across 8 trials of a single synthetic neuron.
    '''
    # Initialize the Activity array to contain 8 trials and (9 stimuli x 120 timepoints = ) 1080 timepoints
    gcamp_input_off = np.zeros((8, 1080))
    # Loop over all trials
    for trial in range(8):
        # There is no response input
        input_off = np.zeros(1080)

        # Add a few (Poisson based) random spikes
        extra_fires = np.random.poisson()
        for extra_fire in range(extra_fires):
            input_off[np.random.randint(0, 120)] = np.random.uniform(0, 1)

        # Convolve with a GCaMP kernel
        exp_kernel_gcamp = np.concatenate((np.zeros(kernel_length), 1 / tau_gcamp * np.exp(
            -np.linspace(0, kernel_length, kernel_length + 1) / tau_gcamp)))
        exp_kernel_gcamp = exp_kernel_gcamp / np.sum(exp_kernel_gcamp)

        # Add gaussian noise based on the noise-level.
        gcamp_input_off[trial, :] = convolve1d(input_off, exp_kernel_gcamp) + np.random.normal(0, noise_level, 1080)

    return gcamp_input_off.mean(axis=0)


def create_model_based_traces_df(n_neurons, noise_level):
    '''
    Create a dataframe of the activity of synthetic neurons.
    :param n_neurons: Total number of neurons (400 of those will be used for the four functional types, the rest will be noise).
    :param noise_level: Current noise level (std of the gaussian noise).
    :return: dataframe with the synthetic traces and their ground truth celltype label.
    '''

    # Initialize the dataframe (9 stimuli times 120 time points = 1080) and cell_type_labels
    all_traces = np.zeros((n_neurons, 1080))
    cell_type_label = np.zeros(n_neurons)
    # Create 100 synthetic neurons of each functional type: left lumi change, right lumi change, left lumi, right lumi.
    for i in range(100):
        all_traces[i*4, :], all_traces[i*4+1, :], all_traces[i*4+2, :], all_traces[i*4+3, :] = create_model_based_traces(noise_level=noise_level)
        cell_type_label[i*4] = 1 # Left lumi change
        cell_type_label[i*4+1] = 2 # Right lumi change
        cell_type_label[i*4+2] = 3 # Left lumi
        cell_type_label[i*4+3] = 4 # Right lumi
    # Fill the rest of the requested amount of neurons with noisy non-responsive neurons.
    for i in range(n_neurons-400):
        all_traces[i+400, :] = create_off_traces(noise_level=noise_level)
        cell_type_label[i+400] = 5

    # Create the traces_df based on the synthetic data.
    first_stim = True
    traces_df = pd.DataFrame()
    # The z position of a neuron is used in a sanity check by the logical statements to make sure the neuron is in the brain.
    column_names = ['ZB_z',]
    ZB_z = np.ones(n_neurons)
    # Loop over all stimuli
    for i, stim_name in enumerate(['lumi_left_dots_left','lumi_left_dots_right','lumi_left_dots_off','lumi_right_dots_left','lumi_right_dots_right','lumi_right_dots_off','lumi_off_dots_left','lumi_off_dots_right','lumi_off_dots_off']):
        # Compute the a,b,c,d,e responses (pre, early, late, early post, late post - stimulus).
        respa = np.nanmean(all_traces[:, i*120:i*120+20], axis=1)
        respb = np.nanmean(all_traces[:, i*120+25:i*120+35], axis=1)
        respc = np.nanmean(all_traces[:, i*120+40:i*120+80], axis=1)
        respd = np.nanmean(all_traces[:, i*120+85:i*120+95], axis=1)
        respe = np.nanmean(all_traces[:, i*120+100:i*120+120], axis=1)

        # Extend the data with the a-e responses as well as the average activity trace.
        column_names.extend([f'{stim_name}_resp_{l}' for l in ['a', 'b', 'c', 'd', 'e']])
        column_names.extend(f'{stim_name}_avg_trace_{i}' for i in range(120))

        # Concatenate all data
        if first_stim:
            data = np.hstack([np.array([respa, respb, respc, respd, respe]).T, all_traces[:, i*120:(i+1)*120]])
            first_stim = False
        else:
            data = np.hstack([data, np.array([respa, respb, respc, respd, respe]).T, all_traces[:, i*120:(i+1)*120]])

    # Create a pandas dataframe with all data.
    full_data = np.hstack([np.array([ZB_z]).T, data])
    traces_df = pd.concat([traces_df, pd.DataFrame(full_data, columns=column_names)])

    return traces_df, cell_type_label


def get_percentage_correct(traces_df, cell_type_label, thresh_resp=0.2, thresh_min=0.1, thresh_peaks_diff=1.25, thresh_below=0.9):
    '''
    This function calculates the performance of the logical statements and linear regression given a dataframe of synthetic data and the ground-truth labels.
    :param traces_df: Dataframe with the average activity per neuron for synthetic neurons.
    :param cell_type_label: List of ground truth labels of the cell types (0=noise, 1=Left Lumi Change, 2=Right Lumi Change, 3=Left Lumi, 4=Right Lumi.
    :param thresh_resp: minimum dF/F activity required to be considered part of the functional types.
    :param thresh_min: During non-responses activity cannot go thresh_min above the pre/post stimulus activity.
    :param thresh_peaks_diff: The change detector peak activity of the stronger contrast needs to be thresh_peaks_diff times higher than the peak activity during the weak contrast stimulus.
    :param thresh_below: The decrease in activity of the luminance integrators needs to go thresh_below times lower than the pre/post stimulus activity.
    '''

    # Classify the synthetic data based on the logical statements
    print('Getting logical statement based dfs')
    diff_left_df = logic_regression_left_diff(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks_diff)
    diff_right_df = logic_regression_right_diff(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks_diff)
    lumi_left_df = logic_regression_left_lumi(traces_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min)
    lumi_right_df = logic_regression_right_lumi(traces_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min)

    # Get the number of correctly labeled cells per category (Since we have hardcoded 100 cells per category, this equals the percentage).
    correct_diff_left = np.sum([i in np.where(cell_type_label == 1)[0] for i in diff_left_df.index])
    correct_diff_right = np.sum([i in np.where(cell_type_label == 2)[0] for i in diff_right_df.index])
    correct_lumi_left = np.sum([i in np.where(cell_type_label == 3)[0] for i in lumi_left_df.index])
    correct_lumi_right = np.sum([i in np.where(cell_type_label == 4)[0] for i in lumi_right_df.index])

    # Get the Stimulus parameters used for linear regression.
    stim_len_timepoints = [120]
    stim_names = ['Same_L', 'Oppo_R', 'Photo_L', 'Oppo_L', 'Same_R', 'Photo_R', 'Motion_L', 'Motion_R', 'No_Stim']

    # Loop over the stimuli to collect all the model input for linear regression.
    for i in range(len(stim_len_timepoints) * len(stim_names)):
        folder_id = i % len(stim_len_timepoints)
        stim_id = int(i / len(stim_len_timepoints))

        left_mot_input, right_mot_input, left_lumi_input, right_lumi_input = get_stim_input_regression(
            stim_len_timepoints[folder_id],
            stim_names[stim_id])

        time = np.linspace(0, (stim_len_timepoints[folder_id] - 1) / 2, stim_len_timepoints[folder_id])

        if i == 0:
            model_input = [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]
        else:
            model_input = np.hstack(
                (model_input, [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]))

    model_params = [5.678052396390699 / 5, 3.765203515714735 / 5, 16.105457978010474 / 5, 7.487431450829603 / 5,
                    0.2136764584807561, 2.0003470409289816, 2.850268651613628]

    # Create the regressors used for linear regression.
    _, regressor_left_lumi, _, _, _, regressor_left_diff, _, regressor_right_lumi, _, _, _, regressor_right_diff = avg_mot_lumi_change(
        model_input, None, *model_params, tau_gcamp=24. / 5, kernel_length=int(150 / 5), window_length=int(20 / 5))

    # Normalize and combine the regressors.
    regressor_left_lumi = (regressor_left_lumi - np.nanmin(regressor_left_lumi)) / (
                np.nanmax(regressor_left_lumi) - np.nanmin(regressor_left_lumi))
    regressor_left_diff = (regressor_left_diff - np.nanmin(regressor_left_diff)) / (
            np.nanmax(regressor_left_diff) - np.nanmin(regressor_left_diff))
    regressor_right_lumi = (regressor_right_lumi - np.nanmin(regressor_right_lumi)) / (
                np.nanmax(regressor_right_lumi) - np.nanmin(regressor_right_lumi))
    regressor_right_diff = (regressor_right_diff - np.nanmin(regressor_right_diff)) / (
            np.nanmax(regressor_right_diff) - np.nanmin(regressor_right_diff))

    regressors_integrators = [regressor_left_lumi, regressor_right_lumi, ]
    regressors_change = [regressor_left_diff, regressor_right_diff]

    # Perform the linear regression
    print('Linear regression')
    (_, _), (lumi_left_ulinreg_df, lumi_right_ulinreg_df), \
        = linear_regression(traces_df, regressors_integrators, rval_thresh=0.8)

    (_, _), (diff_left_ulinreg_df, diff_right_ulinreg_df) \
        = linear_regression(traces_df, regressors_change, rval_thresh=0.6)

    # Get the number of correctly labeled cells per category (Since we have hardcoded 100 cells per category, this equals the percentage).
    correct_diff_linreg_left = np.sum(
        [i in np.where(cell_type_label == 1)[0] for i in diff_left_ulinreg_df.index])
    correct_diff_linreg_right = np.sum(
        [i in np.where(cell_type_label == 2)[0] for i in diff_right_ulinreg_df.index])
    correct_lumi_linreg_left = np.sum(
        [i in np.where(cell_type_label == 3)[0] for i in lumi_left_ulinreg_df.index])
    correct_lumi_linreg_right = np.sum(
        [i in np.where(cell_type_label == 4)[0] for i in lumi_right_ulinreg_df.index])

    return (correct_diff_left, correct_diff_right, correct_lumi_left, correct_lumi_right,
            correct_diff_linreg_left, correct_diff_linreg_right, correct_lumi_linreg_left, correct_lumi_linreg_right,
            diff_left_df, diff_right_df, lumi_left_df, lumi_right_df,
            diff_left_ulinreg_df, diff_right_ulinreg_df, lumi_left_ulinreg_df, lumi_right_ulinreg_df)

def plot_example_synthetic_data(noise_levels, subfigss_diff, subfigss_lumi, n_neurons=402):
    '''
    This function plots the synthetic data example traces for different noise levels.
    This is related to Fig. S4e-f.
    :param noise_levels: List of noise-levels to plot the example data for.
    :param subfigss_diff: List of list of 9 subfigures to show the response of synethetic luminance change detectors to the nine stimuli.
    :param subfigss_lumi: List of list of 9 subfigures to show the response of synthetic luminance integrators to the nine stimuli.
    :param n_neurons: Number of synthetic neurons.
    '''
    if n_neurons < 402:
        exit('You need at least 402 neurons for this simulation. ')
    # Loop over the noise levels.
    for noise_idx, noise_level in enumerate(noise_levels):
        print(noise_level)
        # Create the synthetic data. There will be 100 left lumi change, 100 right lumi change, 100 left lumi and 100 right lumi neurons. The rest will be noise.
        # Since we only plot the example data of those 4 functional types, we do not need to create additional non-responsive neurons here.
        traces_df, cell_type_labels = create_model_based_traces_df(n_neurons, noise_level)

        # cell_type_label 1 = leftward lumi change, cell_type_label 2 = rightward lumi change
        dfL = traces_df[cell_type_labels == 1]
        dfR = traces_df[cell_type_labels == 2]

        # Loop over all stimuli, combine left and rightward neurons and plot their activity.
        pl_color = '#D55E00'
        fillcolor = '#EEAE7C'
        for subfig, stimL, stimR in zip(subfigss_diff[noise_idx],
                                         ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off',
                                          'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                                          'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'],
                                         ['lumi_right_dots_right', 'lumi_right_dots_left', 'lumi_right_dots_off',
                                          'lumi_left_dots_right', 'lumi_left_dots_left', 'lumi_left_dots_off',
                                          'lumi_off_dots_right', 'lumi_off_dots_left', 'lumi_off_dots_off']
                                         ):
            # Plot the median and quartile range response.
            subfig.draw_line(np.arange(0, 60, 0.5), [np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)],
                             yerr_neg = np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 25) for i in range(120)]),
                             yerr_pos = np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 75) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]),
                             lc=pl_color, lw=1, eafc=fillcolor, eaalpha=1.0, ealw=1, eaec=fillcolor)

        # cell_type_label 3 = leftward lumi integrator, cell_type_label 4 = rightward lumi integrator
        dfL = traces_df[cell_type_labels == 3]
        dfR = traces_df[cell_type_labels == 4]

        # Loop over all stimuli, combine left and rightward neurons and plot their activity.
        pl_color = '#E69F00'
        fillcolor = '#F7D280'
        for subfig, stimL, stimR in zip(subfigss_lumi[noise_idx],
                                         ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off',
                                          'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                                          'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'],
                                         ['lumi_right_dots_right', 'lumi_right_dots_left', 'lumi_right_dots_off',
                                          'lumi_left_dots_right', 'lumi_left_dots_left', 'lumi_left_dots_off',
                                          'lumi_off_dots_right', 'lumi_off_dots_left', 'lumi_off_dots_off']
                                         ):
            # Plot the median and quartile range response.
            subfig.draw_line(np.arange(0, 60, 0.5), [np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)],
                             yerr_neg = np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 25) for i in range(120)]),
                             yerr_pos = np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 75) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]),
                             lc=pl_color, lw=1, eafc=fillcolor, eaalpha=1.0, ealw=1, eaec=fillcolor)

    # Plot scale-bars
    subfigss_diff[0][6].draw_line([-4, -4], [0, 1], lc='k')
    subfigss_lumi[0][6].draw_line([-4, -4], [0, 1], lc='k')
    subfigss_diff[2][8].draw_line([40, 60], [-0.54, -0.54], lc='k')
    subfigss_diff[2][8].draw_text(50, -0.9, '20s')
    subfigss_lumi[2][8].draw_line([40, 60], [-0.54, -0.54], lc='k')
    subfigss_lumi[2][8].draw_text(50, -0.9, '20s')
    return

def plot_overview_logical_statements_vs_linear_regression(noise_levels, subfig_diff, subfig_lumi, iterations=5, n_neurons=10000):
    '''
    This function plots the performance of logical statements and linear regression in classifying luminance change detectors (representing brief activity profiles) and
    luminance integrators (representing persistent activity profiles). This is based on synthetic data where we know the ground-truth.
    This is related to Fig. S4g-h.
    :param noise_levels: Array with the different noise-levels to test.
    :param subfig_diff: Subfigure to plot the performance on luminance change detectors.
    :param subfig_lumi: Subfigure to plot the performance on luminance integrators.
    :param iterations: Number of iterations to create the synethetic data and do the fitting per noise level.
    :param n_neurons: Number of synthetic neurons. There are always 100 left luminance change detectors, 100 right luminance change detectors,
    100 left luminance integrators, and 100 right luminance integrators. The rest is noise to resemeble the real brain in which only a few neurons are relevant for the stimulus.
    '''
    if n_neurons < 402:
        exit('You need at least 401 neurons for this simulation. ')
    # Loop over the iterations and noise_levels) .
    for iteration in range(iterations):
        for noise_level in noise_levels:
            print(f'Synthetic data noise level: {noise_level}')
            # Create synethetic data based on the current noise level.
            traces_df, cell_type_labels = create_model_based_traces_df(n_neurons, noise_level)
            # Measure the actual noise level in the synthetic data (The STD of activity during the lumi_off_dots_off stimulus)).
            noise_value = np.nanstd([traces_df[f'lumi_off_dots_off_avg_trace_{i}'].astype(float) for i in range(120)])
            # Calculate the SNR (signal value in synthetic data is 1).
            snr_value = 1 / noise_value

            # Get the percentage correctly labeled cells for logical statements (ls) and linear regression (lr)
            (ls_corr_diff_left, ls_corr_diff_right, ls_corr_lumi_left, ls_corr_lumi_right,
             lr_corr_diff_left, lr_corr_diff_right, lr_corr_lumi_left, lr_corr_lumi_right,
             diff_left_df, diff_right_df, lumi_left_df, lumi_right_df,
             diff_linreg_left_df, diff_linreg_right_df, lumi_linreg_left_df, lumi_linreg_right_df) = get_percentage_correct(traces_df, cell_type_labels)

            # Plot the percentages correctly labeled cells for each noise-level.
            subfig_diff.draw_scatter([snr_value - np.random.uniform(-0.02, 0.02), ], [ls_corr_diff_left], pc='gray', ec=None, ps=3)
            subfig_diff.draw_scatter([snr_value - np.random.uniform(-0.02, 0.02), ], [ls_corr_diff_right], pc='gray', ec=None, ps=3)
            subfig_diff.draw_scatter([snr_value - np.random.uniform(-0.02, 0.02), ], [lr_corr_diff_left], pc='k', ec=None, ps=3)
            subfig_diff.draw_scatter([snr_value - np.random.uniform(-0.02, 0.02), ], [lr_corr_diff_right], pc='k', ec=None, ps=3)

            subfig_lumi.draw_scatter([snr_value + np.random.uniform(-0.02, 0.02), ], [ls_corr_lumi_left], pc='gray', ec=None, ps=3)
            subfig_lumi.draw_scatter([snr_value + np.random.uniform(-0.02, 0.02), ], [ls_corr_lumi_right], pc='gray', ec=None, ps=3)
            subfig_lumi.draw_scatter([snr_value + np.random.uniform(-0.02, 0.02), ], [lr_corr_lumi_left], pc='k', ec=None, ps=3)
            subfig_lumi.draw_scatter([snr_value + np.random.uniform(-0.02, 0.02), ], [lr_corr_lumi_right], pc='k', ec=None, ps=3)

    return

if __name__ == '__main__':
    # A general note for this file: We changed the names of the model nodes later for the paper. In this code we use moslty the old names:
    # multifeature integrator used to be drive, luminance change detector used to be diff, luminance level integrator used to be lumi, luminance increase detector used to be bright, and luminance decrease detector used to be dark. Motion integrator was always motion.

    # Provide the path to save the figures.
    fig_save_path = 'C:/users/katja/Desktop/fig_3.pdf'
    supfig_save_path = 'C:/users/katja/Desktop/fig_S3.pdf'
    supfig2_save_path = 'C:/users/katja/Desktop/fig_S4.pdf'

    # Provide the path to the figure_3 folder.
    fig_3_folder_path = r'Z:\Bahl lab member directories\Katja\paper_data\figure_3'

    # Get the path to the csv file containing all average traces per neuron.
    path_to_traces = rf'{fig_3_folder_path}\imaging_traces_baseline.csv'
    # Get the path to the csv file containing all average traces per neuron reshuffled from the motion_off_luminance_off stimulus.
    path_to_traces_control = rf'{fig_3_folder_path}\imaging_traces_control_baseline.csv'
    # Get the path to the csv file containing all average traces per neuron for the luminance integration experiment.
    path_to_lumiint_traces = rf'{fig_3_folder_path}\imaging_traces_lumi_integrator.csv'
    # Get the path to the mapzebrain regions
    regions_path = rf'{fig_3_folder_path}\all_masks_indexed.hdf5'

    # Get the path to the preprocessed dataframe of an example fish (both the functional plane as well as the overview stack).
    path_to_example_data = rf'{fig_3_folder_path}\2023-01-19_10-38-59_preprocessed_data.h5'
    path_to_example_stack = rf'{fig_3_folder_path}\2023-01-19_12-41-02_preprocessed_data.h5'

    # IDs of the example neurons to be plotted.
    neurons_to_plot = [1386, 1032, 1075, 1238, 1055, 516, 354, 357, 649]

    # Noise levels to create examples of synthetic data with (Fig. S4e-f) as well as the logical statements and linear regression fits (Fig. S4g-h)
    noise_levels_example_data = [2.0, 0.55, 0.3]
    noise_levels_full = [0.3, 0.38, 0.45, 0.55, 0.65, 1., 2.0]

    # Model parameters (given by the code in figure2_S2_behavior_tempdynamics.py). The imaging data is sampled at 0.5s steps, the model based regressors at 0.1s steps, therefore we need to divide the regression model time constants by 5.
    model_params = [5.678052396390699, 3.765203515714735, 16.105457978010474, 7.487431450829603, 0.2136764584807561, 2.0003470409289816, 2.850268651613628]
    model_params_linreg = [5.678052396390699/5, 3.765203515714735/5, 16.105457978010474/5, 7.487431450829603/5,  0.2136764584807561, 2.0003470409289816, 2.850268651613628]

    # Select the regions for Figure 3h and S3a. Note, we originally tested all mapzebrain regions before selecting these.
    regions = ['inferior_medulla_oblongata', 'intermediate_medulla_oblongata', 'superior_medulla_oblongata',
               'superior_dorsal_medulla_oblongata_stripe_1_(entire)', 'superior_dorsal_medulla_oblongata_stripe_2&3',
               'cerebellum', 'tegmentum',
               'tectal_neuropil', 'periventricular_layer',
               'pretectum', 'dorsal_thalamus_proper', 'prethalamus_(ventral_thalamus)',
               'habenula', 'telencephalon', ]

    # Short region names used as plot labels in figure 3h and S3a.
    regions_short_names = [' ',
                           'inf. MO', 'inter. MO', 'sup. MO',
                           'sup. dMO stripe 1', 'sup. dMO stripe 2&3',
                           'cerebellum', 'tegmentum',
                           'tectal neuropil', 'periventricular layer',
                           'pretectum', 'dThalamus', 'vThalamus',
                           'habenula', 'telecephalon', ]

    # Load the traces and control traces dataframe.
    traces_df = pd.read_csv(path_to_traces)
    traces_control_df = pd.read_csv(path_to_traces_control)
    traces_lumiint_df = pd.read_csv(path_to_lumiint_traces)

    print('Traces are loaded. ')

    # Here we define the figures and subpanel outlines (e.g. the limits, ticks and labels of the axes) beloning to figure 3, S3 and S4.
    fig = Figure(fig_width=18, fig_height=17)
    sup_fig_3 = Figure(fig_width=18, fig_height=17)
    sup_fig_4 = Figure(fig_width=18, fig_height=18)

    # Fig. 3c
    example_stack_plot = fig.create_plot(xpos=3.75, ypos=14.9, plot_height=2, plot_width=2, axis_off=True,
                                         xmin=0, xmax=1150, ymin=1150, ymax=0)
    example_loc_plot = fig.create_plot(xpos=3.75, ypos=12.4, plot_height=2, plot_width=2, axis_off=True)
    example_traces_plot = fig.create_plot(xpos=6.5, ypos=12.2, plot_height=4.25, plot_width=7,
                                          xmin=2310, xmax=5100, ymin=-1.6, ymax=9,
                                          yticks=[-0.9, 0.2, 1.3, 2.4, 3.5, 4.6, 5.7, 6.8, 7.9],
                                          yticklabels=['9', '8', '7', '6', '5', '4', '3', '2', '1'],
                                          vspans=[[i*60+2453, i*60+2483, 'lightgray', 1.0] for i in range(44)])
    # Fig. 3d
    cmap = cm.get_cmap('Blues', 80)
    newcolors = cmap(np.linspace(0.25, 1, 80))
    newcolors[0, :] = np.array([1, 1, 1, 1])
    bluewhite_cmap = ListedColormap(newcolors)
    all_cell_overview_plot = fig.create_plot(xpos=13.75, ypos=12.8, plot_height=3, plot_width=3, axis_off=True,
                                             xmin=30, xmax=800, ymin=850, ymax=80, show_colormap=True, zmin=0, zmax=80, colormap=bluewhite_cmap,
                                             zticks=[0, 25, 50, 75], zticklabels=['0', '1', '2', '3+'], zl='neurons/\u00b5m\u00b2')

    # Fig. 3f
    subfigs_traces = create_traces_subplots(fig)

    # Fig. S4d
    sup_subfigs_traces_wta = create_traces_subplots(sup_fig_4, x_l=4.2, y_t=8.3, x_ss=0.75, x_bs=2.5, y_ss=0.75, y_bs=2.5, wta=True)

    # Fig. S4a
    sup_subfigs_traces_linreg = create_traces_subplots(sup_fig_4, x_l=4.2, y_t=16.5, x_ss=0.75, x_bs=2.5, y_ss=0.75, y_bs=2.5, ymax_extra=0.2)

    # Fig. S4c
    sup_subfigs_traces_ctrl = create_traces_subplots(sup_fig_4, x_l=13.1, y_t=16.5, x_ss=0.75, x_bs=2.5, y_ss=0.75, y_bs=2.5)

    # Fig. 3f
    subfigs_locs = create_locs_subplots(fig)

    # Fig. 3g
    loc_comb_plot = fig.create_plot(xpos=0.5, ypos=4.5, plot_height=3.5, plot_width=3.5, axis_off=True,
                                    xmin=30, xmax=800, ymin=850, ymax=80)
    loc_comb_s_plot = fig.create_plot(xpos=0.5, ypos=0.5, plot_height=3.5, plot_width=3.5, axis_off=True,
                                    xmin=30, xmax=800, ymin=850, ymax=80)

    # Fig. S4a
    sup_loc_comb_plot_linreg = sup_fig_4.create_plot(xpos=0.1, ypos=10.2, plot_height=3, plot_width=3, axis_off=True,
                                                     xmin=30, xmax=800, ymin=850, ymax=80)
    # Fig. S4c
    sup_loc_comb_plot_ctrl = sup_fig_4.create_plot(xpos=14.8, ypos=6.7, plot_height=3, plot_width=3, axis_off=True,
                                                   xmin=30, xmax=800, ymin=850, ymax=80)
    # Fig. S4d
    sup_loc_comb_plot_wta = sup_fig_4.create_plot(xpos=9.1, ypos=6.7, plot_height=3, plot_width=3, axis_off=True,
                                                     xmin=30, xmax=800, ymin=850, ymax=80)

    # Fig. S3a (The y-axis is split at 2% therefore we need two plots)
    perc_neurons_plot_bottom = sup_fig_3.create_plot(xpos=1, ypos=13, plot_height=2.5, plot_width=16.75, #5.25
                                         xmin=-1, xmax=91, ymin=0, ymax=2,
                                         xticks=np.arange(90),
                                         xticklabels=regions_short_names * 6,
                                         yticks=[0, 0.5, 1.0, 1.5, 2.0], yl='neurons per region (%)', xticklabels_rotation=90)
    perc_neurons_plot_top = sup_fig_3.create_plot(xpos=1, ypos=15.5, plot_height=0.7, plot_width=16.75, #5.25
                                         xmin=-2, xmax=92, ymin=2, ymax=12,
                                         yticks=[5, 10,])

    example_brain_xy_overview_plots = [[]] * len(regions)
    example_brain_yz_overview_plots = [[]] * len(regions)
    for r in range(len(regions)):
        xr = r%7
        yr = int(r/7)
        example_brain_xy_overview_plots[r] = sup_fig_3.create_plot(xpos=1+2.2*xr, ypos=8.5-2*yr, plot_height=2, plot_width=2 / 2.274, axis_off=True)
        example_brain_yz_overview_plots[r] = sup_fig_3.create_plot(xpos=2+2.2*xr, ypos=8.5-2*yr, plot_height=2, plot_width=2 / 4.395, axis_off=True)

    # Fig. S3b
    subfigs_traces_lumiint = create_lumiint_traces_subplots(sup_fig_3, x_l=0.5, y_t=4)

    # Fig. S3c
    loc_plot_lumiint = sup_fig_3.create_plot(xpos=4, ypos=2, plot_height=3, plot_width=3, axis_off=True, xmin=30, xmax=800, ymin=850, ymax=80)

    # Fig. S3d
    tau_scatter_lumiint = sup_fig_3.create_plot(xpos=9, ypos=1, plot_height=5., plot_width=4.,
                                  xmin=-1, xmax=3, ymin=0, ymax=6,
                                  yticks=[0, 1, 2, 3, 4, 5],
                                  xticks=[0, 1, 2,],
                                  xticklabels=['strong', 'medium', 'weak', ],
                                  yl='time constant (s)')

    # Fig. S4b
    sup_subfigoverlap = sup_fig_4.create_plot(xpos=10, ypos=12.1, plot_height=5.1, plot_width=2.2,
                                              xmin=-1, xmax=11, ymin=0, ymax=1200,
                                              yticks=[0, 250, 500, 750, 1000],
                                              xticks=[0, 1, 3, 4, 6, 7, 9, 10],
                                              xticklabels=['integrators', 'change detectors', 'integrators', 'change detectors', 'integrators', 'change detectors', 'integrators', 'change detectors'],
                                              yl='number of neurons', xticklabels_rotation=90)

    # Fig. S4e-f
    subfigs_diff_traces = [create_traces_single_subplots(sup_fig_4, x_l=0.75, y_t=4.9),
                           create_traces_single_subplots(sup_fig_4, x_l=3.5, y_t=4.9),
                           create_traces_single_subplots(sup_fig_4, x_l=6.25, y_t=4.9)]
    subfigs_lumi_traces = [create_traces_single_subplots(sup_fig_4, x_l=9.75, y_t=4.9, ymax_extra=0.5),
                           create_traces_single_subplots(sup_fig_4, x_l=12.5, y_t=4.9, ymax_extra=0.5),
                           create_traces_single_subplots(sup_fig_4, x_l=15.25, y_t=4.9, ymax_extra=0.5)]
    # Fig. S4g-h
    overview_plot_diff = sup_fig_4.create_plot(xpos=1, ypos=1.0, plot_height=2, plot_width=7.5,
                                          xmin=1, xmax=10, ymin=-5, ymax=105,
                                          yticks=[0, 25, 50, 75, 100], xticks=[2, 4, 6, 8, 10],
                                          vspans=[[4., 7, 'lightgray', 1.0], ],
                                         xl='signal-to-noise-ratio', yl='correctly classified neurons (%)')
    overview_plot_lumi = sup_fig_4.create_plot(xpos=10, ypos=1.0, plot_height=2, plot_width=7.5,
                                          xmin=1, xmax=10, ymin=-5, ymax=105,
                                          yticks=[0, 25, 50, 75, 100], xticks=[2, 4, 6, 8, 10],
                                          vspans=[[4., 7, 'lightgray', 1.0], ],
                                         xl='signal-to-noise-ratio', yl='correctly classified neurons (%)')

    # Plot the example traces (Fig. 3c)
    sub_plot_example_traces(path_to_example_data, path_to_example_stack, neurons_to_plot, example_stack_plot, example_loc_plot, example_traces_plot)
    # Plot the overview of the total cells (Fig. 3d)
    sub_plot_total_cells(traces_df, all_cell_overview_plot)
    # Plot the traces and locations per functional type (Fig. 3f and 3g)
    sub_plot_traces(traces_df, subfigs_traces, subfigs_locs, loc_comb_plot)
    # Add the model predictions to the traces plots (Fig. 3f)
    sub_plot_add_model_prediction_to_traces(subfigs_traces, model_params)
    # Plot the control traces and locations per functional type (Fig. 3g, S4b)
    sub_plot_control(traces_df, traces_control_df, loc_comb_s_plot, sup_subfigoverlap)
    # Plot the number of neurons per region and the percentage of functional type neurons per region per fish (Fig. 3h, S4e).
    sub_plot_n_neurons_per_region(traces_df, perc_neurons_plot_bottom, perc_neurons_plot_top, regions, regions_short_names, regions_path)
    # Plot the brain region cartoons examined in (Fig. s3a).
    sub_plot_brain_region_overview(regions, regions_short_names, fig_3_folder_path, example_brain_xy_overview_plots, example_brain_yz_overview_plots)
    # Plot the lumi integrator check (Fig. S3b-d)
    subplot_lumi_integrator_check(traces_lumiint_df, subfigs_traces_lumiint, loc_plot_lumiint, tau_scatter_lumiint)
    # Plot the linear regression based traces and locations (Fig. S4a-b)
    sub_plot_linear_regression_traces(traces_df, sup_subfigs_traces_linreg, sup_loc_comb_plot_linreg, sup_subfigoverlap,
                                     model_params=model_params_linreg)
    # Plot the number of linear regression based control traces (Fig. S4b)
    sub_plot_linear_regression_traces(traces_control_df, None, None, sup_subfigoverlap, model_params=model_params_linreg)
    # Plot the control and WTA traces and locations (Fig. S4c-d)
    sub_plot_control_and_wta_traces(traces_df, traces_control_df, sup_subfigs_traces_ctrl, sup_subfigs_traces_wta, sup_loc_comb_plot_ctrl, sup_loc_comb_plot_wta)
    # Add the model prediction to the control traces (Fig. S4c)
    sub_plot_add_model_prediction_to_traces(sup_subfigs_traces_ctrl, model_params)
    # Add the model prediction to the WTA traces (Fig. S4d)
    sub_plot_add_model_prediction_to_traces(sup_subfigs_traces_wta, model_params, wta=True)
    # Plot the synthetic example data for multiple noise-levels
    plot_example_synthetic_data(noise_levels_example_data, subfigs_diff_traces, subfigs_lumi_traces)
    # Plot the overview of the comparison between logical statements and linear regression based on synthetic data
    plot_overview_logical_statements_vs_linear_regression(noise_levels_full, overview_plot_diff, overview_plot_lumi)

    # fig.save(fig_save_path)
    sup_fig_3.save(supfig_save_path)
    sup_fig_4.save(supfig2_save_path)

    # Note that Figure 3a-b,e only contain explanatory cartoons without actual data and therefore are not part of this code.

