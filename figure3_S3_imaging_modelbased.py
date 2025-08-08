import h5py
import numpy as np
import pandas as pd
from scipy.stats import linregress
from analysis_helpers.analysis.utils.figure_helper import Figure
from scipy.ndimage import convolve1d
from matplotlib import cm
from matplotlib.colors import ListedColormap
from multifeature_integration_paper.logic_regression_functions import logic_regression_left_motion, logic_regression_left_drive, logic_regression_left_bright, logic_regression_left_dark, logic_regression_left_diff, logic_regression_left_lumi, logic_regression_right_motion, logic_regression_right_drive, logic_regression_right_bright, logic_regression_right_dark, logic_regression_right_diff, logic_regression_right_lumi
from multifeature_integration_paper.logic_regression_functions import logic_regression_right_motion_wta, logic_regression_left_motion_wta, logic_regression_left_lumi_wta, logic_regression_right_lumi_wta
from multifeature_integration_paper.useful_small_funcs import rolling_end_window, create_combined_region_npy_mask

def get_stim_input(stim_len, stim_type, zero_coh=0.1):
    mot_on, mot_off, lumi_on, lumi_off = [100, 400, 100, 400]
    mot_fac, lumi_fac = [1., 1.]
    lumi_pre, lumi_bright, lumi_dark = [0.3, 1., 0.]

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

    # WTA motion vs lumi
    baseline_left_integrated_mot = (left_integrated_mot - 0.1) / 0.9
    baseline_right_integrated_mot = (right_integrated_mot - 0.1) / 0.9
    baseline_left_integrated_ph = (left_integrated_ph - 0.3) / 0.7
    baseline_right_integrated_ph = (right_integrated_ph - 0.3) / 0.7

    left_integrated_mot_bu = baseline_left_integrated_mot - np.clip(baseline_right_integrated_ph, 0, 1) * baseline_left_integrated_mot
    right_integrated_mot_bu = baseline_right_integrated_mot - np.clip(baseline_left_integrated_ph, 0, 1) * baseline_right_integrated_mot
    left_integrated_ph = baseline_left_integrated_ph - baseline_right_integrated_mot * np.clip(baseline_left_integrated_ph, 0, 1)
    right_integrated_ph = baseline_right_integrated_ph - baseline_left_integrated_mot * np.clip(baseline_right_integrated_ph, 0, 1)
    left_integrated_mot = left_integrated_mot_bu
    right_integrated_mot = right_integrated_mot_bu

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

    drive_to_left = w_mot * left_integrated_mot + w_repulsor_pos * repulsion_from_right + baseline + w_attractor_pos * left_integrated_ph
    drive_to_right = w_mot * right_integrated_mot + w_repulsor_pos * repulsion_from_left + baseline + w_attractor_pos * right_integrated_ph

    # For integration of the drive
    exp_kernel_drive = np.concatenate((np.zeros(kernel_length), 1/tau_drive * np.exp(-np.linspace(0, kernel_length, kernel_length+1) / tau_drive)))
    exp_kernel_drive = exp_kernel_drive / np.sum(exp_kernel_drive)
    left_integrated_drive = convolve1d(drive_to_left, exp_kernel_drive)
    right_integrated_drive = convolve1d(drive_to_right, exp_kernel_drive)

    swims_to_left_tot = ((left_integrated_drive - right_integrated_drive) / (left_integrated_drive + right_integrated_drive) + 1) / 2

    swims_to_left_tot_rw = rolling_end_window(swims_to_left_tot, window_length)
    swims_to_left_tot_rw = swims_to_left_tot_rw * 100
    swims_to_left_tot_rw[np.isinf(swims_to_left_tot_rw)] = 50
    swims_to_left_tot_rw[np.isnan(swims_to_left_tot_rw)] = 50

    # Tau GCaMP Migault et al. 2018 for H2B-6s
    exp_kernel_gcamp = np.concatenate((np.zeros(kernel_length), 1/tau_gcamp * np.exp(-np.linspace(0, kernel_length, kernel_length+1) / tau_gcamp)))
    exp_kernel_gcamp = exp_kernel_gcamp / np.sum(exp_kernel_gcamp)

    if subfigss is not None:
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

    drive_to_left = w_mot * left_integrated_mot + w_repulsor_pos * repulsion_from_right + baseline + w_attractor_pos * left_integrated_ph
    drive_to_right = w_mot * right_integrated_mot + w_repulsor_pos * repulsion_from_left + baseline + w_attractor_pos * right_integrated_ph

    # For integration of the drive
    exp_kernel_drive = np.concatenate((np.zeros(kernel_length), 1/tau_drive * np.exp(-np.linspace(0, kernel_length, kernel_length+1) / tau_drive)))
    exp_kernel_drive = exp_kernel_drive / np.sum(exp_kernel_drive)
    left_integrated_drive = convolve1d(drive_to_left, exp_kernel_drive)
    right_integrated_drive = convolve1d(drive_to_right, exp_kernel_drive)

    swims_to_left_tot = ((left_integrated_drive - right_integrated_drive) / (left_integrated_drive + right_integrated_drive) + 1) / 2

    swims_to_left_tot_rw = rolling_end_window(swims_to_left_tot, window_length)
    swims_to_left_tot_rw = swims_to_left_tot_rw * 100
    swims_to_left_tot_rw[np.isinf(swims_to_left_tot_rw)] = 50
    swims_to_left_tot_rw[np.isnan(swims_to_left_tot_rw)] = 50

    # Tau GCaMP Migault et al. 2018 for H2B-6s
    exp_kernel_gcamp = np.concatenate((np.zeros(kernel_length), 1/tau_gcamp * np.exp(-np.linspace(0, kernel_length, kernel_length+1) / tau_gcamp)))
    exp_kernel_gcamp = exp_kernel_gcamp / np.sum(exp_kernel_gcamp)

    if subfigss is not None and linregress is False:
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
    mot_on, mot_off, lumi_on, lumi_off = [20, 80, 20, 80]
    mot_fac, lumi_fac = [1., 1.]
    lumi_pre, lumi_bright, lumi_dark = [0.3, 1., 0.]

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

def create_traces_subplots(fig, x_l=7.5, y_t=10., x_ss=1, x_bs=5.40, y_ss=1, y_bs=3.5, wta=False, ymax_extra=0):

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
    x_bs = 5.4
    y_bs = 3.5
    motion_locs_plot = fig.create_plot(xpos=x_l, ypos=y_t, plot_height=2, plot_width=2, axis_off=True,
                                          xmin=30, xmax=800, ymin=850, ymax=80,
                                       legend_xpos=x_l-2.75, legend_ypos=y_t+2.75)
    lumi_locs_plot = fig.create_plot(xpos=x_l+x_bs, ypos=y_t, plot_height=2, plot_width=2, axis_off=True,
                                          xmin=30, xmax=800, ymin=850, ymax=80,
                                       legend_xpos=x_l+x_bs-2.75, legend_ypos=y_t+2.75)

    if not wta:
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

def sub_plot_traces(traces_df, subfigss, subfigs_loc, subfig_loc_comb):
    print('Finding medium threshold cells')
    thresh_resp = 0.2
    thresh_min = 0.1 #0.3
    thresh_peaks_diff = 1.25
    thresh_peaks = 1.5
    thresh_below = 0.9
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

    for dfL, dfR, subfigs, color, fillcolor in zip([motion_left_df, lumi_left_df, dark_left_df, bright_left_df, drive_left_df, diff_left_df],
                                                    [motion_right_df, lumi_right_df, dark_right_df, bright_right_df, drive_right_df, diff_right_df], subfigss,
                                             ['#359B73', '#E69F00',  '#9F0162', '#F748A5', '#2271B2', '#D55E00'],
                                             ['#8DCDB4', '#F7D280', '#CC7CAD', '#F7A4D0', '#93BADA', '#EEAE7C']):
        for subfig, stimL, stimR in zip(subfigs, ['lumi_left_dots_left',  'lumi_left_dots_right',  'lumi_left_dots_off',
                                          'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                                          'lumi_off_dots_left',   'lumi_off_dots_right',   'lumi_off_dots_off'],
                                        ['lumi_right_dots_right', 'lumi_right_dots_left', 'lumi_right_dots_off',
                                         'lumi_left_dots_right', 'lumi_left_dots_left', 'lumi_left_dots_off',
                                         'lumi_off_dots_right', 'lumi_off_dots_left', 'lumi_off_dots_off']
                                        ):
            subfig.draw_line(np.arange(0, 60, 0.5), [np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)],
                             yerr_neg = np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 25) for i in range(120)]),
                             yerr_pos = np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 75) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]),
                             lc=color, lw=1, eafc=fillcolor, eaalpha=1.0, ealw=1, eaec=fillcolor)

        subfigs[6].draw_line([-4, -4], [0, 0.5], lc='k')
        subfigs[6].draw_text(-20, 0., '0.5 dF/F\u2080', textlabel_rotation=90, textlabel_va="bottom")

    subfigss[4][8].draw_line([40, 60], [-0.34, -0.34], lc='k')
    subfigss[4][8].draw_text(50, -0.6, '20s')

    for dfL, dfR, subfig, color, intens_color, label in zip(
            [motion_left_df, lumi_left_df, dark_left_df, bright_left_df, drive_left_df, diff_left_df],
            [motion_right_df, lumi_right_df, dark_right_df, bright_right_df, drive_right_df, diff_right_df], subfigs_loc,
            ['#8DCDB4', '#F7D280', '#CC7CAD', '#F7A4D0', '#93BADA', '#EEAE7C'],
            ['#359B73', '#E69F00', '#9F0162', '#F748A5', '#2271B2', '#D55E00'],
            ['Motion integrators', 'Luminance integrators', 'Luminance decrease detectors',
             'Luminance increase detectors', 'Multifeature integrators', 'Luminance change detectors']    ):

        subfig.draw_scatter(dfL['ZB_x'].astype(float)*0.798, dfL['ZB_y'].astype(float)*0.798, label=label, pc=intens_color, ec=intens_color, ps=0.75, elw=0.25, alpha=0.75)
        subfig.draw_scatter(dfL['ZB_z'].astype(float)*2 + 515, dfL['ZB_y'].astype(float)*0.798, pc=intens_color, ec=intens_color, ps=0.75, elw=0.25, alpha=0.75)
        subfig.draw_scatter(dfR['ZB_x'].astype(float)*0.798, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=intens_color, ps=0.75, elw=0.25, alpha=0.75)
        subfig.draw_scatter(dfR['ZB_z'].astype(float)*2 + 515, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=intens_color, ps=0.75, elw=0.25, alpha=0.75)
        subfig.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
        subfig.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')

        subfig_loc_comb.draw_scatter(dfL['ZB_x'].astype(float)*0.798, dfL['ZB_y'].astype(float)*0.798, pc=intens_color, ec=intens_color, elw=0.25, ps=0.75, alpha=0.75)
        subfig_loc_comb.draw_scatter(dfL['ZB_z'].astype(float)*2 + 515, dfL['ZB_y'].astype(float)*0.798, pc=intens_color, ec=intens_color, elw=0.25, ps=0.75, alpha=0.75)
        subfig_loc_comb.draw_scatter(dfR['ZB_x'].astype(float)*0.798, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=intens_color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc_comb.draw_scatter(dfR['ZB_z'].astype(float)*2 + 515, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=intens_color, ps=0.75, elw=0.25, alpha=0.75)

    subfig_loc_comb.draw_text(387.5, 50, 'selected cells')
    subfig_loc_comb.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
    subfig_loc_comb.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')
    subfigs_loc[4].draw_line([420, 520], [780, 780], lc='k')
    subfigs_loc[4].draw_text(470, 820, '100\u00b5m')

    return

def linear_regression(df, regressors, rval_thresh=0.85):
    stims = ['lumi_left_dots_left',  'lumi_left_dots_right',  'lumi_left_dots_off',
             'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
             'lumi_off_dots_left',   'lumi_off_dots_right',   'lumi_off_dots_off']

    print(f'Regressing through {len(df)} cells')
    good_cells = [[]] * len(regressors)
    cell_type = [[]] * len(df)
    for cell in range(len(df)):
        if cell % 1000 == 0:
            print(cell)
        trace = []
        for stim in stims:
            trace = np.append(trace, np.array([df[f'{stim}_avg_trace_{i}'][cell] for i in range(120)]))

        norm_trace = (trace - np.nanpercentile(trace, 5)) / (np.nanpercentile(trace, 95) - np.nanpercentile(trace, 5))

        all_coefs = [[]] * len(regressors)
        for r, regressor in enumerate(regressors):
            slope, intercept, r_value, p_value, std_err = linregress(regressor[~np.isnan(norm_trace)],
                                                                     norm_trace[~np.isnan(norm_trace)])
            if r_value > rval_thresh:
                all_coefs[r] = r_value
            else:
                all_coefs[r] = -1
            if r_value > rval_thresh:
                good_cells[r] = np.append(good_cells[r], 1)
            else:
                good_cells[r] = np.append(good_cells[r], 0)

        if np.max(all_coefs) == -1:
            cell_type[cell] = -1
        else:
            cell_type[cell] = np.argmax(all_coefs)
    print('Found all regressions, saving into dfs. ')
    all_dfs = [[]] * len(regressors)
    all_unique_dfs = [[]] * len(regressors)
    for r in range(len(regressors)):
        all_dfs[r] = df[np.logical_and(good_cells[r] >= 1, df['ZB_z'].astype(float) > 0)]
        all_unique_dfs[r] = df[np.logical_and(np.array(cell_type) == r, df['ZB_z'].astype(float) > 0)]

    return all_dfs, all_unique_dfs


def sub_plot_linear_regression_traces(traces_df, subfigss, subfig_loc, subfigoverlap, model_params=[5.73/5, 2.88/5, 14.54/5, 7.61/5, 0.214, 1.922, 2.88]):
    stim_len_timepoints = [120]
    stim_names = ['Same_L', 'Oppo_R', 'Photo_L', 'Oppo_L', 'Same_R', 'Photo_R', 'Motion_L', 'Motion_R', 'No_Stim']

    for i in range(len(stim_len_timepoints) * len(stim_names)):
        folder_id = i % len(stim_len_timepoints)
        stim_id = int(i / len(stim_len_timepoints))

        left_mot_input, right_mot_input, left_lumi_input, right_lumi_input = get_stim_input_regression(stim_len_timepoints[folder_id],
                                                                                                       stim_names[stim_id])

        time = np.linspace(0, (stim_len_timepoints[folder_id] - 1) / 2, stim_len_timepoints[folder_id])

        if i == 0:
            model_input = [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]
        else:
            model_input = np.hstack((model_input, [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]))

    regressor_left_mot, regressor_left_lumi, regressor_left_dark, regressor_left_bright, regressor_left_drive, regressor_left_diff,\
        regressor_right_mot, regressor_right_lumi, regressor_right_dark, regressor_right_bright, regressor_right_drive, regressor_right_diff = avg_mot_lumi_change(model_input, None, *model_params, tau_gcamp=24./5, kernel_length=int(150/5), window_length=int(20/5))

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

    print('Finding linear regression based cells')

    regressors_integrators = [regressor_left_mot, regressor_left_drive, regressor_left_lumi,
                  regressor_right_mot, regressor_right_drive, regressor_right_lumi, ]
    regressors_change = [ regressor_left_diff, regressor_left_bright, regressor_left_dark,
                  regressor_right_diff, regressor_right_bright, regressor_right_dark]

    (motion_left_med_df, drive_left_med_df, lumi_left_med_df, motion_right_med_df, drive_right_med_df, lumi_right_med_df), \
        (motion_left_umed_df, drive_left_umed_df, lumi_left_umed_df, motion_right_umed_df, drive_right_umed_df, lumi_right_umed_df), \
        = linear_regression(traces_df, regressors_integrators, rval_thresh=0.8)

    (diff_left_med_df, bright_left_med_df,dark_left_med_df, diff_right_med_df, bright_right_med_df, dark_right_med_df), \
        (diff_left_umed_df, bright_left_umed_df,dark_left_umed_df, diff_right_umed_df, bright_right_umed_df, dark_right_umed_df) \
        = linear_regression(traces_df, regressors_change, rval_thresh=0.6)

    print(f'There are {len(traces_df)} total neurons. ')

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

    subfigoverlap.draw_vertical_bars([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                                     [num_lumi, num_motion_lumi_overlap, num_motion, num_motion_drive_overlap, num_drive, num_lumi_drive_overlap, num_motion_drive_lumi_overlap,
                                      num_diff, num_bright_diff_overlap, num_bright, num_bright_dark_overlap, num_dark, num_dark_diff_overlap, num_diff_bright_dark_overlap],
                                       vertical_bar_bottom=[0, num_lumi, num_lumi + num_motion_lumi_overlap, num_lumi + num_motion_lumi_overlap + num_motion, num_lumi + num_motion_lumi_overlap + num_motion + num_motion_drive_overlap, num_lumi + num_motion_lumi_overlap + num_motion + num_motion_drive_overlap + num_drive, num_lumi + num_motion_lumi_overlap + num_motion + num_motion_drive_overlap + num_drive + num_lumi_drive_overlap,
                                                            0, num_diff, num_diff + num_bright_diff_overlap, num_diff + num_bright_diff_overlap + num_bright, num_diff + num_bright_diff_overlap + num_bright + num_bright_dark_overlap, num_diff + num_bright_diff_overlap + num_bright + num_bright_dark_overlap + num_dark, num_diff + num_bright_diff_overlap + num_bright + num_bright_dark_overlap + num_dark + num_dark_diff_overlap],
                                       lc=['#E69F00', '#808080', '#359B73', '#808080', '#2271B2', '#808080', '#404040',
                                           '#D55E00', '#808080', '#F748A5', '#808080', '#9F0162', '#808080', '#404040'])
    subfigoverlap.draw_text(0.5, 1200, 'linear\nregression')

    for dfL, dfR, subfigs, color, fillcolor in zip([motion_left_med_df, lumi_left_med_df, dark_left_med_df, bright_left_med_df, drive_left_med_df, diff_left_med_df],
                                                   [motion_right_med_df, lumi_right_med_df, dark_right_med_df, bright_right_med_df, drive_right_med_df, diff_right_med_df],
                                                   subfigss,
                                                     ['#359B73', '#E69F00',  '#9F0162', '#F748A5', '#2271B2', '#D55E00'],
                                                     ['#8DCDB4', '#F7D280', '#CC7CAD', '#F7A4D0', '#93BADA', '#EEAE7C']):
        for subfig, stimL, stimR in zip(subfigs, ['lumi_left_dots_left',  'lumi_left_dots_right',  'lumi_left_dots_off',
                                              'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                                              'lumi_off_dots_left',   'lumi_off_dots_right',   'lumi_off_dots_off'],
                                        ['lumi_right_dots_right', 'lumi_right_dots_left', 'lumi_right_dots_off',
                                         'lumi_left_dots_right', 'lumi_left_dots_left', 'lumi_left_dots_off',
                                         'lumi_off_dots_right', 'lumi_off_dots_left', 'lumi_off_dots_off']
                                        ):
            subfig.draw_line(np.arange(0, 60, 0.5), [np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)],
                             yerr_neg = np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 25) for i in range(120)]),
                             yerr_pos = np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 75) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]),
                             lc=color, lw=1, eafc=fillcolor, eaalpha=1.0, ealw=1, eaec=fillcolor)

        subfigs[6].draw_line([-4, -4], [0, 0.5], lc='k')

    subfigss[4][8].draw_line([40, 60], [-0.34, -0.34], lc='k')
    subfigss[4][8].draw_text(50, -0.6, '20s')

    avg_mot_lumi_change(model_input, subfigss, *model_params, tau_gcamp=24. / 5, kernel_length=int(150 / 5),
                        window_length=int(20 / 5), linregress=True)

    for dfL, dfR, color in zip(
            [motion_left_umed_df, lumi_left_umed_df, dark_left_umed_df, bright_left_umed_df, drive_left_umed_df, diff_left_umed_df],
            [motion_right_umed_df, lumi_right_umed_df, dark_right_umed_df, bright_right_umed_df, drive_right_umed_df, diff_right_umed_df],
            ['#359B73', '#E69F00', '#9F0162', '#F748A5', '#2271B2', '#D55E00']):

        subfig_loc.draw_scatter(dfL['ZB_x'].astype(float)*0.798, dfL['ZB_y'].astype(float)*0.798, pc=color, ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc.draw_scatter(dfL['ZB_z'].astype(float)*2 + 515, dfL['ZB_y'].astype(float)*0.798, pc=color, ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc.draw_scatter(dfR['ZB_x'].astype(float)*0.798, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc.draw_scatter(dfR['ZB_z'].astype(float)*2 + 515, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
        subfig_loc.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')

    subfig_loc.draw_line([420, 520], [780, 780], lc='k')
    subfig_loc.draw_text(470, 820, '100\u00b5m')

    return

def sub_plot_control_and_wta_traces(traces_df, traces_control_df, subfigss, subfigss_wta, subfig_loc_ctrl, subfig_loc_wta):
    thresh_resp = 0.2
    thresh_min = 0.1  # 0.3
    thresh_peaks_diff = 1.25
    thresh_peaks = 1.5
    thresh_below = 0.9
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

    for dfL, dfR, subfigs, color, fillcolor in zip([motion_left_df, lumi_left_df, dark_left_df, bright_left_df, drive_left_df, diff_left_df],
                                                   [motion_right_df, lumi_right_df, dark_right_df, bright_right_df, drive_right_df, diff_right_df],
                                                   subfigss,
                                                     ['#359B73', '#E69F00',  '#9F0162', '#F748A5', '#2271B2', '#D55E00'],
                                                     ['#8DCDB4', '#F7D280', '#CC7CAD', '#F7A4D0', '#93BADA', '#EEAE7C']):
        for subfig, stimL, stimR in zip(subfigs, ['lumi_left_dots_left',  'lumi_left_dots_right',  'lumi_left_dots_off',
                                              'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                                              'lumi_off_dots_left',   'lumi_off_dots_right',   'lumi_off_dots_off'],
                                        ['lumi_right_dots_right', 'lumi_right_dots_left', 'lumi_right_dots_off',
                                         'lumi_left_dots_right', 'lumi_left_dots_left', 'lumi_left_dots_off',
                                         'lumi_off_dots_right', 'lumi_off_dots_left', 'lumi_off_dots_off']
                                        ):
            subfig.draw_line(np.arange(0, 60, 0.5), [np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)],
                             yerr_neg = np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 25) for i in range(120)]),
                             yerr_pos = np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 75) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]),
                             lc=color, lw=1, eafc=fillcolor, eaalpha=1.0, ealw=1, eaec=fillcolor)

        subfigs[6].draw_line([-4, -4], [0, 0.5], lc='k')

    subfigss[4][8].draw_line([40, 60], [-0.34, -0.34], lc='k')
    subfigss[4][8].draw_text(50, -0.6, '20s')

    for dfL, dfR, color in zip(
            [motion_left_df, lumi_left_df, dark_left_df, bright_left_df, drive_left_df, diff_left_df],
            [motion_right_df, lumi_right_df, dark_right_df, bright_right_df, drive_right_df, diff_right_df],
            ['#359B73', '#E69F00', '#9F0162', '#F748A5', '#2271B2', '#D55E00']):

        subfig_loc_ctrl.draw_scatter(dfL['ZB_x'].astype(float)*0.798, dfL['ZB_y'].astype(float)*0.798, pc=color, ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc_ctrl.draw_scatter(dfL['ZB_z'].astype(float)*2 + 515, dfL['ZB_y'].astype(float)*0.798, pc=color, ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc_ctrl.draw_scatter(dfR['ZB_x'].astype(float)*0.798, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc_ctrl.draw_scatter(dfR['ZB_z'].astype(float)*2 + 515, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc_ctrl.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
        subfig_loc_ctrl.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')

    subfig_loc_ctrl.draw_line([420, 520], [780, 780], lc='k')
    subfig_loc_ctrl.draw_text(470, 820, '100\u00b5m')

    thresh_resp = 0.2
    thresh_min = 0.1  # 0.3
    thresh_below = 0.9
    motion_left_df_wta = logic_regression_left_motion_wta(traces_df, thresh_resp=thresh_resp, thresh_min=thresh_min, shuffle_stim_idx=False)
    lumi_left_df_wta = logic_regression_left_lumi_wta(traces_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min, shuffle_stim_idx=False)
    motion_right_df_wta = logic_regression_right_motion_wta(traces_df, thresh_resp=thresh_resp, thresh_min=thresh_min, shuffle_stim_idx=False)
    lumi_right_df_wta = logic_regression_right_lumi_wta(traces_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min, shuffle_stim_idx=False)

    print('We found WTA neurons: ')
    print(f'{len(motion_left_df_wta)} Motion left')
    print(f'{len(motion_right_df_wta)} Motion right')
    print(f'{len(lumi_left_df_wta)} Lumi left')
    print(f'{len(lumi_right_df_wta)} Lumi right')

    for dfL, dfR, subfigs, color, fillcolor in zip([motion_left_df_wta, lumi_left_df_wta,],
                                                   [motion_right_df_wta, lumi_right_df_wta,],
                                                   subfigss_wta,
                                                     ['#359B73', '#E69F00', ],
                                                     ['#8DCDB4', '#F7D280', ]):
        for subfig, stimL, stimR in zip(subfigs, ['lumi_left_dots_left',  'lumi_left_dots_right',  'lumi_left_dots_off',
                                              'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                                              'lumi_off_dots_left',   'lumi_off_dots_right',   'lumi_off_dots_off'],
                                        ['lumi_right_dots_right', 'lumi_right_dots_left', 'lumi_right_dots_off',
                                         'lumi_left_dots_right', 'lumi_left_dots_left', 'lumi_left_dots_off',
                                         'lumi_off_dots_right', 'lumi_off_dots_left', 'lumi_off_dots_off']
                                        ):
            subfig.draw_line(np.arange(0, 60, 0.5), [np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)],
                             yerr_neg=np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 25) for i in range(120)]),
                             yerr_pos=np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 75) for i in range(120)]) - np.array([np.nanpercentile(np.append(dfL[f'{stimL}_avg_trace_{i}'].astype(float), dfR[f'{stimR}_avg_trace_{i}'].astype(float)), 50) for i in range(120)]),
                             lc=color, lw=1, eafc=fillcolor, eaalpha=1.0, ealw=1, eaec=fillcolor)

        subfigs[6].draw_line([-4, -4], [0, 0.5], lc='k')

    subfigss[4][8].draw_line([40, 60], [-0.34, -0.34], lc='k')
    subfigss[4][8].draw_text(50, -0.6, '20s')

    for dfL, dfR, color in zip(
            [motion_left_df_wta, lumi_left_df_wta,],
            [motion_right_df_wta, lumi_right_df_wta,],
            ['#359B73', '#E69F00', '#9F0162', '#F748A5', '#2271B2', '#D55E00']):

        subfig_loc_wta.draw_scatter(dfL['ZB_x'].astype(float)*0.798, dfL['ZB_y'].astype(float)*0.798, pc=color, ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc_wta.draw_scatter(dfL['ZB_z'].astype(float)*2 + 515, dfL['ZB_y'].astype(float)*0.798, pc=color, ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc_wta.draw_scatter(dfR['ZB_x'].astype(float)*0.798, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc_wta.draw_scatter(dfR['ZB_z'].astype(float)*2 + 515, dfR['ZB_y'].astype(float)*0.798, pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
        subfig_loc_wta.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
        subfig_loc_wta.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')

    subfig_loc_wta.draw_line([420, 520], [780, 780], lc='k')
    subfig_loc_wta.draw_text(470, 820, '100\u00b5m')

    return


def sub_plot_add_model_prediction_to_traces(subfigss, model_params=[5.73, 2.88, 14.54, 7.61, 0.214, 19.22, 2.88], wta=False):
    stim_len_timepoints = [600]
    stim_names = ['Motion', 'Photo', 'Same', 'Oppo']

    for i in range(len(stim_len_timepoints) * len(stim_names)):
        folder_id = i % len(stim_len_timepoints)
        stim_id = int(i / len(stim_len_timepoints))

        left_mot_input, right_mot_input, left_lumi_input, right_lumi_input = get_stim_input(stim_len_timepoints[folder_id],
                                                                                            stim_names[stim_id])

        time = np.linspace(0, (stim_len_timepoints[folder_id] - 1) / 10, stim_len_timepoints[folder_id])

        if i == 0:
            model_input = [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]
        else:
            model_input = np.hstack((model_input, [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]))

        if i < len(stim_len_timepoints) * len(stim_names) - 1:
            model_input = np.hstack((model_input, [np.zeros(50), 0.1 * np.ones(50), 0.1 * np.ones(50), 0.3 * np.ones(50),
                                      0.3 * np.ones(50)]))

    if wta:
        wta_mot_lumi_change(model_input, subfigss, *model_params)
    else:
        avg_mot_lumi_change(model_input, subfigss, *model_params)

    return

def sub_plot_control(traces_df, traces_control_df, subfig_s_loc_comb, subfigoverlap):
    print('Finding medium left')
    thresh_resp = 0.2
    thresh_min = 0.1
    thresh_peaks_diff = 1.25
    thresh_peaks = 1.5
    thresh_below = 0.9
    motion_left_df = logic_regression_left_motion(traces_df, thresh_resp=thresh_resp, thresh_min=thresh_min, shuffle_stim_idx=False)
    drive_left_df = logic_regression_left_drive(traces_df, thresh_resp=thresh_resp, shuffle_stim_idx=False)
    lumi_left_df = logic_regression_left_lumi(traces_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min, shuffle_stim_idx=False)
    diff_left_df = logic_regression_left_diff(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks_diff, shuffle_stim_idx=False)
    bright_left_df = logic_regression_left_bright(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
    dark_left_df = logic_regression_left_dark(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)

    print('Finding medium right')
    motion_right_df = logic_regression_right_motion(traces_df, thresh_resp=thresh_resp, thresh_min=thresh_min, shuffle_stim_idx=False)
    drive_right_df = logic_regression_right_drive(traces_df, thresh_resp=thresh_resp, shuffle_stim_idx=False)
    lumi_right_df = logic_regression_right_lumi(traces_df, thresh_resp=thresh_resp, thresh_below=thresh_below, thresh_min=thresh_min, shuffle_stim_idx=False)
    diff_right_df = logic_regression_right_diff(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks_diff, shuffle_stim_idx=False)
    bright_right_df = logic_regression_right_bright(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)
    dark_right_df = logic_regression_right_dark(traces_df, thresh_resp=thresh_resp, thresh_peaks=thresh_peaks, thresh_min=thresh_min, shuffle_stim_idx=False)

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

    print('Finding medium left control')
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
    subfig_s_loc_comb.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
    subfig_s_loc_comb.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')

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

    num_motion = len(motion_left_df_s) + len(motion_right_df_s) - num_motion_drive_overlap - num_motion_lumi_overlap - num_motion_drive_lumi_overlap
    num_drive = len(drive_left_df_s) + len(drive_right_df_s) - num_motion_drive_overlap - num_lumi_drive_overlap - num_motion_drive_lumi_overlap
    num_lumi = len(lumi_left_df_s) + len(lumi_right_df_s) - num_motion_lumi_overlap - num_lumi_drive_overlap - num_motion_drive_lumi_overlap
    num_diff = len(diff_left_df_s) + len(diff_right_df_s) - num_bright_diff_overlap - num_dark_diff_overlap - num_diff_bright_dark_overlap
    num_bright = len(bright_left_df_s) + len(bright_right_df_s) - num_bright_diff_overlap - num_bright_dark_overlap - num_diff_bright_dark_overlap
    num_dark = len(dark_left_df_s) + len(dark_right_df_s) - num_dark_diff_overlap - num_bright_dark_overlap - num_diff_bright_dark_overlap

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
    preproc_hdf5 = h5py.File(path_to_volume, "r")
    avg_im = np.array(preproc_hdf5['average_stack_green_channel'])
    for z, offset in zip(range(5), [0, 50, 250, 300, 350]):
        subfiga.draw_image(np.clip(avg_im[4-z, :, :], np.nanpercentile(avg_im[4-z, :, :], 5), np.nanpercentile(avg_im[4-z, :, :], 95)), colormap='gray',
                           extent=(offset, 800+offset, 800+offset, offset), image_origin='upper')
        subfiga.draw_line([offset, 800+offset, 800+offset, offset, offset], [offset, offset, 800+offset, 800+offset, offset], lc='w', lw=0.5)
    subfiga.draw_scatter([100, 150, 200, 900, 950, 1000], [900, 950, 1000, 100, 150, 200], ec='k', pc='k', ps=1)
    preproc_hdf5.close()

    preproc_hdf5 = h5py.File(path_to_trace_data, "r")
    avg_im = np.array(preproc_hdf5['average_stack_green_channel']).reshape(799, 799)
    subfigb.draw_image(np.clip(avg_im, np.nanpercentile(avg_im, 5), np.nanpercentile(avg_im, 95)), colormap='gray', extent=(0, 800, 800, 0), image_origin='upper')
    for i in range(2103):
        unit_contour = np.array(preproc_hdf5['z_plane0000']['cellpose_segmentation']['unit_contours'][f'{10000+i}'])
        if i in neurons_to_plot:
            continue
        else:
            subfigb.draw_line(unit_contour[:, 0], unit_contour[:, 1], lc='tab:blue', lw=0.2)
    for i in neurons_to_plot:
        unit_contour = np.array(preproc_hdf5['z_plane0000']['cellpose_segmentation']['unit_contours'][f'{10000+i}'])
        subfigb.draw_line(unit_contour[:, 0], unit_contour[:, 1], lc='#00FAFF', lw=0.4)

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

    traces = np.array(preproc_hdf5['z_plane0000']['cellpose_segmentation']['F'])[:, :-50]  # Skipping the last 50 frames because the last stimulus was incomplete.
    stim_starts = np.array(preproc_hdf5['z_plane0000']['stimulus_information'][:, 0])
    stim_ends = np.array(preproc_hdf5['z_plane0000']['stimulus_information'][:, 1])
    stim_types = np.array(preproc_hdf5['z_plane0000']['stimulus_information'][:, 2])
    im_times = np.array(preproc_hdf5['z_plane0000']['imaging_information'][:, 0])[:-50]
    preproc_hdf5.close()

    for i, n in enumerate(neurons_to_plot):
        norm_trace = (traces[n, :] - np.max(traces[n, :])) / (np.max(traces[n, :]) - np.min(traces[n, :]))
        subfigc.draw_line(im_times, norm_trace + i * 1.1)
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
    x = traces_df['ZB_x'][traces_df['ZB_z'] > 0].astype(float)
    y = traces_df['ZB_y'][traces_df['ZB_z'] > 0].astype(float)
    z = traces_df['ZB_z'][traces_df['ZB_z'] > 0].astype(float)

    heatmapxy, xedges, yedges = np.histogram2d(x * 0.798, y * 0.798, bins=[np.arange(35, 475, 5), np.arange(85, 845, 5)])
    heatmapzy, zedges, yedges = np.histogram2d(z * 2, y * 0.798, bins=[np.arange(35, 285, 5), np.arange(85, 845, 5)])
    extentxy = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
    extentzy = [zedges[0]+515, zedges[-1]+515, yedges[-1], yedges[0]]

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

    subfig.draw_image(heatmapxy.T, colormap=bluewhite_cmapxy, extent=extentxy, image_origin='upper')
    subfig.draw_image(heatmapzy.T, colormap=bluewhite_cmapzy, extent=extentzy, image_origin='upper')
    print(np.array(heatmapxy.T).max())
    print(np.array(heatmapxy.T).min())
    print(np.array(heatmapzy.T).max())
    print(np.array(heatmapzy.T).min())
    subfig.draw_line([420, 520], [780, 780], lc='k')
    subfig.draw_text(470, 820, '100\u00b5m')
    subfig.draw_text(560, 900, '~140000 cells total')
    subfig.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
    subfig.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')

    return

def sub_plot_n_neurons_per_region(traces_df, n_neurons_plot, perc_neurons_plot_bottom, perc_neurons_plot_top, regions, regions_short_names, regions_path=r'Z:\Zebrafish atlases\z_brain_atlas\region_masks_mapzebrain_atlas2024\all_masks_indexed.hdf5'):
    thresh_resp = 0.2
    thresh_min = 0.1
    thresh_peaks_diff = 1.25
    thresh_peaks = 1.5
    thresh_below = 0.9

    region_masks = create_combined_region_npy_mask(regions_path, regions=regions)

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

    print('Found all neurons. ')
    for count, (left_df, right_df, type_color) in enumerate(zip([motion_left_df, drive_left_df, lumi_left_df, diff_left_df, bright_left_df, dark_left_df],
                                                                [motion_right_df, drive_right_df, lumi_right_df, diff_right_df, bright_right_df, dark_right_df],
                                                                ['#359B73', '#2271B2', '#E69F00', '#D55E00', '#F748A5', '#9F0162',])):

        mask_left = np.zeros((621, 1406, 138))
        mask_left[left_df['ZB_x'].astype(int), left_df['ZB_y'].astype(int), left_df['ZB_z'].astype(int)] = 1
        mask_right = np.zeros((621, 1406, 138))
        mask_right[right_df['ZB_x'].astype(int), right_df['ZB_y'].astype(int), right_df['ZB_z'].astype(int)] = 1

        overlap = np.append(region_masks[mask_left.astype(bool)], region_masks[mask_right.astype(bool)])
        most_common_regions_idx = np.argsort(np.bincount(overlap.astype(int)))[-3:][::-1]
        if 0 in most_common_regions_idx:
            most_common_regions_idx = np.argsort(np.bincount(overlap.astype(int)))[-4:][::-1]
            most_common_regions_idx = most_common_regions_idx[np.where(most_common_regions_idx != 0)[0]]
        most_common_regions_cnt = np.bincount(overlap.astype(int))[most_common_regions_idx]
        most_common_regions = np.array(regions_short_names)[most_common_regions_idx]

        n_neurons_plot.draw_vertical_bars(np.arange(3) + count * 4, most_common_regions_cnt, lc=type_color)
        for r in range(3):
            n_neurons_plot.draw_text(r + count * 4, -10, most_common_regions[r], textlabel_rotation=90, textlabel_va='top')

    max_percentage = 0
    avg_over_fish = np.nan * np.ones((len(np.unique(traces_df['fish_idx'])), len(regions), 6))
    for f_idx, fish in enumerate(np.unique(traces_df['fish_idx'])):
        fish_df = traces_df[traces_df['fish_idx'] == fish]
        mask_fish = np.zeros((621, 1406, 138))
        mask_fish[fish_df['ZB_x'].astype(int), fish_df['ZB_y'].astype(int), fish_df['ZB_z'].astype(int)] = 1

        neurons_per_region = np.histogram(region_masks[mask_fish.astype(bool)], bins=np.arange(0, len(regions)+2))[0]
        good_regions = np.where(neurons_per_region > 100)[0]
        print(f'Good regions: {good_regions}')
        print(f'NEurons in Telecephalon: {neurons_per_region[14]}')

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

        print(motion_left_df.shape[0], motion_right_df.shape[0])

        print('Found all neurons. ')
        for count, (left_df, right_df, type_color) in enumerate(zip([motion_left_df, drive_left_df, lumi_left_df, diff_left_df, bright_left_df, dark_left_df],
                                                                    [motion_right_df, drive_right_df, lumi_right_df, diff_right_df, bright_right_df, dark_right_df],
                                                                    ['#359B73', '#2271B2', '#E69F00', '#D55E00', '#F748A5', '#9F0162',])):
            mask_left = np.zeros((621, 1406, 138))
            mask_left[left_df['ZB_x'].astype(int), left_df['ZB_y'].astype(int), left_df['ZB_z'].astype(int)] = 1
            mask_right = np.zeros((621, 1406, 138))
            mask_right[right_df['ZB_x'].astype(int), right_df['ZB_y'].astype(int), right_df['ZB_z'].astype(int)] = 1

            overlap = np.append(region_masks[mask_left.astype(bool)], region_masks[mask_right.astype(bool)])

            for r in range(len(regions_short_names)):
                if r == 0 or r not in good_regions:
                    continue
                percentage_neurons = np.sum(overlap == r) / neurons_per_region[r] * 100
                avg_over_fish[f_idx, r-1, count] = percentage_neurons
                if percentage_neurons > 2:
                    perc_neurons_plot_top.draw_scatter(r + len(regions_short_names) * count, percentage_neurons, pc=type_color, ec=None)
                else:
                    perc_neurons_plot_bottom.draw_scatter(r + len(regions_short_names) * count, percentage_neurons, pc=type_color, ec=None)
                if percentage_neurons > max_percentage:
                    max_percentage = percentage_neurons

    for count in range(6):
        perc_neurons_plot_bottom.draw_scatter(np.arange(1, len(regions)+1) + count * len(regions_short_names), np.nanmedian(avg_over_fish[:, :, count], axis=0), pt='_', pc='k')
    print(f'Max percentage: {max_percentage}')
    return

if __name__ == '__main__':
    path_to_traces = r'C:\Users\Katja\Desktop\imaging_traces_baseline.csv'  # Faster to load locally than through the server.
    path_to_traces_control = r'C:\Users\Katja\Desktop\imaging_traces_control_baseline.csv'  # Faster to load locally than through the server.
    path_to_example_data = r'Y:\M11 2P microscopes\Katja\dots_luminance_simultaneous\2023-01-19_10-38-59\2023-01-19_10-38-59_preprocessed_data.h5'
    path_to_example_stack = r'Y:\M11 2P microscopes\Katja\dots_luminance_simultaneous\2023-01-19_12-41-02\2023-01-19_12-41-02_preprocessed_data.h5'
    neurons_to_plot = [1386, 1032, 1075, 1238, 1055, 516, 354, 357, 649]
    model_params = [5.678052396390699, 3.765203515714735, 16.105457978010474, 7.487431450829603, 0.2136764584807561, 2.0003470409289816, 2.850268651613628]
    model_params_linreg = [5.678052396390699/5, 3.765203515714735/5, 16.105457978010474/5, 7.487431450829603/5,  0.2136764584807561, 2.0003470409289816, 2.850268651613628]
    model_params_old = [5.73/5, 2.88/5, 14.54/5, 7.61/5, 0.214, 1.922, 2.88]

    regions = ['inferior_medulla_oblongata', 'intermediate_medulla_oblongata', 'superior_medulla_oblongata',
               'superior_dorsal_medulla_oblongata_stripe_1_(entire)', 'superior_dorsal_medulla_oblongata_stripe_2&3',
               'cerebellum', 'tegmentum',
               'tectal_neuropil', 'periventricular_layer',
               'pretectum', 'dorsal_thalamus_proper', 'prethalamus_(ventral_thalamus)',
               'habenula', 'telencephalon', ]

    regions_short_names = [' ',
                           'inf. MO', 'inter. MO', 'sup. MO',
                           'sup. dMO stripe 1', 'sup. dMO stripe 2&3',
                           'cerebellum', 'tegmentum',
                           'tectal neuropil', 'periventricular layer',
                           'pretectum', 'dThalamus', 'vThalamus',
                           'habenula', 'telecephalon', ]

    traces_df = pd.read_csv(path_to_traces)
    traces_control_df = pd.read_csv(path_to_traces_control)

    print('Traces are loaded. ')

    fig = Figure(fig_width=18, fig_height=17)
    sup_fig_3 = Figure(fig_width=18, fig_height=17)

    example_stack_plot = fig.create_plot(xpos=3.75, ypos=14.9, plot_height=2, plot_width=2, axis_off=True,
                                         xmin=0, xmax=1150, ymin=1150, ymax=0)
    example_loc_plot = fig.create_plot(xpos=3.75, ypos=12.4, plot_height=2, plot_width=2, axis_off=True)
    example_traces_plot = fig.create_plot(xpos=6.5, ypos=12.2, plot_height=4.25, plot_width=7,
                                          xmin=2310, xmax=5100, ymin=-1.6, ymax=9,
                                          yticks=[-0.9, 0.2, 1.3, 2.4, 3.5, 4.6, 5.7, 6.8, 7.9],
                                          yticklabels=['9', '8', '7', '6', '5', '4', '3', '2', '1'],
                                          vspans=[[i*60+2453, i*60+2483, 'lightgray', 1.0] for i in range(44)])
    cmap = cm.get_cmap('Blues', 80)
    newcolors = cmap(np.linspace(0.25, 1, 80))
    newcolors[0, :] = np.array([1, 1, 1, 1])
    bluewhite_cmap = ListedColormap(newcolors)
    all_cell_overview_plot = fig.create_plot(xpos=13.75, ypos=12.8, plot_height=3, plot_width=3, axis_off=True,
                                             xmin=30, xmax=800, ymin=850, ymax=80, show_colormap=True, zmin=0, zmax=80, colormap=bluewhite_cmap,
                                             zticks=[0, 25, 50, 75], zticklabels=['0', '1', '2', '3+'], zl='neurons/\u00b5m\u00b2')

    subfigs_traces = create_traces_subplots(fig)

    sup_subfigs_traces_wta = create_traces_subplots(sup_fig_3, x_l=4.2, y_t=7.2, x_ss=0.75, x_bs=2.5, y_ss=0.75, y_bs=2.5, wta=True)
    sup_subfigs_traces_linreg = create_traces_subplots(sup_fig_3, x_l=4.2, y_t=15.5, x_ss=0.75, x_bs=2.5, y_ss=0.75, y_bs=2.5, ymax_extra=0.2)
    sup_subfigs_traces_ctrl = create_traces_subplots(sup_fig_3, x_l=13.1, y_t=15.5, x_ss=0.75, x_bs=2.5, y_ss=0.75, y_bs=2.5)

    subfigs_locs = create_locs_subplots(fig)

    loc_comb_plot = fig.create_plot(xpos=0.1, ypos=4.5, plot_height=3, plot_width=3, axis_off=True,
                                    xmin=30, xmax=800, ymin=850, ymax=80)
    loc_comb_s_plot = fig.create_plot(xpos=3.25, ypos=4.5, plot_height=3, plot_width=3, axis_off=True,
                                    xmin=30, xmax=800, ymin=850, ymax=80)

    sup_loc_comb_plot_linreg = sup_fig_3.create_plot(xpos=0.1, ypos=9.2, plot_height=3, plot_width=3, axis_off=True,
                                                     xmin=30, xmax=800, ymin=850, ymax=80)
    sup_loc_comb_plot_ctrl = sup_fig_3.create_plot(xpos=14.8, ypos=5.7, plot_height=3, plot_width=3, axis_off=True,
                                                   xmin=30, xmax=800, ymin=850, ymax=80)
    sup_loc_comb_plot_wta = sup_fig_3.create_plot(xpos=9.1, ypos=5.7, plot_height=3, plot_width=3, axis_off=True,
                                                     xmin=30, xmax=800, ymin=850, ymax=80)

    n_neurons_plot = fig.create_plot(xpos=1, ypos=2.2, plot_height=2.25, plot_width=5.25,
                                     xmin=-2, xmax=24, ymin=0, ymax=120,
                                     xticks=[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22,],
                                     xticklabels=[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',],
                                     yticks=[0, 25, 50, 75, 100], yl='Neurons per region')
    perc_neurons_plot_bottom = sup_fig_3.create_plot(xpos=1, ypos=2.2, plot_height=2.5, plot_width=16.75, #5.25
                                         xmin=-1, xmax=91, ymin=0, ymax=2,
                                         xticks=np.arange(90),
                                         xticklabels=regions_short_names * 6,
                                         yticks=[0, 0.5, 1.0, 1.5, 2.0], yl='Neurons per region (%)', xticklabels_rotation=90)
    perc_neurons_plot_top = sup_fig_3.create_plot(xpos=1, ypos=4.7, plot_height=0.7, plot_width=16.75, #5.25
                                         xmin=-2, xmax=92, ymin=2, ymax=12,
                                         yticks=[5, 10,])

    sup_subfigoverlap = sup_fig_3.create_plot(xpos=10, ypos=11.1, plot_height=5.1, plot_width=2.2,
                                              xmin=-1, xmax=8, ymin=0, ymax=1200,
                                              yticks=[0, 250, 500, 750, 1000],
                                              xticks=[0, 1, 3, 4, 6, 7],
                                              xticklabels=['Integrators', 'Change detectors', 'Integrators', 'Change detectors', 'Integrators', 'Change detectors'],
                                              yl='Number of neurons', xticklabels_rotation=90)

    sub_plot_example_traces(path_to_example_data, path_to_example_stack, neurons_to_plot, example_stack_plot, example_loc_plot, example_traces_plot)
    sub_plot_total_cells(traces_df, all_cell_overview_plot)
    sub_plot_traces(traces_df, subfigs_traces, subfigs_locs, loc_comb_plot)
    sub_plot_add_model_prediction_to_traces(subfigs_traces, model_params)

    sub_plot_control(traces_df, traces_control_df, loc_comb_s_plot, sup_subfigoverlap)
    sub_plot_n_neurons_per_region(traces_df, n_neurons_plot, perc_neurons_plot_bottom, perc_neurons_plot_top, regions, regions_short_names)

    sub_plot_linear_regression_traces(traces_df, sup_subfigs_traces_linreg, sup_loc_comb_plot_linreg, sup_subfigoverlap,
                                     model_params=model_params_linreg)

    sub_plot_control_and_wta_traces(traces_df, traces_control_df, sup_subfigs_traces_ctrl, sup_subfigs_traces_wta, sup_loc_comb_plot_ctrl, sup_loc_comb_plot_wta)
    sub_plot_add_model_prediction_to_traces(sup_subfigs_traces_ctrl, model_params)
    sub_plot_add_model_prediction_to_traces(sup_subfigs_traces_wta, model_params, wta=True)

    fig.save('C:/users/katja/Desktop/fig3.pdf')
    sup_fig_3.save('C:/users/katja/Desktop/sup_fig3.pdf')



