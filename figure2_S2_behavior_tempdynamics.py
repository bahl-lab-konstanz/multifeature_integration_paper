import numpy as np
from pathlib import Path
import pandas as pd
from multifeature_integration_paper.figure_helper import Figure
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind
from sklearn.model_selection import GroupShuffleSplit
from scipy.ndimage import convolve1d
from multifeature_integration_paper.useful_small_funcs import cohens_d, rolling_end_window, fill_nans

def get_stim_input(folder_name, stim_len, stim_type, zero_coh=0.1):
    '''
    This function obtains the input to the models for each experiment stimulus.
    :param folder_name: Name of the experiment stimulus.
    :param stim_len: Length of the stimulus in 0.1s steps (e.g. a 25s stimulus is written here as 250).
    :param stim_type: Stimulus type (Motion, Photo, Same or Oppo)
    :param zero_coh: To avoid divisions by zero we set the perceived motion at 0% coherence to a small value.
    :return: Four input arrays: motion_left, motion_right, luminance_left, luminance_right.
    '''
    # Get the experiment specific parameters of when motion start and stops, luminance starts and stops. The strength factors measured from the data,
    # as well as the luminance factors pre/post stimulus and on the bright and dark side of the stimulus.
    if folder_name == 'converted_phototaxis_dotmotion_integration_simultaneous_white':  # Simultaneous
        mot_on, mot_off, lumi_on, lumi_off = [50, 200, 50, 200]
        mot_fac, lumi_fac = [1., 1.]
        lumi_pre, lumi_bright, lumi_dark = [1., 1., 0.]
    elif folder_name == 'phototaxis_dotmotion_simultaneous_low_Sep':  # Simultaneous low
        mot_on, mot_off, lumi_on, lumi_off = [200, 400, 200, 400]
        mot_fac, lumi_fac = [0.12, 1.17]
        lumi_pre, lumi_bright, lumi_dark = [1., 1., 0.]
    elif folder_name == 'converted_phototaxis_dotmotion_integration_peppersalt':  # Peppersalt
        mot_on, mot_off, lumi_on, lumi_off = [50, 150, 150, 250]
        mot_fac, lumi_fac = [0.37, 0.19]
        lumi_pre, lumi_bright, lumi_dark = [0.7, 1., 0.]
    elif folder_name == 'phototaxis_dotmotion_reverse_Sep':  # Reverse
        mot_on, mot_off, lumi_on, lumi_off = [150, 250, 50, 150]
        mot_fac, lumi_fac = [0.41, 1.29]
        lumi_pre, lumi_bright, lumi_dark = [1., 1., 0.]
    elif folder_name == 'phototaxis_dotmotion_white_Sep':  # White
        mot_on, mot_off, lumi_on, lumi_off = [50, 150, 150, 250]
        mot_fac, lumi_fac = [0.33, 0.64]
        lumi_pre, lumi_bright, lumi_dark = [1., 1., 0.]
    elif folder_name == 'converted_phototaxis_dotmotion_integration_long':
        mot_on, mot_off, lumi_on, lumi_off = [50, 250, 250, 450]
        mot_fac, lumi_fac = [0.90, 1.00]
        lumi_pre, lumi_bright, lumi_dark = [0., 0.7, 0.]
    elif folder_name == 'beh_simultaneous_blackwhite_white':
        mot_on, mot_off, lumi_on, lumi_off = [50, 200, 50, 200]
        mot_fac, lumi_fac = [0.82, 0.53]
        lumi_pre, lumi_bright, lumi_dark = [1., 1., 0.]
    elif folder_name == 'beh_simultaneous_blackwhite_black':
        mot_on, mot_off, lumi_on, lumi_off = [50, 200, 50, 200]
        mot_fac, lumi_fac = [1.02, 1.08]
        lumi_pre, lumi_bright, lumi_dark = [0., 1., 0.]
    elif folder_name == 'phototaxis_dotmotion_integration_halfoverlap':
        mot_on, mot_off, lumi_on, lumi_off = [100, 250, 50, 200]
        mot_fac, lumi_fac = [1.00, 1.65]
        lumi_pre, lumi_bright, lumi_dark = [1., 1., 0.]
    elif folder_name == 'beh_simultaneous_10only':
        mot_on, mot_off, lumi_on, lumi_off = [50, 200, 50, 200]
        mot_fac, lumi_fac = [0.13, 1.04]
        lumi_pre, lumi_bright, lumi_dark = [1., 1., 0.]
    elif folder_name == 'imaging_prediction':
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

def avg_mot_lumi(model_input, tau_mot=4.26, tau_ph_eye=12.66, tau_drive=6.46, w_mot=2.783, w_attractor_pos=0.126):
    '''
    This function contains the additive model with motion and luminance level.
    :param model_input: List of 5 input arrays (time, motion left, motion right, luminance left, luminance right).
    :param tau_mot: Timeconstant of the motion integrator.
    :param tau_ph_eye: Timeconstant of the luminance level integrator.
    :param tau_drive: Timeconstant of the multifeature integrator.
    :param w_mot: Weight of the motion pathway
    :param w_attractor_pos: Weight of the luminance level pathway.
    :return: Curve matching the percentage left swims over time.
    '''
    # We add some baseline activity to both multifeature nodes to avoid division by zero. Since both nodes contain this baseline, it doesn't affect the percentage left swims.
    baseline = 1

    time, left_input_mot, right_input_mot, left_input_ph, right_input_ph = model_input

    # Integrate motion
    exp_kernel_mot = np.concatenate((np.zeros(150), 1/tau_mot * np.exp(-np.linspace(0, 150, 151) / tau_mot)))
    exp_kernel_mot = exp_kernel_mot / np.sum(exp_kernel_mot)
    left_integrated_mot = convolve1d(left_input_mot, exp_kernel_mot)
    right_integrated_mot = convolve1d(right_input_mot, exp_kernel_mot)

    # Integrate luminance for each eye
    exp_kernel_ph_eye = np.concatenate((np.zeros(150), 1/tau_ph_eye * np.exp(-np.linspace(0, 150, 151) / tau_ph_eye)))
    exp_kernel_ph_eye = exp_kernel_ph_eye / np.sum(exp_kernel_ph_eye)
    left_integrated_ph = convolve1d(left_input_ph, exp_kernel_ph_eye)
    right_integrated_ph = convolve1d(right_input_ph, exp_kernel_ph_eye)

    # Linear weighted sum going ino the multifeature node (drive)
    drive_to_left = w_mot * left_integrated_mot + baseline + w_attractor_pos * left_integrated_ph
    drive_to_right = w_mot * right_integrated_mot + baseline + w_attractor_pos * right_integrated_ph

    # For integration of the drive
    exp_kernel_drive = np.concatenate((np.zeros(150), 1 / tau_drive * np.exp(-np.linspace(0, 150, 151) / tau_drive)))
    exp_kernel_drive = exp_kernel_drive / np.sum(exp_kernel_drive)
    left_integrated_drive = convolve1d(drive_to_left, exp_kernel_drive)
    right_integrated_drive = convolve1d(drive_to_right, exp_kernel_drive)

    # Get the ratio of leftward swims
    swims_to_left_tot = ((left_integrated_drive - right_integrated_drive) / (
                left_integrated_drive + right_integrated_drive) + 1) / 2

    # Apply a rolling window to match the data preprocessing. And transform ratios to percentages.
    swims_to_left_tot_rw = rolling_end_window(swims_to_left_tot, 20)
    swims_to_left_tot_rw = swims_to_left_tot_rw * 100
    swims_to_left_tot_rw[np.isinf(swims_to_left_tot_rw)] = 50
    swims_to_left_tot_rw[np.isnan(swims_to_left_tot_rw)] = 50

    return swims_to_left_tot_rw

def avg_mot_change(model_input, tau_mot=4.26, tau_ph_rep=12.66, tau_drive=6.46, w_mot=2.783, w_repulsor_pos=1.857):
    '''
    This function contains the additive model with motion and changes in luminance.
    :param model_input: List of 5 input arrays (time, motion left, motion right, luminance left, luminance right).
    :param tau_mot: Timeconstant of the motion integrator.
    :param tau_ph_rep: Timeconstant of the integrator in the luminance change pathway.
    :param tau_drive: Timeconstant of the multifeature integrator.
    :param w_mot: Weight of the motion pathway
    :param w_repulsor_pos: Weight of the luminance change pathway.
    :return: Curve matching the percentage left swims over time.
    '''
    # We add some baseline activity to both multifeature nodes to avoid division by zero. Since both nodes contain this baseline, it doesn't affect the percentage left swims.
    baseline = 1

    time, left_input_mot, right_input_mot, left_input_ph, right_input_ph = model_input

    # Integrate motion
    exp_kernel_mot = np.concatenate((np.zeros(150), 1 / tau_mot * np.exp(-np.linspace(0, 150, 151) / tau_mot)))
    exp_kernel_mot = exp_kernel_mot / np.sum(exp_kernel_mot)
    left_integrated_mot = convolve1d(left_input_mot, exp_kernel_mot)
    right_integrated_mot = convolve1d(right_input_mot, exp_kernel_mot)

    # Calculate repulsion force
    exp_kernel_ph_rep = np.concatenate((np.zeros(150), 1 / tau_ph_rep * np.exp(-np.linspace(0, 150, 151) / tau_ph_rep)))
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
    drive_to_left = w_mot * left_integrated_mot + w_repulsor_pos * repulsion_from_right + baseline
    drive_to_right = w_mot * right_integrated_mot + w_repulsor_pos * repulsion_from_left + baseline

    # For integration of the drive
    exp_kernel_drive = np.concatenate((np.zeros(150), 1 / tau_drive * np.exp(-np.linspace(0, 150, 151) / tau_drive)))
    exp_kernel_drive = exp_kernel_drive / np.sum(exp_kernel_drive)
    left_integrated_drive = convolve1d(drive_to_left, exp_kernel_drive)
    right_integrated_drive = convolve1d(drive_to_right, exp_kernel_drive)

    # Get the ratio of leftward swims
    swims_to_left_tot = ((left_integrated_drive - right_integrated_drive) / (
                left_integrated_drive + right_integrated_drive) + 1) / 2

    # Apply a rolling window to match the data preprocessing. And transform ratios to percentages.
    swims_to_left_tot_rw = rolling_end_window(swims_to_left_tot, 20)
    swims_to_left_tot_rw = swims_to_left_tot_rw * 100
    swims_to_left_tot_rw[np.isinf(swims_to_left_tot_rw)] = 50
    swims_to_left_tot_rw[np.isnan(swims_to_left_tot_rw)] = 50

    return swims_to_left_tot_rw

def avg_lumi_change(model_input, tau_ph_eye=12.66, tau_ph_rep=12.66, tau_drive=6.46, w_attractor_pos=0.126, w_repulsor_pos=1.857):
    '''
    This function contains the additive model with luminance level and changes in luminance.
    :param model_input: List of 5 input arrays (time, motion left, motion right, luminance left, luminance right).
    :param tau_ph_eye: Timeconstant of the luminance level integrator.
    :param tau_ph_rep: Timeconstant of the integrator in the luminance change pathway.
    :param tau_drive: Timeconstant of the multifeature integrator.
    :param w_attractor_pos: Weight of the luminance level pathway.
    :param w_repulsor_pos: Weight of the luminance change pathway.
    :return: Curve matching the percentage left swims over time.
    '''
    # We add some baseline activity to both multifeature nodes to avoid division by zero. Since both nodes contain this baseline, it doesn't affect the percentage left swims.
    baseline = 1

    time, left_input_mot, right_input_mot, left_input_ph, right_input_ph = model_input

    # Integrate luminance for each eye
    exp_kernel_ph_eye = np.concatenate((np.zeros(150), 1 / tau_ph_eye * np.exp(-np.linspace(0, 150, 151) / tau_ph_eye)))
    exp_kernel_ph_eye = exp_kernel_ph_eye / np.sum(exp_kernel_ph_eye)
    left_integrated_ph = convolve1d(left_input_ph, exp_kernel_ph_eye)
    right_integrated_ph = convolve1d(right_input_ph, exp_kernel_ph_eye)

    # Calculate repulsion force
    exp_kernel_ph_rep = np.concatenate((np.zeros(150), 1 / tau_ph_rep * np.exp(-np.linspace(0, 150, 151) / tau_ph_rep)))
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
    drive_to_left = w_repulsor_pos * repulsion_from_right + baseline + w_attractor_pos * left_integrated_ph
    drive_to_right = w_repulsor_pos * repulsion_from_left + baseline + w_attractor_pos * right_integrated_ph

    # For integration of the drive
    exp_kernel_drive = np.concatenate((np.zeros(150), 1 / tau_drive * np.exp(-np.linspace(0, 150, 151) / tau_drive)))
    exp_kernel_drive = exp_kernel_drive / np.sum(exp_kernel_drive)
    left_integrated_drive = convolve1d(drive_to_left, exp_kernel_drive)
    right_integrated_drive = convolve1d(drive_to_right, exp_kernel_drive)

    # Get the ratio of leftward swims
    swims_to_left_tot = ((left_integrated_drive - right_integrated_drive) / (
                left_integrated_drive + right_integrated_drive) + 1) / 2

    # Apply a rolling window to match the data preprocessing. And transform ratios to percentages.
    swims_to_left_tot_rw = rolling_end_window(swims_to_left_tot, 20)
    swims_to_left_tot_rw = swims_to_left_tot_rw * 100
    swims_to_left_tot_rw[np.isinf(swims_to_left_tot_rw)] = 50
    swims_to_left_tot_rw[np.isnan(swims_to_left_tot_rw)] = 50

    return swims_to_left_tot_rw

def avg_mot_lumi_change(model_input, tau_mot=4.26, tau_ph_eye=12.66, tau_ph_rep=12.66, tau_drive=6.46,
                        w_mot=2.783, w_attractor_pos=0.126, w_repulsor_pos=1.857):
    '''
    This function contains the additive model with motion, luminance level and changes in luminance.
    :param model_input: List of 5 input arrays (time, motion left, motion right, luminance left, luminance right).
    :param tau_mot: Timeconstant of the motion integrator.
    :param tau_ph_eye: Timeconstant of the luminance level integrator.
    :param tau_ph_rep: Timeconstant of the integrator in the luminance change pathway.
    :param tau_drive: Timeconstant of the multifeature integrator.
    :param w_mot: Weight of the motion pathway
    :param w_attractor_pos: Weight of the luminance level pathway.
    :param w_repulsor_pos: Weight of the luminance change pathway.
    :return: Curve matching the percentage left swims over time.
    '''
    # We add some baseline activity to both multifeature nodes to avoid division by zero. Since both nodes contain this baseline, it doesn't affect the percentage left swims.
    baseline = 1

    time, left_input_mot, right_input_mot, left_input_ph, right_input_ph = model_input

    # Integrate motion
    exp_kernel_mot = np.concatenate((np.zeros(150), 1 / tau_mot * np.exp(-np.linspace(0, 150, 151) / tau_mot)))
    exp_kernel_mot = exp_kernel_mot / np.sum(exp_kernel_mot)
    left_integrated_mot = convolve1d(left_input_mot, exp_kernel_mot)
    right_integrated_mot = convolve1d(right_input_mot, exp_kernel_mot)

    # Integrate luminance for each eye
    exp_kernel_ph_eye = np.concatenate((np.zeros(150), 1 / tau_ph_eye * np.exp(-np.linspace(0, 150, 151) / tau_ph_eye)))
    exp_kernel_ph_eye = exp_kernel_ph_eye / np.sum(exp_kernel_ph_eye)
    left_integrated_ph = convolve1d(left_input_ph, exp_kernel_ph_eye)
    right_integrated_ph = convolve1d(right_input_ph, exp_kernel_ph_eye)

    # Calculate repulsion force
    exp_kernel_ph_rep = np.concatenate((np.zeros(150), 1 / tau_ph_rep * np.exp(-np.linspace(0, 150, 151) / tau_ph_rep)))
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
    exp_kernel_drive = np.concatenate((np.zeros(150), 1 / tau_drive * np.exp(-np.linspace(0, 150, 151) / tau_drive)))
    exp_kernel_drive = exp_kernel_drive / np.sum(exp_kernel_drive)
    left_integrated_drive = convolve1d(drive_to_left, exp_kernel_drive)
    right_integrated_drive = convolve1d(drive_to_right, exp_kernel_drive)

    # Get the ratio of leftward swims
    swims_to_left_tot = ((left_integrated_drive - right_integrated_drive) / (
                left_integrated_drive + right_integrated_drive) + 1) / 2

    # Apply a rolling window to match the data preprocessing. And transform ratios to percentages.
    swims_to_left_tot_rw = rolling_end_window(swims_to_left_tot, 20)
    swims_to_left_tot_rw = swims_to_left_tot_rw * 100
    swims_to_left_tot_rw[np.isinf(swims_to_left_tot_rw)] = 50
    swims_to_left_tot_rw[np.isnan(swims_to_left_tot_rw)] = 50

    return swims_to_left_tot_rw

def avg_mot_lumi_change_full_node_outputs(model_input, tau_mot=4.26, tau_ph_eye=12.66, tau_ph_rep=12.66, tau_drive=6.46,
                                          w_mot=2.783, w_attractor_pos=0.126, w_repulsor_pos=1.857):
    '''
    This function contains the additive model with motion, luminance level and changes in luminance. It returns the contributions of several nodes.
    :param model_input: List of 5 input arrays (time, motion left, motion right, luminance left, luminance right).
    :param tau_mot: Timeconstant of the motion integrator.
    :param tau_ph_eye: Timeconstant of the luminance level integrator.
    :param tau_ph_rep: Timeconstant of the integrator in the luminance change pathway.
    :param tau_drive: Timeconstant of the multifeature integrator.
    :param w_mot: Weight of the motion pathway
    :param w_attractor_pos: Weight of the luminance level pathway.
    :param w_repulsor_pos: Weight of the luminance change pathway.
    :return: motion_left-, motion_right-, lumi_left-, lumi_right-, change_left-, change_right-, multifeature_left-, multifeature_right- contributions,  Curve matching the percentage left swims over time.
    '''
    # We add some baseline activity to both multifeature nodes to avoid division by zero. Since both nodes contain this baseline, it doesn't affect the percentage left swims.
    baseline = 1

    time, left_input_mot, right_input_mot, left_input_ph, right_input_ph = model_input

    # Integrate motion
    exp_kernel_mot = np.concatenate((np.zeros(150), 1 / tau_mot * np.exp(-np.linspace(0, 150, 151) / tau_mot)))
    exp_kernel_mot = exp_kernel_mot / np.sum(exp_kernel_mot)
    left_integrated_mot = convolve1d(left_input_mot, exp_kernel_mot)
    right_integrated_mot = convolve1d(right_input_mot, exp_kernel_mot)

    # Integrate luminance for each eye
    exp_kernel_ph_eye = np.concatenate((np.zeros(150), 1 / tau_ph_eye * np.exp(-np.linspace(0, 150, 151) / tau_ph_eye)))
    exp_kernel_ph_eye = exp_kernel_ph_eye / np.sum(exp_kernel_ph_eye)
    left_integrated_ph = convolve1d(left_input_ph, exp_kernel_ph_eye)
    right_integrated_ph = convolve1d(right_input_ph, exp_kernel_ph_eye)

    # Calculate repulsion force
    exp_kernel_ph_rep = np.concatenate((np.zeros(150), 1 / tau_ph_rep * np.exp(-np.linspace(0, 150, 151) / tau_ph_rep)))
    exp_kernel_ph_rep = exp_kernel_ph_rep / np.sum(exp_kernel_ph_rep)
    left_integrated_rep = convolve1d(left_input_ph, exp_kernel_ph_rep)
    right_integrated_rep = convolve1d(right_input_ph, exp_kernel_ph_rep)

    dark_left = np.clip(left_integrated_rep - left_input_ph, 0, np.inf)
    dark_right = np.clip(right_integrated_rep - right_input_ph, 0, np.inf)
    bright_left = np.clip(left_input_ph - left_integrated_rep, 0, np.inf)
    bright_right = np.clip(right_input_ph - right_integrated_rep, 0, np.inf)
    repulsion_from_left = bright_left + dark_left
    repulsion_from_right = bright_right + dark_right

    # Linear weighted sum going into the multifeature (drive) node.
    drive_to_left = w_mot * left_integrated_mot + w_repulsor_pos * repulsion_from_right + baseline + w_attractor_pos * left_integrated_ph
    drive_to_right = w_mot * right_integrated_mot + w_repulsor_pos * repulsion_from_left + baseline + w_attractor_pos * right_integrated_ph

    # For integration of the drive
    exp_kernel_drive = np.concatenate((np.zeros(150), 1 / tau_drive * np.exp(-np.linspace(0, 150, 151) / tau_drive)))
    exp_kernel_drive = exp_kernel_drive / np.sum(exp_kernel_drive)
    left_integrated_drive = convolve1d(drive_to_left, exp_kernel_drive)
    right_integrated_drive = convolve1d(drive_to_right, exp_kernel_drive)

    # Get the ratio of leftward swims
    swims_to_left_tot = ((left_integrated_drive - right_integrated_drive) / (
                left_integrated_drive + right_integrated_drive) + 1) / 2

    # Apply a rolling window to match the data preprocessing. And transform ratios to percentages.
    swims_to_left_tot_rw = rolling_end_window(swims_to_left_tot, 20)
    swims_to_left_tot_rw = swims_to_left_tot_rw * 100
    swims_to_left_tot_rw[np.isinf(swims_to_left_tot_rw)] = 50
    swims_to_left_tot_rw[np.isnan(swims_to_left_tot_rw)] = 50

    return (w_mot * left_integrated_mot, w_mot * right_integrated_mot,
            w_attractor_pos * left_integrated_ph, w_attractor_pos * right_integrated_ph,
            w_repulsor_pos * repulsion_from_left, w_repulsor_pos * repulsion_from_right,
            left_integrated_drive, right_integrated_drive, swims_to_left_tot_rw)

def avg_mot_lumi_change_nomfint(model_input, tau_mot=4.26, tau_ph_eye=12.66, tau_ph_rep=12.66,
                                w_mot=2.783, w_attractor_pos=0.126, w_repulsor_pos=1.857):
    '''
    This function contains the additive model with motion, luminance level and changes in luminance. In this model the multifeature neurons simply add without temporal integration.
    :param model_input: List of 5 input arrays (time, motion left, motion right, luminance left, luminance right).
    :param tau_mot: Timeconstant of the motion integrator.
    :param tau_ph_eye: Timeconstant of the luminance level integrator.
    :param tau_ph_rep: Timeconstant of the integrator in the luminance change pathway.
    :param w_mot: Weight of the motion pathway
    :param w_attractor_pos: Weight of the luminance level pathway.
    :param w_repulsor_pos: Weight of the luminance change pathway.
    :return: Curve matching the percentage left swims over time.
    '''
    # We add some baseline activity to both multifeature nodes to avoid division by zero. Since both nodes contain this baseline, it doesn't affect the percentage left swims.
    baseline = 1

    time, left_input_mot, right_input_mot, left_input_ph, right_input_ph = model_input

    # Integrate motion
    exp_kernel_mot = np.concatenate((np.zeros(150), 1 / tau_mot * np.exp(-np.linspace(0, 150, 151) / tau_mot)))
    exp_kernel_mot = exp_kernel_mot / np.sum(exp_kernel_mot)
    left_integrated_mot = convolve1d(left_input_mot, exp_kernel_mot)
    right_integrated_mot = convolve1d(right_input_mot, exp_kernel_mot)

    # Integrate luminance for each eye
    exp_kernel_ph_eye = np.concatenate((np.zeros(150), 1 / tau_ph_eye * np.exp(-np.linspace(0, 150, 151) / tau_ph_eye)))
    exp_kernel_ph_eye = exp_kernel_ph_eye / np.sum(exp_kernel_ph_eye)
    left_integrated_ph = convolve1d(left_input_ph, exp_kernel_ph_eye)
    right_integrated_ph = convolve1d(right_input_ph, exp_kernel_ph_eye)

    # Calculate repulsion force
    exp_kernel_ph_rep = np.concatenate((np.zeros(150), 1 / tau_ph_rep * np.exp(-np.linspace(0, 150, 151) / tau_ph_rep)))
    exp_kernel_ph_rep = exp_kernel_ph_rep / np.sum(exp_kernel_ph_rep)
    left_integrated_rep = convolve1d(left_input_ph, exp_kernel_ph_rep)
    right_integrated_rep = convolve1d(right_input_ph, exp_kernel_ph_rep)

    dark_left = np.clip(left_integrated_rep - left_input_ph, 0, np.inf)
    dark_right = np.clip(right_integrated_rep - right_input_ph, 0, np.inf)
    bright_left = np.clip(left_input_ph - left_integrated_rep, 0, np.inf)
    bright_right = np.clip(right_input_ph - right_integrated_rep, 0, np.inf)
    repulsion_from_left = bright_left + dark_left
    repulsion_from_right = bright_right + dark_right

    # Compute the linear weighted sum to go into the multifeature (drive) node.
    left_integrated_drive = w_mot * left_integrated_mot + w_repulsor_pos * repulsion_from_right + baseline + w_attractor_pos * left_integrated_ph
    right_integrated_drive = w_mot * right_integrated_mot + w_repulsor_pos * repulsion_from_left + baseline + w_attractor_pos * right_integrated_ph

    # Get the ratio leftward swims
    swims_to_left_tot = ((left_integrated_drive - right_integrated_drive) / (
                left_integrated_drive + right_integrated_drive) + 1) / 2

    # Apply a rolling window to match the data preprocessing and transform ratios to percentages.
    swims_to_left_tot_rw = rolling_end_window(swims_to_left_tot, 20)
    swims_to_left_tot_rw = swims_to_left_tot_rw * 100
    swims_to_left_tot_rw[np.isinf(swims_to_left_tot_rw)] = 50
    swims_to_left_tot_rw[np.isnan(swims_to_left_tot_rw)] = 50

    return swims_to_left_tot_rw

def avg_mot_lumi_change_withmotinhib(model_input, tau_mot=4.26, tau_ph_eye=12.66, tau_ph_rep=12.66, tau_drive=6.46,
                        w_mot=2.783, w_attractor_pos=0.126, w_repulsor_pos=1.857, w_mot_inhib=0):
    '''
    This function contains the additive model with motion, luminance level and changes in luminance. Additionally, it contains a mutually inhibitory connection between the left and right motion integrators.
    :param model_input: List of 5 input arrays (time array, motion left, motion right, luminance left, luminance right).
    :param tau_mot: Timeconstant of the motion integrator.
    :param tau_ph_eye: Timeconstant of the luminance level integrator.
    :param tau_ph_rep: Timeconstant of the integrator in the luminance change pathway.
    :param tau_drive: Timeconstant of the multifeature integrator.
    :param w_mot: Weight of the motion pathway
    :param w_attractor_pos: Weight of the luminance level pathway.
    :param w_repulsor_pos: Weight of the luminance change pathway.
    :param w_mot_inhib: Relative weight of the motion inhibition.
    :return: Curve matching the percentage left swims over time.
    '''
    # We add some baseline activity to both multifeature nodes to avoid division by zero. Since both nodes contain this baseline, it doesn't affect the percentage left swims.
    baseline = 1

    time, left_input_mot, right_input_mot, left_input_ph, right_input_ph = model_input

    # Integrate motion
    exp_kernel_mot = np.concatenate((np.zeros(150), 1 / tau_mot * np.exp(-np.linspace(0, 150, 151) / tau_mot)))
    exp_kernel_mot = exp_kernel_mot / np.sum(exp_kernel_mot)
    left_integrated_mot = convolve1d(left_input_mot, exp_kernel_mot)
    right_integrated_mot = convolve1d(right_input_mot, exp_kernel_mot)
    # Compute the mutual inhibition between the motion nodes (and make sure they remain positive).
    left_integrated_mot = left_integrated_mot - w_mot_inhib * right_integrated_mot
    right_integrated_mot = right_integrated_mot - w_mot_inhib * left_integrated_mot
    left_integrated_mot[left_integrated_mot < 0] = 0
    right_integrated_mot[right_integrated_mot < 0] = 0

    # Integrate luminance for each eye
    exp_kernel_ph_eye = np.concatenate((np.zeros(150), 1 / tau_ph_eye * np.exp(-np.linspace(0, 150, 151) / tau_ph_eye)))
    exp_kernel_ph_eye = exp_kernel_ph_eye / np.sum(exp_kernel_ph_eye)
    left_integrated_ph = convolve1d(left_input_ph, exp_kernel_ph_eye)
    right_integrated_ph = convolve1d(right_input_ph, exp_kernel_ph_eye)

    # Calculate repulsion force
    exp_kernel_ph_rep = np.concatenate((np.zeros(150), 1 / tau_ph_rep * np.exp(-np.linspace(0, 150, 151) / tau_ph_rep)))
    exp_kernel_ph_rep = exp_kernel_ph_rep / np.sum(exp_kernel_ph_rep)
    left_integrated_rep = convolve1d(left_input_ph, exp_kernel_ph_rep)
    right_integrated_rep = convolve1d(right_input_ph, exp_kernel_ph_rep)

    dark_left = np.clip(left_integrated_rep - left_input_ph, 0, np.inf)
    dark_right = np.clip(right_integrated_rep - right_input_ph, 0, np.inf)
    bright_left = np.clip(left_input_ph - left_integrated_rep, 0, np.inf)
    bright_right = np.clip(right_input_ph - right_integrated_rep, 0, np.inf)
    repulsion_from_left = bright_left + dark_left
    repulsion_from_right = bright_right + dark_right

    # Compute the linear weighted sum into the multifeature node (drive).
    drive_to_left = w_mot * left_integrated_mot + w_repulsor_pos * repulsion_from_right + baseline + w_attractor_pos * left_integrated_ph
    drive_to_right = w_mot * right_integrated_mot + w_repulsor_pos * repulsion_from_left + baseline + w_attractor_pos * right_integrated_ph

    # For integration of the drive
    exp_kernel_drive = np.concatenate((np.zeros(150), 1 / tau_drive * np.exp(-np.linspace(0, 150, 151) / tau_drive)))
    exp_kernel_drive = exp_kernel_drive / np.sum(exp_kernel_drive)
    left_integrated_drive = convolve1d(drive_to_left, exp_kernel_drive)
    right_integrated_drive = convolve1d(drive_to_right, exp_kernel_drive)

    # Calculate the ratio of leftward swims.
    swims_to_left_tot = ((left_integrated_drive - right_integrated_drive) / (
                left_integrated_drive + right_integrated_drive) + 1) / 2

    # Apply a rollowing window to match the data processing step. and transform ratios to percentage.
    swims_to_left_tot_rw = rolling_end_window(swims_to_left_tot, 20)
    swims_to_left_tot_rw = swims_to_left_tot_rw * 100
    swims_to_left_tot_rw[np.isinf(swims_to_left_tot_rw)] = 50
    swims_to_left_tot_rw[np.isnan(swims_to_left_tot_rw)] = 50

    return swims_to_left_tot_rw


def get_data(folder_name, stim_len_timepoints, stim_name, path_to_folders, nsplits=3, debug=False):
    '''
    This function retrieves the data of the specified experiments and stimulus types.
    :param folder_names: List of experiment stimuli to train on.
    :param stim_len_timepoints: List of the length of each stimulus in 0.1s steps (e.g. a 25s stimulus is written as 250 in this list).
    :param stim_names: List of stimulus types to train on (e.g. Motion, Photo, Same and Oppo).
    :param path_to_folders: path to the folder containing all experiments.
    :param nsplits: Number of splits to split the data in, one of the splits will be used as testing data, the rest will be used as training data. Default is 3.
    :param debug: Print how many fish are used for testing and how many for training.
    :return: The data: mean training values, SEM training values, mean testing values, SEM testing values.
    '''
    # Load the analysed dataframe.
    try:
        path_to_local_folder = rf'{path_to_folders}\{folder_name}'
        df_path = f'{path_to_local_folder}/data_analysed.hdf5'
        df = pd.read_hdf(df_path)
    except:
        path_to_local_folder = rf'{path_to_folders}\{folder_name}\Analysis'
        df_path = f'{path_to_local_folder}/data_analysed.hdf5'
        df = pd.read_hdf(df_path)

    # Group the data into the splits.
    gkf = GroupShuffleSplit(n_splits=nsplits)
    datasplitss = {}
    datasplitss.update({f"{folder_name}": [(train_ids, test_ids) for train_ids, test_ids in
                                           gkf.split(df.index.unique('experiment_ID'),
                                                     groups=df.index.unique('experiment_ID').tolist())]})

    # Create the two separate dataframes for the training and testing data.
    train_df = df[np.isin(df.index.get_level_values('experiment_ID'),
                          df.index.unique('experiment_ID')[datasplitss[f'{folder_name}'][0][0]])]
    test_df = df[np.isin(df.index.get_level_values('experiment_ID'),
                         df.index.unique('experiment_ID')[datasplitss[f'{folder_name}'][0][1]])]

    if debug:
        print(f'{folder_name} - {stim_name}: ')
        print(f'Train N FISH: {len(train_df.index.unique("experiment_ID"))}')
        print(f'Test N FISH: {len(test_df.index.unique("experiment_ID"))}\n')

    # Extract the mean and sem of percentage_left swims for the specific stimulus type. Both for the training and testing data set.
    stimulus_data = train_df.xs(stim_name, level='stimulus_name')
    data_mean = stimulus_data.groupby('window_time').mean()['percentage_left']
    data_sem = stimulus_data.groupby('window_time').std()['percentage_left'] / np.sqrt(
        len(train_df.index.unique('experiment_ID')))
    stimulus_data_test = test_df.xs(stim_name, level='stimulus_name')
    data_mean_test = stimulus_data_test.groupby('window_time').mean()['percentage_left']
    data_sem_test = stimulus_data_test.groupby('window_time').std()['percentage_left'] / np.sqrt(
        len(test_df.index.unique('experiment_ID')))

    # This is needed in case there were no bouts in the first 0.1 s (this happens). We need to make sure the length of the dataframe matches across stimulus types.
    for iter in range(10):
        if len(data_mean) < stim_len_timepoints:
            data_mean = pd.concat([pd.Series([0.5]), data_mean])
            data_sem = pd.concat([pd.Series([0.]), data_sem])

        if len(data_mean_test) < stim_len_timepoints:
            data_mean_test = pd.concat([pd.Series([0.5]), data_mean_test])
            data_sem_test = pd.concat([pd.Series([0.]), data_sem_test])

    # Replace first 0,5 seconds of each stimulus with a flat line - there are not enough bouts in this initial time for a good percentage estimate.
    # And since the stimulus input is always 0 here - it won't affect the model fitting.
    data_mean[:0.5] = 0.5
    data_sem[:0.5] = 0.05
    data_mean_test[:0.5] = 0.5
    data_sem_test[:0.5] = 0.05

    data_mean = data_mean * 100
    data_sem = data_sem * 100
    data_mean_test = data_mean_test * 100
    data_sem_test = data_sem_test * 100

    # Fill any potential nans since scipy's curve-fit can't handle it.
    data_mean = fill_nans(data_mean)
    data_sem = fill_nans(data_sem)
    data_mean_test = fill_nans(data_mean_test)
    data_sem_test = fill_nans(data_sem_test)

    return data_mean, data_sem, data_mean_test, data_sem_test

def train_model_once(model_func, folder_names, stim_len_timepoints, stim_names, path_to_folders, subfig=None):
    '''
    This function trains the model once.
    :param model_func: name of the model, e.g. avg_mot_lumi_change.
    :param folder_names: List of experiment stimuli to train on.
    :param stim_len_timepoints: List of the length of each stimulus in 0.1s steps (e.g. a 25s stimulus is written as 250 in this list).
    :param stim_names: List of stimulus types to train on (e.g. Motion, Photo, Same and Oppo).
    :param path_to_folders: Path the folder containing all experiment folders.
    :param subfig: If not None plot the data and model fit from the training. This is not used anymore in the paper.
    :return: best model parameters, best_mse, list_of_folder_ids
    '''

    n_loops = len(stim_len_timepoints) * len(stim_names)
    list_of_folder_ids = []
    list_of_stim_ids = []
    # Loop over randomly drawn stimulus combinations.
    for i in range(n_loops):
        random_folder_id = np.random.randint(len(folder_names))
        random_stim_id = np.random.randint(len(stim_names))

        list_of_folder_ids = np.append(list_of_folder_ids, random_folder_id)
        list_of_stim_ids = np.append(list_of_stim_ids, random_stim_id)

        # Get the data, split into a training (2/3 of the data) and a testing (1/3 of the data) set.
        # Note that training is only based on the mean, the sem we only use for the plotting of the data.
        data_mean, data_sem, data_mean_test, data_sem_test = get_data(folder_names[random_folder_id],
                                                                      stim_len_timepoints[random_folder_id],
                                                                      stim_names[random_stim_id],
                                                                      path_to_folders=path_to_folders)
        # Get the input into the model
        left_mot_input, right_mot_input, left_lumi_input, right_lumi_input = get_stim_input(
            folder_names[random_folder_id], stim_len_timepoints[random_folder_id], stim_names[random_stim_id])

        time = np.linspace(0, (len(data_mean) - 1) / 10, len(data_mean))

        # Collect the model input, data mean and sem for training and testing, folder and stim ids.
        if i == 0:
            model_input = [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]
            data_full = data_mean
            data_sem_full = data_sem
            data_full_test = data_mean_test
            data_sem_full_test = data_sem_test
            folder_ids = random_folder_id * np.ones(stim_len_timepoints[random_folder_id])
            stim_ids = random_stim_id * np.ones(stim_len_timepoints[random_folder_id])
        else:
            model_input = np.hstack(
                (model_input, [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]))
            data_full = np.append(data_full, data_mean)
            data_full_test = np.append(data_full_test, data_mean_test)
            data_sem_full = np.append(data_sem_full, data_sem)
            data_sem_full_test = np.append(data_sem_full_test, data_sem_test)
            folder_ids = np.append(folder_ids, random_folder_id * np.ones(stim_len_timepoints[random_folder_id]))
            stim_ids = np.append(stim_ids, random_stim_id * np.ones(stim_len_timepoints[random_folder_id]))

        # We add a chunk of 5s between the different stimuli to allow the model to return back to baseline before starting the next stimulus.
        if i < n_loops - 1:
            model_input = np.hstack((model_input, [np.zeros(50), np.zeros(50), np.zeros(50), np.ones(50), np.ones(50)]))
            data_full = np.append(data_full, 50 * np.ones(50))
            data_full_test = np.append(data_full_test, 50 * np.ones(50))
            data_sem_full = np.append(data_sem_full, 0 * np.ones(50))
            data_sem_full_test = np.append(data_sem_full_test, 0 * np.ones(50))
            folder_ids = np.append(folder_ids, -1 * np.ones(50))
            stim_ids = np.append(stim_ids, -1 * np.ones(50))

    # Find the best_mse across 5 model fitting with different initital parameter guesses.
    # Usually scipy's curve-fit finds the best fit immediately, but since it sometimes gets stuck and returns a warning 'optimal parameters not found', we give it 5 tries to be quite sure it always finds the true best fit.
    best_mse = 1000
    for c in range(5):
        # Define the initial parameters, as well as the bounds for each model option (timeconstants are bound between 0 and 100, weights are bound between 0 and 25).
        if model_func == avg_mot_lumi:
            tau_mot = np.random.uniform(0.1, 100)
            tau_ph_eye = np.random.uniform(0.1, 100)
            tau_drive = np.random.uniform(0.1, 100)
            w_mot = np.random.uniform(0.1, 25)
            w_attractor_pos = np.random.uniform(0.1, 25)

            p0 = [tau_mot, tau_ph_eye, tau_drive, w_mot, w_attractor_pos,]
            bounds = ([0, 0, 0, 0, 0], [100, 100, 100, 25, 25])
        elif model_func == avg_mot_change:
            tau_mot = np.random.uniform(0.1, 100)
            tau_ph_rep = np.random.uniform(0.1, 100)
            tau_drive = np.random.uniform(0.1, 100)
            w_mot = np.random.uniform(0.1, 25)
            w_repulsor_pos = np.random.uniform(0.1, 25)

            p0 = [tau_mot, tau_ph_rep, tau_drive, w_mot, w_repulsor_pos,]
            bounds = ([0, 0, 0, 0, 0], [100, 100, 100, 25, 25])
        elif model_func == avg_lumi_change:
            tau_ph_eye = np.random.uniform(0.1, 100)
            tau_ph_rep = np.random.uniform(0.1, 100)
            tau_drive = np.random.uniform(0.1, 100)
            w_attractor_pos = np.random.uniform(0.1, 25)
            w_repulsor_pos = np.random.uniform(0.1, 25)

            p0 = [tau_ph_eye, tau_ph_rep, tau_drive, w_attractor_pos, w_repulsor_pos]
            bounds = ([0, 0, 0, 0, 0], [100, 100, 100, 25, 25])
        elif model_func == avg_mot_lumi_change:
            tau_mot = np.random.uniform(0.1, 100)
            tau_ph_eye = np.random.uniform(0.1, 100)
            tau_ph_rep = np.random.uniform(0.1, 100)
            tau_drive = np.random.uniform(0.1, 100)
            w_mot = np.random.uniform(0.1, 25)
            w_attractor_pos = np.random.uniform(0.1, 25)
            w_repulsor_pos = np.random.uniform(0.1, 25)

            p0 = [tau_mot, tau_ph_eye, tau_ph_rep, tau_drive, w_mot, w_attractor_pos, w_repulsor_pos,]
            bounds = ([0, 0, 0, 0, 0, 0, 0], [100, 100, 100, 100, 25, 25, 25])

        elif model_func == avg_mot_lumi_change_nomfint:
            tau_mot = np.random.uniform(0.1, 100)
            tau_ph_eye = np.random.uniform(0.1, 100)
            tau_ph_rep = np.random.uniform(0.1, 100)
            w_mot = np.random.uniform(0.1, 25)
            w_attractor_pos = np.random.uniform(0.1, 25)
            w_repulsor_pos = np.random.uniform(0.1, 25)

            p0 = [tau_mot, tau_ph_eye, tau_ph_rep, w_mot, w_attractor_pos, w_repulsor_pos,]
            bounds = ([0, 0, 0, 0, 0, 0], [100, 100, 100, 25, 25, 25])

        elif model_func == avg_mot_lumi_change_withmotinhib:
            tau_mot = np.random.uniform(0.1, 100)
            tau_ph_eye = np.random.uniform(0.1, 100)
            tau_ph_rep = np.random.uniform(0.1, 100)
            tau_drive = np.random.uniform(0.1, 100)
            w_mot = np.random.uniform(0.1, 25)
            w_attractor_pos = np.random.uniform(0.1, 25)
            w_repulsor_pos = np.random.uniform(0.1, 25)
            w_mot_inhib = np.random.uniform(0.001, 0.999)

            p0 = [tau_mot, tau_ph_eye, tau_ph_rep, tau_drive, w_mot, w_attractor_pos, w_repulsor_pos, w_mot_inhib]
            bounds = ([0, 0, 0, 0, 0, 0, 0, 0], [100, 100, 100, 100, 25, 25, 25, 1])

        # Find the best fit between the model and the training data.
        try:
            popt, pcov = curve_fit(model_func, model_input, data_full, p0=p0, bounds=bounds)
        except:
            print('Warning optimal params not found, using initial params instead.')
            popt = p0
        # Calculate the MSE between the left-out testing data and the model.
        test_mse = np.nanmean(np.square(model_func(model_input, *popt) - data_full_test))

        # Update the best MSE and parameter set.
        if test_mse < best_mse:
            best_mse = test_mse
            best_popt = popt

    # If subfig is not None, plot the data with the model fit. Since the stimuli are drawn randomly and differ in length, we cannot know before how long the x-axis should be. We give a warning if not all data is shown.
    if subfig is not None:
        if len(data_full_test) > 7500:
            print(f'Warning: Length of data outside xmax (not the full plot will be visible) {len(data_full_test)}. ')
        subfig.draw_line(np.arange(len(data_full_test)), data_full_test, yerr=data_sem_full_test, lc='tab:blue', lw=1,
                         eafc='#AEC7E8', eaalpha=1.0, ealw=1, eaec='#AEC7E8')
        subfig.draw_line(np.arange(len(data_full_test)), model_func(model_input, *best_popt), lc='k', lw=1.)
        subfig.draw_line([len(data_full_test)-150, len(data_full_test)], [21, 21], lc='k')
        subfig.draw_text(len(data_full_test)-75, 15, '15s')
        subfig.draw_text(0, 105, 'Model training on 20x random stimuli M or L', textlabel_ha='left')
    return best_popt, best_mse, list_of_folder_ids

def train_model_full(model_func, folder_names, stim_len_timepoints, stim_names, path_to_folders, subfig=None, train_loops=10, debug=False):
    '''
    This function trains the model on all stimuli in folder_names and all stimulus_types in stim_names.
    :param model_func: Name of the model to train, e.g. avg_mot_lumi_change
    :param folder_names: List of experiment stimuli to train on.
    :param stim_len_timepoints: List of the length of each stimulus in 0.1s steps (e.g. a 25s stimulus is written as 250 in this list).
    :param stim_names: List of stimulus types to train on (e.g. Motion, Photo, Same and Oppo).
    :param path_to_folders: path to folder containing all experiment folders.
    :param subfig: Subfigure to draw the model training fit. This is not used anymore in the paper.
    :param train_loops: Number of training-rounds.
    :param debug: If True print all parameters, the best parameter set and all training MSEs.
    :return: All parameter sets, all training MSEs, a list of the lists with folder-ids used in each round.
    '''
    # Inititalize the best_test_mse at a high-value to be beaten by the model fits. initialize lists of all values to return.
    best_test_mse = 1000
    all_params = []
    all_mse = []
    list_of_lists_of_folder_ids = []
    # Loop over training rounds and train the model.
    for t in range(train_loops):
        print(f'Training round {t}')
        popt, test_mse, folder_ids = train_model_once(model_func,
                                                      folder_names,
                                                      stim_len_timepoints,
                                                      stim_names,
                                                      path_to_folders,
                                                      subfig=subfig)
        list_of_lists_of_folder_ids = np.append(list_of_lists_of_folder_ids, folder_ids)

        all_params = np.append(all_params, popt)
        all_mse = np.append(all_mse, test_mse)
        # Update the best mse and parameter set.
        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_params = popt

    if debug:
        print(best_params)
        print(all_params)
        print(all_mse)

    return all_params, all_mse, list_of_lists_of_folder_ids

def test_model_once(model_func, params, folder_names, stim_len_timepoints, stim_names, path_to_folders, subfig_data=None, subfig_model=None, training_folder_ids=None, pick_stim_ids=None, pl_color='k', debug=False):
    '''
    This function tests the model fit once.
    This is related to most of the subpanels in Figure 2 and S2.
    :param model_func: Name of the model, e.g. avg_mot_lumi_change.
    :param params: List of model parameters to test.
    :param folder_names: List of experiment stimuli to test.
    :param stim_len_timepoints: List with the length of each stimulus in 0.1s (a stimulus of 25 seconds should be written as 250 in this list).
    :param stim_names: List of stimulus-types to test, e.g. Motion, Photo, Same, Oppo.
    :param path_to_folders: Path to folder containing all experiment folders.
    :param subfig_data: Subfigure to plot the data in.
    :param subfig_model: Subfigure to plot the model in.
    :param training_folder_ids: Default None. If specified it contains a list of integers indicating the experiment stimuli to test (used in Fig. 2g to match the drawn cartoons).
    :param pick_stim_ids: Default None. If specified it contains a list of integers indicating the stimulus types to test (used in Fig. 2g to match the drawn cartoons).
    :param pl_color: Color to use for the model plot.
    :param debug: if True print the test_mse for each fit.
    :return: test_mse (mean-squared-error between the model and the data).
    '''
    # Loop over all stimuli and stimulus-types to test (this can be randomly drawn, or following the list if training_folder_ids and/or pick_stim_ids are not None.
    list_of_stim_ids = []
    for i in range(len(stim_len_timepoints) * len(stim_names)):
        if training_folder_ids is None:
            folder_id = i % len(stim_len_timepoints)
        else:
            folder_id = int(training_folder_ids[i])

        if pick_stim_ids is None and training_folder_ids is None:
            stim_id = int(i / len(stim_len_timepoints))
        elif pick_stim_ids is None:
            stim_id = np.random.choice(np.arange(len(stim_names)))
        else:
            stim_id = int(pick_stim_ids[i])

        # Append the list of stimulus ids.
        list_of_stim_ids = np.append(list_of_stim_ids, stim_id)

        # Load the data of a specific stimulus and stimulus-type.
        _, _, data_mean, data_sem = get_data(folder_names[folder_id],
                                             stim_len_timepoints[folder_id],
                                             stim_names[stim_id],
                                             path_to_folders=path_to_folders,
                                             nsplits=3)
        # Get the model input for this stimulus and stimulus-type.
        left_mot_input, right_mot_input, left_lumi_input, right_lumi_input = get_stim_input(folder_names[folder_id],
                                                                                            stim_len_timepoints[
                                                                                                folder_id],
                                                                                            stim_names[stim_id])

        time = np.linspace(0, (len(data_mean) - 1) / 10, len(data_mean))

        #Collect the model_inputs, mean of the data, sem of the data, stimulus-ids (folder_ids) and stimulus-types.
        if i == 0:
            model_input = [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]
            data_full = data_mean
            data_sem_full = data_sem
            # The in_stim_tracker keeps track of which data should be plotted (we stick 5s pieces of flat data between the stimuli to allow any longer timescale model
            # fits to return to baseline before starting the next stim. These 5s pieces are skipped in the plotting based on in_stim_tracker.
            in_stim_tracker = np.zeros(len(data_mean))
            folder_ids = folder_id * np.ones(stim_len_timepoints[folder_id])
            stim_ids = stim_id * np.ones(stim_len_timepoints[folder_id])
        else:
            model_input = np.hstack(
                (model_input, [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]))
            data_full = np.append(data_full, data_mean)
            data_sem_full = np.append(data_sem_full, data_sem)
            in_stim_tracker = np.append(in_stim_tracker, np.zeros(len(data_mean)))
            folder_ids = np.append(folder_ids, folder_id * np.ones(stim_len_timepoints[folder_id]))
            stim_ids = np.append(stim_ids, stim_id * np.ones(stim_len_timepoints[folder_id]))

        # Here we attach a 5s piece between each stimulus to allow all models to return back to baseline before the next stimulus starts.
        if i < len(stim_len_timepoints) * len(stim_names) - 1:
            model_input = np.hstack((model_input, [np.zeros(50), np.zeros(50), np.zeros(50), np.ones(50), np.ones(50)]))
            data_full = np.append(data_full, 50 * np.ones(50))
            data_sem_full = np.append(data_sem_full, 0 * np.ones(50))
            in_stim_tracker = np.append(in_stim_tracker, np.ones(50))
            folder_ids = np.append(folder_ids, -1 * np.ones(50))
            stim_ids = np.append(stim_ids, -1 * np.ones(50))

    # Calculate the entire Mean-squared-error between the data and the model.
    test_mse = np.nanmean(np.square(model_func(model_input, *params) - data_full))
    if debug:
        print(test_mse)

    # Plot the data if subfig_data is not None.
    if subfig_data is not None:
        print('length of data', len(data_full))
        data_full[in_stim_tracker.astype(bool)] = np.nan
        data_sem_full[in_stim_tracker.astype(bool)] = np.nan
        subfig_data.draw_line(np.arange(len(data_full)), data_full, yerr=data_sem_full, lc='k', lw=1,
                         eafc='#404040', eaalpha=1.0, ealw=1, eaec='#404040')
        subfig_data.draw_line([len(data_full)-150, len(data_full)], [21, 21], lc='k')
        subfig_data.draw_text(len(data_full)-75, 15, '15s')

    # Plot the model fit if subfig_model is not None.
    if subfig_model is not None:
        model_output = model_func(model_input, *params)
        model_output[in_stim_tracker.astype(bool)] = np.nan
        subfig_model.draw_line(np.arange(len(data_full)), model_output, lc=pl_color, lw=1.)

    return test_mse


def plot_model_contribution_single_experiment(model_func, params, folder_name, stim_len_timepoints, stim_name, path_to_folders, subfig_contributions_left=None, subfig_contributions_right=None):
    '''
    This function plots the contributions of the main 3 nodes to the model output.
    This is related to figure S2c.
    :param model_func: Name of the model function to use, e.g. avg_mot_lumi_change.
    :param params: Model parameters, in case of the avg_mot_lumi_change model these are the four weigths and three timeconstants.
    :param folder_name: Folder name of the experiment to plot.
    :param stim_len_timepoints: Length of the stimulus in 0.1s steps (a 25s stimulus is written as 250 here).
    :param stim_name: The name of the stimulus type (e.g. Motion, Photo, Same, Oppo)
    :param path_to_folders: Path to folder containing all experiment folders.
    :param subfig_contributions_left: subfigure to plot the contributions to the leftward multifeature integrator.
    :param subfig_contributions_right: Subfigure to plot the contributions to the rightward multifeature integrator (with flipped y-axis).
    '''

    # Load the data of one specific experiment (folder_name) and one specific stimulus type (Motion, Photo, Same or Oppo).
    _, _, data_mean, data_sem = get_data(folder_name,
                                         stim_len_timepoints,
                                         stim_name,
                                         path_to_folders=path_to_folders,
                                         nsplits=3)

    # Load the input to the model for the same folder_name and stimulus type (stim_name).
    left_mot_input, right_mot_input, left_lumi_input, right_lumi_input = get_stim_input(folder_name,
                                                                                        stim_len_timepoints,
                                                                                        stim_name)

    time = np.linspace(0, (len(data_mean) - 1) / 10, len(data_mean))

    model_input = [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]

    # Plot the contributions of the left/right motion integrators, left/right luminance integrators and left/right luminance change detectors.
    if subfig_contributions_left is not None or subfig_contributions_right is not None:
        mot_left, mot_right, ph_left, ph_right, diff_left, diff_right, drive_left, drive_right, model_output = model_func(model_input, *params)

        subfig_contributions_left.draw_line(np.arange(len(mot_left)), mot_left, lc='#359B73', lw=1.)
        # We flip the y-axis of the rightward contributions for better visualization
        subfig_contributions_right.draw_line(np.arange(len(mot_right)), -mot_right, lc='#359B73', lw=1.)
        subfig_contributions_left.draw_line(np.arange(len(ph_left)), ph_left, lc='#E69F00', lw=1.)
        subfig_contributions_right.draw_line(np.arange(len(ph_right)), -ph_right, lc='#E69F00', lw=1.)
        # Note that the leftward change detectors feeds into the right multifeature integrator and is therefore flipped.
        subfig_contributions_right.draw_line(np.arange(len(diff_left)), -diff_left, lc='#D55E00', lw=1.)
        subfig_contributions_left.draw_line(np.arange(len(diff_right)), diff_right, lc='#D55E00', lw=1.)
    return


def sub_plot_temp_dyn_data(path_to_analysed_data, subfigs):
    '''
    This function plots the behavior curves over time.
    This is related to figure 2a-d
    :param path_to_analysed_data: the path to the combined dataframe for figure 2a-d. Each row in this dataframe contains the percentage left within a rolling window.
    :param subfigs: list of subfigures to plot the decision curves over time.
    '''

    # Load the data
    analysed_df = pd.read_hdf(path_to_analysed_data)

    # Looping over the four stimulus-types (only-motion, only-luminance, congruent, and conflicting).
    stim_names = ['Mot', 'Lumi', 'Same', 'Oppo']
    plot_names = ['only-motion (M)', 'only-luminance (L)', 'congruent (M=L)', 'conflicting (M\u2260L)']
    for st, pl_name, plot in zip(stim_names, plot_names, subfigs):
        # Select and then plot the data of the current stimulus with white background during pre- and post stim.
        stim = f'{st} W'
        stim_df = analysed_df.xs(stim, level='stimulus_name')
        n_fish = len(stim_df.index.unique('experiment_ID'))
        grouped_df = stim_df.groupby('window_time')

        mean = grouped_df['percentage_left'].mean()[0.5:]
        std = grouped_df['percentage_left'].std()[0.5:]
        sem = std / np.sqrt(n_fish)
        binned_time = mean.index.unique('window_time')

        plot.draw_line(x=binned_time, y=mean, yerr=sem, lc='#676767', eafc='#989898', eaalpha=1.0, lw=1, ealw=1, eaec='#989898')

        # Select and then plot the data of the current stimulus with black background during pre- and post stim.
        stim = f'{st} B'
        stim_df = analysed_df.xs(stim, level='stimulus_name')
        n_fish = len(stim_df.index.unique('experiment_ID'))
        grouped_df = stim_df.groupby('window_time')

        mean = grouped_df['percentage_left'].mean()[0.5:]
        std = grouped_df['percentage_left'].std()[0.5:]
        sem = std / np.sqrt(n_fish)
        binned_time = mean.index.unique('window_time')

        plot.draw_line(x=binned_time, y=mean, yerr=sem, lc='k', eafc='#404040', alpha=0.5, eaalpha=0.5, lw=1, ealw=1, eaec='#404040')
        plot.draw_text(12.5, 1.25, f'{pl_name}')

    subfigs[-1].draw_line([20, 25], [0.26, 0.26], lc='k')
    subfigs[-1].draw_text(22.5, 0.20, '5s')
    return

def sub_plot_modelling_example_traces(plot_model_ADD_example, plot_model_ADD_only_example, plot_data_example, path_to_folders):
    '''
    This function plots the example traces of the multifeature and unifeature fitting strategies, as well as the behavioral data.
    This is related to figure 2g.
    :param plot_model_ADD_example: Subfigure to plot the multifeature fitted model.
    :param plot_model_ADD_only_example: Subfigure to plot the unifeature fiteed model.
    :param plot_data_example: Subfigure to plot the fish data.
    :param path_to_folders: path to folder containing all experiment folders.
    '''

    # We define the multifeature training stimuli.
    folder_names_half = ['converted_phototaxis_dotmotion_integration_simultaneous_white', # Simultaneous
                        'phototaxis_dotmotion_white_Sep', # White
                        'phototaxis_dotmotion_simultaneous_low_Sep', # Simultaneous low
                         'converted_phototaxis_dotmotion_integration_peppersalt', # peppersalt
                        'converted_phototaxis_dotmotion_integration_long']  # long
    stim_len_timepoints_half = np.array([250, 300, 450, 350, 600])
    stim_names_half = ['Motion', 'Photo', 'Same', 'Oppo']

    # We define the unifeature training stimuli.
    folder_names_onlyML = ['converted_phototaxis_dotmotion_integration_simultaneous_white',  # Simultaneous
                        'phototaxis_dotmotion_simultaneous_low_Sep',  # Simultaneous low
                        'converted_phototaxis_dotmotion_integration_peppersalt',  # Peppersalt
                        'phototaxis_dotmotion_reverse_Sep',  # Reverse
                        'phototaxis_dotmotion_white_Sep',  # White
                        'converted_phototaxis_dotmotion_integration_long',  # Long
                        'beh_simultaneous_blackwhite_black',  # Simultaneous black
                        'beh_simultaneous_blackwhite_white',  # Simultaneous white
                        'phototaxis_dotmotion_integration_halfoverlap',  # Halfoverlap
                        'beh_simultaneous_10only']  # Simultaneous verylow
    stim_len_timepoints_onlyML = np.array([250, 450, 350, 300, 300, 600, 250, 250, 350, 250])
    stim_names_onlyML = ['Motion', 'Photo',]

    # We define the common testing stimuli to make sure neither model is trained on this data.
    folder_names_test = ['beh_simultaneous_blackwhite_white',  # simultaneous white
                        'phototaxis_dotmotion_reverse_Sep',  # reverse
                         'beh_simultaneous_blackwhite_black',  # simultaneous black
                        'phototaxis_dotmotion_integration_halfoverlap',  # Halfoverlap
                        'beh_simultaneous_10only']  #simultaneous very low
    stim_len_timepoints_test = np.array([250, 300, 250, 350, 250])
    stim_names_test = ['Same', 'Oppo']

    # We hardcoded the random stimulus order to align with the stimulus cartoons.
    testing_folder_ids = np.array([0, 1, 2, 3, 4, 3, 1, 0, 4, 2])
    testing_stim_ids = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 1])

    total_time = 450 + np.sum(stim_len_timepoints_test[testing_folder_ids.astype(int)])
    print('total time example traces is:, ', total_time)

    # Training the multifeature fitted model
    model_params, model_train_mse, _ = train_model_full(model_func=avg_mot_lumi_change,
                                                        folder_names=folder_names_half,
                                                        stim_len_timepoints=stim_len_timepoints_half,
                                                        stim_names=stim_names_half,
                                                        path_to_folders=path_to_folders,
                                                        subfig=None,
                                                        train_loops=1)

    model_params = np.array(model_params).reshape(-1, 7)

    # Testing the multifeature fitted model; Note that in this case there is only 1 set of parameters, because we trained n=1 train_loops.
    model_test_mse = []
    for params in model_params:
        model_test_mse = np.append(model_test_mse, test_model_once(model_func=avg_mot_lumi_change,
                                                                   params=params,
                                                                   folder_names=folder_names_test,
                                                                   stim_len_timepoints=stim_len_timepoints_test,
                                                                   stim_names=stim_names_test,
                                                                   path_to_folders=path_to_folders,
                                                                   subfig_data=None,
                                                                   subfig_model=plot_model_ADD_example,
                                                                   training_folder_ids=testing_folder_ids,
                                                                   pick_stim_ids=testing_stim_ids,
                                                                   pl_color='#808080'))

    # Training the unifeature fitted model
    model_params, model_train_mse, _ = train_model_full(model_func=avg_mot_lumi_change,
                                                        folder_names=folder_names_onlyML,
                                                        stim_len_timepoints=stim_len_timepoints_onlyML,
                                                        stim_names=stim_names_onlyML,
                                                        path_to_folders=path_to_folders,
                                                        subfig=None,
                                                        train_loops=1)

    model_params = np.array(model_params).reshape(-1, 7)

    # Testing the unifeature fitted model; Note that in this case there is only 1 set of parameters, because we trained n=1 train_loops.
    # The fish data will also be plotted in this, since subfig_data in the test_model_once function is not None.
    model_test_mse = []
    for params in model_params:
        model_test_mse = np.append(model_test_mse, test_model_once(model_func=avg_mot_lumi_change,
                                                                   params=params,
                                                                   folder_names=folder_names_test,
                                                                   stim_len_timepoints=stim_len_timepoints_test,
                                                                   stim_names=stim_names_test,
                                                                   path_to_folders=path_to_folders,
                                                                   subfig_data=plot_data_example,
                                                                   subfig_model=plot_model_ADD_only_example,
                                                                   training_folder_ids=testing_folder_ids,
                                                                   pick_stim_ids=testing_stim_ids,
                                                                   pl_color='cyan'))

    return

def sub_plot_example_traces_model_fit(path_to_analysed_data, subfiga, subfigb, subfigcs, subfigds, subfiges, subfigfs, path_to_folders):
    '''
    This figure plots the data and model fit of the decision-curves over time, as well as the node contributions.
    This is related to Fig. S2b,c.
    :param path_to_analysed_data: the path to the combined dataframe for figure 2a-d. Each row in this dataframe contains the percentage left within a rolling window.
    :param subfiga: Subfigure containing the decision curves over time for stimuli with a white-background pre- and post-stimulus.
    :param subfigb: Subfigure containing the decision curves over time for stimuli with a black-background pre- and post-stimulus.
    :param subfigcs: List of subfigures containing the leftward node contributions to stimuli with a white-background pre- and post-stimulus.
    :param subfigds: List of subfigures containing the rightward node contributions to stimuli with a white-background pre- and post-stimulus.
    :param subfiges: List of subfigures containing the leftward node contributions to stimuli with a black-background pre- and post-stimulus.
    :param subfigfs: List of subfigures containing the rightward node contributions to stimuli with a black-background pre- and post-stimulus.
    :param path_to_folders: path to folder containing all experiment folders.
    '''

    # These model parameters are printed by sub_plot_modelling_overview_mse_tau_w, to save time in creating this figure we hard-coded them here.
    model_params = [8, 52, 13, 11, 3.9, 0.9, 4]

    # Add a white line to highlight 50%, or the decision-baseline.
    subfiga.draw_line([0, 1150], [50, 50], lc='w', lw=1.5)
    subfigb.draw_line([0, 1150], [50, 50], lc='w', lw=1.5)

    # Load the data.
    analysed_df = pd.read_hdf(path_to_analysed_data)

    # The stimulus names relate to only-motion, only-luminance, congruent, conflicting.
    stim_names = ['Mot', 'Lumi', 'Same', 'Oppo']

    # The x-starts are used to sequentially plot all stimuli within the same subfigure.
    x_starts = [0, 300, 600, 900, 1200, 1500, 1800, 2100]
    for st, x_start in zip(stim_names, x_starts):
        # The W indicates the white background pre- and post stimulus.
        stim = f'{st} W'
        stim_df = analysed_df.xs(stim, level='stimulus_name')
        n_fish = len(stim_df.index.unique('experiment_ID'))
        grouped_df = stim_df.groupby('window_time')

        # We extract the mean and sem decision curves. We ignore the first 0.5 second of data, since there the windows are not complete and contain less data causing the percentages to fluctuate.
        mean = grouped_df['percentage_left'].mean()[0.5:]
        std = grouped_df['percentage_left'].std()[0.5:]
        sem = std / np.sqrt(n_fish)
        binned_time = mean.index.unique('window_time') * 10 + x_start

        subfiga.draw_line(x=binned_time, y=mean*100, yerr=sem*100, lc='#676767', eafc='#989898', eaalpha=1.0, lw=1, ealw=1, eaec='#989898')

        # The B the black background pre- and post stimulus.
        stim = f'{st} B'
        stim_df = analysed_df.xs(stim, level='stimulus_name')
        n_fish = len(stim_df.index.unique('experiment_ID'))
        grouped_df = stim_df.groupby('window_time')

        # We extract the mean and sem decision curves. We ignore the first 0.5 second of data, since there the windows are not complete and contain less data causing the percentages to fluctuate.
        mean = grouped_df['percentage_left'].mean()[0.5:]
        std = grouped_df['percentage_left'].std()[0.5:]
        sem = std / np.sqrt(n_fish)
        binned_time = mean.index.unique('window_time') * 10 + x_start

        subfigb.draw_line(x=binned_time, y=mean*100, yerr=sem*100, lc='k', eafc='#404040', alpha=0.5, eaalpha=0.5, lw=1, ealw=1, eaec='#404040')

    # Add the model fit to the plot with white background
    folder_names_test = ['beh_simultaneous_blackwhite_white', ]
    stim_len_timepoints_test = np.array([250])
    stim_names_test = ['Motion', 'Photo', 'Same', 'Oppo']

    testing_folder_ids = np.array([0, 0, 0, 0])
    testing_stim_ids = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    test_model_once(model_func=avg_mot_lumi_change,
                    params=model_params,
                    folder_names=folder_names_test,
                    stim_len_timepoints=stim_len_timepoints_test,
                    stim_names=stim_names_test,
                    path_to_folders=path_to_folders,
                    subfig_data=None,
                    subfig_model=subfiga,
                    training_folder_ids=testing_folder_ids,
                    pick_stim_ids=testing_stim_ids,
                    pl_color='cyan')

    # Plot the model contributions of all three main pathways - motion, luminance level, change in luminance- to the white background traces.
    for i in range(len(stim_names_test)):
        plot_model_contribution_single_experiment(model_func=avg_mot_lumi_change_full_node_outputs,
                                                  params=model_params,
                                                  folder_name=folder_names_test[0],
                                                  stim_len_timepoints=stim_len_timepoints_test[0],
                                                  stim_name=stim_names_test[i],
                                                  path_to_folders=path_to_folders,
                                                  subfig_contributions_left=subfigcs[i],
                                                  subfig_contributions_right=subfigds[i])

    ## Add the model fit to the plot with black background
    folder_names_test = ['beh_simultaneous_blackwhite_black',  # simultaneous black
                         ]
    stim_len_timepoints_test = np.array([250])
    stim_names_test = ['Motion', 'Photo', 'Same', 'Oppo']

    testing_folder_ids = np.array([0, 0, 0, 0])
    testing_stim_ids = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    test_model_once(model_func=avg_mot_lumi_change,
                    params=model_params,
                    folder_names=folder_names_test,
                    stim_len_timepoints=stim_len_timepoints_test,
                    stim_names=stim_names_test,
                    path_to_folders=path_to_folders,
                    subfig_data=None,
                    subfig_model=subfigb,
                    training_folder_ids=testing_folder_ids,
                    pick_stim_ids=testing_stim_ids,
                    pl_color='cyan')

    # Plot the model contributions of all three main pathways - motion, luminance level, change in luminance- to the black background traces.
    for i in range(len(stim_names_test)):
        plot_model_contribution_single_experiment(model_func=avg_mot_lumi_change_full_node_outputs,
                                                  params=model_params,
                                                  folder_name=folder_names_test[0],
                                                  stim_len_timepoints=stim_len_timepoints_test[0],
                                                  stim_name=stim_names_test[i],
                                                  path_to_folders=path_to_folders,
                                                  subfig_contributions_left=subfiges[i],
                                                  subfig_contributions_right=subfigfs[i])

    subfigb.draw_line([1100, 1150], [22, 22], lc='k')
    subfigb.draw_text(1125, 15, '5s')
    subfigfs[-1].draw_line([200, 250], [-4, -4], lc='k')
    subfigfs[-1].draw_text(225, -5, '5s')
    return

def sub_plot_modelling_overview_mse_tau_w(n_training_rounds, subfig_mse_btm, subfig_mse_top, subfig_mse_sup, subfig_tau, subfig_w, path_to_folders):
    '''
    This function plots the full overview of the model performance across multifeature and unifeature training/testing strategies, the fitted timeconstants and the fitted weights.
    This is related to Fig. 2h,i,j and Fig. S2d.
    :param n_training_rounds: Integer number of training iterations.
    :param subfig_mse_btm: Subfigure of the mean-squared-error of all model and training alterantives. Up to MSE=75 (the y-axis is split to accommodate for the very high error of the silenced motion-model).
    :param subfig_mse_top: Subfigure of the mean-squared-error of all model and training alterantives. Up from MSE=75 (the y-axis is split to accommodate for the very high error of the silenced motion-model).
    :param subfig_mse_sup: Subfigure of the mean-squared error of alternative models in the supplemental information.
    :param subfig_tau: Subfigure of the fitted timeconstant distributions
    :param subfig_w: Subfigure of the fitted weight distributions.
    :param path_to_folders: path to folder containing all experiment folders.
    :return:
    '''

    # We define the multifeature training stimuli.
    folder_names_half = ['converted_phototaxis_dotmotion_integration_simultaneous_white',  # Simultaneous
                        'phototaxis_dotmotion_white_Sep',  # White
                        'phototaxis_dotmotion_simultaneous_low_Sep',  # simul low
                         'converted_phototaxis_dotmotion_integration_peppersalt',  # peppersalt
                        'converted_phototaxis_dotmotion_integration_long']   # long
    stim_len_timepoints_half = np.array([250, 300, 450, 350, 600])
    stim_names_half = ['Motion', 'Photo', 'Same', 'Oppo']

    # We define the unifeature training stimuli.
    folder_names_onlyML = ['converted_phototaxis_dotmotion_integration_simultaneous_white',  # Simultaneous
                           'phototaxis_dotmotion_simultaneous_low_Sep',  # Simultaneous low
                           'converted_phototaxis_dotmotion_integration_peppersalt',  # Peppersalt
                           'phototaxis_dotmotion_reverse_Sep',  # Reverse
                           'phototaxis_dotmotion_white_Sep',  # White
                           'converted_phototaxis_dotmotion_integration_long',  # Long
                           'beh_simultaneous_blackwhite_black',  # Simultaneous black
                           'beh_simultaneous_blackwhite_white',  # Simultaneous white
                           'phototaxis_dotmotion_integration_halfoverlap',  # Halfoverlap
                           'beh_simultaneous_10only']  # Simultaneous verylow
    stim_len_timepoints_onlyML = np.array([250, 450, 350, 300, 300, 600, 250, 250, 350, 250])
    stim_names_onlyML = ['Motion', 'Photo', ]

    # We define the multifeature testing stimuli.
    folder_names_half_test = ['beh_simultaneous_blackwhite_white',  # simultaneous white
                        'phototaxis_dotmotion_reverse_Sep',  # Reverse
                         'beh_simultaneous_blackwhite_black',  # simultaneous black
                        'phototaxis_dotmotion_integration_halfoverlap',  # Halfoverlap
                        'beh_simultaneous_10only']  #simulataneous very low
    stim_len_timepoints_half_test = np.array([250, 300, 250, 350, 250])
    stim_names_half_test = ['Motion', 'Photo', 'Same', 'Oppo']

    # We define the unifeature testing stimuli.
    folder_names_onlyML_test = ['converted_phototaxis_dotmotion_integration_simultaneous_white',  # Simultaneous
                           'phototaxis_dotmotion_simultaneous_low_Sep',  # Simultaneous low
                           'converted_phototaxis_dotmotion_integration_peppersalt',  # Peppersalt
                           'phototaxis_dotmotion_reverse_Sep',  # Reverse
                           'phototaxis_dotmotion_white_Sep',  # White
                           'converted_phototaxis_dotmotion_integration_long',  # Long
                           'beh_simultaneous_blackwhite_black',  # Simultaneous black
                           'beh_simultaneous_blackwhite_white',  # Simultaneous white
                           'phototaxis_dotmotion_integration_halfoverlap',  # Halfoverlap
                           'beh_simultaneous_10only']  # Simultaneous verylow
    stim_len_timepoints_onlyML_test = np.array([250, 450, 350, 300, 300, 600, 250, 250, 350, 250])
    stim_names_onlyML_test = ['Same', 'Oppo', ]

    # Prepare lists to store the model performance of 8 models. For in the main figure: 2x original model with multifeature and unifeature training; 3x one of the pathways silenced. For in the supplemental figure: original, no multifeature integration, mutual inhibtion between motion integrators.
    model_test_mses = [[]] * 8

    print(f'Running model avg_mot_lumi_change multifeature training')
    # Training the original model using multifeature training/testing
    model_params, model_train_mse, all_folder_ids = train_model_full(model_func=avg_mot_lumi_change,
                                                                     folder_names=folder_names_half,
                                                                     stim_len_timepoints=stim_len_timepoints_half,
                                                                     stim_names=stim_names_half,
                                                                     path_to_folders=path_to_folders,
                                                                     subfig=None,
                                                                     train_loops=n_training_rounds)
    # Re-arranging the 7 fitted parameters.
    model_params = np.array(model_params).reshape(-1, 7)
    for param in range(7):
        print('Final param means: ')
        print(f'param {param}: {model_params[:, param]}')

    # Testing the performance of each of the n_trainings_round fits.
    all_folder_ids = np.array(all_folder_ids).reshape(-1, 20)
    for params, folder_ids in zip(model_params, all_folder_ids):
        model_test_mses[0] = np.append(model_test_mses[0],
                                       test_model_once(model_func=avg_mot_lumi_change, params=params,
                                                       folder_names=folder_names_half_test,
                                                       stim_len_timepoints=stim_len_timepoints_half_test,
                                                       stim_names=stim_names_half_test,
                                                       path_to_folders=path_to_folders,
                                                       subfig_data=None, subfig_model=None,
                                                       training_folder_ids=folder_ids))

    # Plotting the model performance of the original model (note the mse-subfigure has a split y-axes at 75).
    btm_idx = np.array(model_test_mses[0]) < 75
    top_idx = np.array(model_test_mses[0]) >= 75
    if np.sum(btm_idx) > 0:
        subfig_mse_btm.draw_scatter(np.zeros(len(model_test_mses[0]))[btm_idx] + np.random.uniform(-0.4, 0.4, len(model_test_mses[0]))[btm_idx], model_test_mses[0][btm_idx], pc='cadetblue',  ec='cadetblue')
    if np.sum(top_idx) > 0:
        subfig_mse_top.draw_scatter(np.zeros(len(model_test_mses[0]))[top_idx] + np.random.uniform(-0.4, 0.4, len(model_test_mses[0]))[top_idx], model_test_mses[0][top_idx], pc='cadetblue',  ec='cadetblue')
    if np.nanmedian(model_test_mses[0]) < 75:
        subfig_mse_btm.draw_line([-0.4, 0.4], [np.nanmedian(model_test_mses[0]), np.nanmedian(model_test_mses[0])], lc='k', lw='1')
    else:
        subfig_mse_top.draw_line([-0.4, 0.4], [np.nanmedian(model_test_mses[0]), np.nanmedian(model_test_mses[0])], lc='k', lw='1')

    # Plotting the model parameters and the median value (split into 4 timeconstants and 3 weights).
    for param in range(7):
        if param < 4:
            subfig_tau.draw_scatter(2*param * np.ones(len(model_params[:, param])) + np.random.uniform(-0.4, 0.4, len(model_params[:, param])), model_params[:, param], pc='cadetblue', ec='cadetblue')
            subfig_tau.draw_line([2*param - 0.4, 2*param + 0.4], [np.nanmedian(model_params[:, param]), np.nanmedian(model_params[:, param])], lc='k', lw='1')
        else:
            subfig_w.draw_scatter((2*param - 8) * np.ones(len(model_params[:, param])) + np.random.uniform(-0.4, 0.4, len(model_params[:, param])), model_params[:, param], pc='cadetblue', ec='cadetblue')
            subfig_w.draw_line([2*param - 8 - 0.4, 2*param - 8 + 0.4], [np.nanmedian(model_params[:, param]), np.nanmedian(model_params[:, param])], lc='k', lw='1')

    # Loop over all main-figure alternative models to train them using unifeature training/testing and plot their performance, and only for the original model, parameters.
    for model_counter, (modelfunction, pl_color, param_size) in enumerate(zip(
            [avg_mot_lumi_change, avg_mot_lumi, avg_mot_change, avg_lumi_change],
            ['cyan', '#D55E00', '#E69F00', '#359B73'],
            [7, 5, 5, 5])):

        print(f'Running model {model_counter} unifeature training')
        # Training the model using unifeature training/testing
        model_params, model_train_mse, all_folder_ids = train_model_full(model_func=modelfunction,
                                                                         folder_names=folder_names_onlyML,
                                                                         stim_len_timepoints=stim_len_timepoints_onlyML,
                                                                         stim_names=stim_names_onlyML,
                                                                         path_to_folders=path_to_folders,
                                                                         subfig=None,
                                                                         train_loops=n_training_rounds)
        # Re-arranging the 5 or 7 fitted parameters.
        model_params = np.array(model_params).reshape(-1, param_size)
        for param in range(param_size):
            print('Final param means: ')
            print(f'param {param}: {model_params[:, param]}')

        # Testing the performance of each of the n_trainings_round fits.
        all_folder_ids = np.array(all_folder_ids).reshape(-1, 20)
        for params, folder_ids in zip(model_params, all_folder_ids):
            model_test_mses[model_counter+1] = np.append(model_test_mses[model_counter+1],
                                                         test_model_once(model_func=modelfunction, params=params,
                                                                         folder_names=folder_names_onlyML_test,
                                                                         stim_len_timepoints=stim_len_timepoints_onlyML_test,
                                                                         stim_names=stim_names_onlyML_test,
                                                                         path_to_folders=path_to_folders,
                                                                         subfig_data=None, subfig_model=None,
                                                                         training_folder_ids=folder_ids))

        # Plotting the model performance of the model (note the mse-subfigure has a split y-axes at 75).
        btm_idx = np.array(model_test_mses[model_counter+1]) < 75
        top_idx = np.array(model_test_mses[model_counter+1]) >= 75
        if np.sum(btm_idx) > 0:
            subfig_mse_btm.draw_scatter((model_counter + 1) * np.ones(len(model_test_mses[model_counter+1]))[btm_idx] + np.random.uniform(-0.4, 0.4, len(model_test_mses[model_counter+1]))[btm_idx],
                                        model_test_mses[model_counter+1][btm_idx], pc=pl_color, ec=pl_color)
        if np.sum(top_idx) > 0:
            subfig_mse_top.draw_scatter((model_counter + 1) * np.ones(len(model_test_mses[model_counter+1]))[top_idx] + np.random.uniform(-0.4, 0.4, len(model_test_mses[model_counter+1]))[top_idx],
                                        model_test_mses[model_counter+1][top_idx], pc=pl_color, ec=pl_color)
        if np.nanmedian(model_test_mses[model_counter+1]) < 75:
            subfig_mse_btm.draw_line([model_counter + 0.6, model_counter + 1.4], [np.nanmedian(model_test_mses[model_counter+1]), np.nanmedian(model_test_mses[model_counter+1])], lc='k', lw='1')
        else:
            subfig_mse_top.draw_line([model_counter + 0.6, model_counter + 1.4], [np.nanmedian(model_test_mses[model_counter+1]), np.nanmedian(model_test_mses[model_counter+1])], lc='k', lw='1')

        # Only for the original model: Plotting the model parameters and the median value (split into 4 timeconstants and 3 weights).
        if model_counter == 0:
            for param in range(param_size):
                if param < 4:
                    subfig_tau.draw_scatter((2*param+1) * np.ones(len(model_params[:, param])) + np.random.uniform(-0.4, 0.4, len(model_params[:, param])), model_params[:, param], pc=pl_color, ec=pl_color)
                    subfig_tau.draw_line([2*param+1-0.4, 2*param+1+0.4], [np.nanmedian(model_params[:, param]), np.nanmedian(model_params[:, param])], lc='k', lw='1')
                    print(f'Param {param}: {np.nanmedian(model_params[:, param])}')
                else:
                    subfig_w.draw_scatter((2*param - 7) * np.ones(len(model_params[:, param])) + np.random.uniform(-0.4, 0.4, len(model_params[:, param])), model_params[:, param], pc=pl_color, ec=pl_color)
                    subfig_w.draw_line([2*param-7-0.4, 2*param-7+0.4], [np.nanmedian(model_params[:, param]), np.nanmedian(model_params[:, param])], lc='k', lw='1')
                    print(f'Param {param}: {np.nanmedian(model_params[:, param])}')


    # Loop over all supplemental-figure alternative models to train them using unifeature training/testing and plot their performance.
    for model_counter, (modelfunction, pl_color, param_size) in enumerate(zip(
            [avg_mot_lumi_change, avg_mot_lumi_change_nomfint, avg_mot_lumi_change_withmotinhib],
            ['cyan', 'tab:blue', 'green'],
            [7, 6, 8])):

        print(f'Running model {model_counter} unifeature training')
        # Training the model using unifeature training/testing
        model_params, model_train_mse, all_folder_ids = train_model_full(model_func=modelfunction,
                                                                         folder_names=folder_names_onlyML,
                                                                         stim_len_timepoints=stim_len_timepoints_onlyML,
                                                                         stim_names=stim_names_onlyML,
                                                                         path_to_folders=path_to_folders,
                                                                         subfig=None,
                                                                         train_loops=n_training_rounds)
        # Re-arranging the 6, 7 or 8 fitted parameters.
        model_params = np.array(model_params).reshape(-1, param_size)
        for param in range(param_size):
            print('Final param means: ')
            print(f'param {param}: {model_params[:, param]}')

        # Testing the performance of each of the n_trainings_round fits.
        all_folder_ids = np.array(all_folder_ids).reshape(-1, 20)
        for params, folder_ids in zip(model_params, all_folder_ids):
            model_test_mses[model_counter+5] = np.append(model_test_mses[model_counter+5],
                                                         test_model_once(model_func=modelfunction, params=params,
                                                                         folder_names=folder_names_onlyML_test,
                                                                         stim_len_timepoints=stim_len_timepoints_onlyML_test,
                                                                         stim_names=stim_names_onlyML_test,
                                                                         path_to_folders=path_to_folders,
                                                                         subfig_data=None, subfig_model=None,
                                                                         training_folder_ids=folder_ids))

        # Plotting the model performance of the model.
        subfig_mse_sup.draw_scatter((model_counter + 1) * np.ones(len(model_test_mses[model_counter+5])) + np.random.uniform(-0.4, 0.4, len(model_test_mses[model_counter+5])),
                                        model_test_mses[model_counter+5], pc=pl_color, ec=pl_color)

        subfig_mse_sup.draw_line([model_counter + 0.6, model_counter + 1.4], [np.nanmedian(model_test_mses[model_counter+5]), np.nanmedian(model_test_mses[model_counter+5])], lc='k', lw='1')


    # Do the statistical comparisons between the main-figure model fits using a t-test. The pval is compared to a Bonferonni corrected threshold.
    for model1, model2, h in zip([0, 1, 1, 1],
                                 [1, 2, 3, 4],
                                 [80, 110, 140, 180]):
        subfig_mse_top.draw_line([model1, model1, model2, model2], [h-1, h, h, h-1], lc='k')
        _, pval = ttest_ind(model_test_mses[model1], model_test_mses[model2])
        if pval < 0.001/4:
            subfig_mse_top.draw_text((model1+model2)/2, h+2, '***')
        elif pval < 0.01/4:
            subfig_mse_top.draw_text((model1+model2)/2, h+2, '**')
        elif pval < 0.05/4:
            subfig_mse_top.draw_text((model1+model2)/2, h+2, '*')
        else:
            subfig_mse_top.draw_text((model1+model2)/2, h+4, 'ns')
        effect_size = cohens_d(model_test_mses[model1], model_test_mses[model2])
        print(f'model1 {model1} vs model2 {model2}, pval {pval}: Cohen D effect size {effect_size}')

    # Do the statistical comparisons between the supplemental-figure model fits using a t-test. The pval is compared to a Bonferonni corrected threshold.
    for model1, model2, h in zip([5, 5],
                                 [6, 7],
                                 [40, 60]):
        subfig_mse_sup.draw_line([model1 - 4, model1 - 4, model2 - 4, model2 - 4], [h-1, h, h, h-1], lc='k')
        _, pval = ttest_ind(model_test_mses[model1], model_test_mses[model2])
        if pval < 0.001/2:
            subfig_mse_sup.draw_text((model1-4+model2-4)/2, h+2, '***')
        elif pval < 0.01/2:
            subfig_mse_sup.draw_text((model1-4+model2-4)/2, h+2, '**')
        elif pval < 0.05/2:
            subfig_mse_sup.draw_text((model1-4+model2-4)/2, h+2, '*')
        else:
            subfig_mse_sup.draw_text((model1-4+model2-4)/2, h+4, 'ns')
        effect_size = cohens_d(model_test_mses[model1], model_test_mses[model2])
        print(f'model1 {model1} vs model2 {model2}, pval {pval}: Cohen D effect size {effect_size}')
    return


if __name__ == '__main__':
    # A general note for this file: We changed the names of the model nodes later for the paper. In this code we use moslty the old names:
    # multifeature integrator used to be drive, luminance change detector used to be diff, luminance level integrator used to be lumi, luminance increase detector used to be bright, and luminance decrease detector used to be dark. Motion integrator was always motion.
    # In the stimuli, lateral luminance cue is often referred to as Photo (from phototaxis stimulus), congruent stimuli are referred to as Same, conflicting stimuli as Oppo.

    # Provide the path to save the figures.
    fig_save_path = 'C:/users/katja/Desktop/fig_2.pdf'
    supfig_save_path = 'C:/users/katja/Desktop/fig_S2.pdf'

    # Provide the path to the figure_2 folder.
    fig_2_folder_path = r'Z:\Bahl lab member directories\Katja\paper_data\figure_2'

    # Get the paths to the combined dataframes for figure 2a-d.
    path_to_analysed = Path(fr'{fig_2_folder_path}\data_analysed.hdf5')

    # Get the path to the folder containing all experiment folders.
    path_to_folders = Path(fr'{fig_2_folder_path}')

    # Here we define the figures and subpanel outlines (e.g. the limits, ticks and labels of the axes) beloning to figure 2 and S2.
    fig = Figure(fig_width=18, fig_height=17)
    subfig = Figure(fig_width=18, fig_height=8)

    # Fig. 2a
    plot_motion = fig.create_plot(xpos=1, ypos=13.5, plot_height=2, plot_width=3.5, errorbar_area=True,
                                  xmin=0, xmax=25, yl='Left swims (%)', ymin=0.25,
                                  ymax=0.9, yticks=[0.30, 0.50, 0.70, 0.90], yticklabels=['30', '50', '70', '90'], hlines=[0.5],
                                  helper_lines_lc='w', helper_lines_dashes=(2, 0), helper_lines_lw=1, vspans=[[5, 20, 'lightgray', 1.0], ])

    # Fig. 2b
    plot_lumi = fig.create_plot(xpos=5, ypos=13.5, plot_height=2, plot_width=3.5, errorbar_area=True,
                                  xmin=0, xmax=25, ymin=0.25,
                                  ymax=0.9, yticks=[0.30, 0.50, 0.70, 0.90], yticklabels=['', '', '', ''], hlines=[0.5],
                                  helper_lines_lc='w', helper_lines_dashes=(2, 0), helper_lines_lw=1, vspans=[[5, 20, 'lightgray', 1.0], ])

    # Fig. 2c
    plot_same = fig.create_plot(xpos=9, ypos=13.5, plot_height=2, plot_width=3.5, errorbar_area=True,
                                  xmin=0, xmax=25, ymin=0.25,
                                  ymax=0.9, yticks=[0.30, 0.50, 0.70, 0.90], yticklabels=['', '', '', ''], hlines=[0.5],
                                  helper_lines_lc='w', helper_lines_dashes=(2, 0), helper_lines_lw=1, vspans=[[5, 20, 'lightgray', 1.0], ])

    # Fig. 2d
    plot_oppo = fig.create_plot(xpos=13, ypos=13.5, plot_height=2, plot_width=3.5, errorbar_area=True,
                                  xmin=0, xmax=25, ymin=0.25,
                                  ymax=0.9, yticks=[0.30, 0.50, 0.70, 0.90], yticklabels=['', '', '', ''], hlines=[0.5],
                                  helper_lines_lc='w', helper_lines_dashes=(2, 0), helper_lines_lw=1, vspans=[[5, 20, 'lightgray', 1.0], ])

    # Fig. 2g
    plot_model_ADD_example = fig.create_plot(xpos=1, ypos=9, plot_height=1.4, plot_width=7, errorbar_area=True,
                                               xmin=0, xmax=3250, ymin=20, ymax=90, yticks=[50, 75], yl='Left swims (%)')
    plot_model_ADD_only_example = fig.create_plot(xpos=1, ypos=8., plot_height=1.4, plot_width=7, errorbar_area=True,
                                               xmin=0, xmax=3250, ymin=20, ymax=90, yticks=[50, 75], yl='Left swims (%)')
    plot_model_data_example = fig.create_plot(xpos=1, ypos=6.5, plot_height=1.4, plot_width=7, errorbar_area=True,
                                               xmin=0, xmax=3250, ymin=20, ymax=90, yticks=[25, 50, 75], yl='Left swims (%)')

    # Fig. S2b
    plot_white_example = subfig.create_plot(xpos=1, ypos=2.5, plot_height=1.4, plot_width=6.5, errorbar_area=True,
                                                 xmin=0, xmax=1150, ymin=20, ymax=90, yticks=[25, 50, 75], yl='Left swims (%)',
                                                 vspans=[[50, 200, 'lightgray', 1.0], [350, 500, 'lightgray', 1.0], [650, 800, 'lightgray', 1.0], [950, 1100, 'lightgray', 1.0]])
    plot_black_example = subfig.create_plot(xpos=8.5, ypos=2.5, plot_height=1.4, plot_width=6.5, errorbar_area=True,
                                                 xmin=0, xmax=1150, ymin=20, ymax=90, yticks=[25, 50, 75],
                                                 vspans=[[50, 200, 'lightgray', 1.0], [350, 500, 'lightgray', 1.0], [650, 800, 'lightgray', 1.0], [950, 1100, 'lightgray', 1.0]])

    # Fig. S2c
    plot_contributions_examples_white_left = [[]] * 4
    plot_contributions_examples_white_right = [[]] * 4
    plot_contributions_examples_black_left = [[]] * 4
    plot_contributions_examples_black_right = [[]] * 4
    for i in range(4):
        if i == 0:
            plot_contributions_examples_white_left[i] = subfig.create_plot(xpos=1 + 1.7*i, ypos=1.3, plot_height=0.7, plot_width=1.4, errorbar_area=True,
                                                                 xmin=0, xmax=250, ymin=-0.2, ymax=4.5, yticks=[0, 2, 4],
                                                                 yticklabels=['0', '2', '4'], yl='Node activity (au)',
                                                                 vspans=[[50, 200, 'lightgray', 1.0], ])
            plot_contributions_examples_white_right[i] = subfig.create_plot(xpos=1 + 1.7*i, ypos=0.5, plot_height=0.7, plot_width=1.4, errorbar_area=True,
                                                                 xmin=0, xmax=250, ymin=-4.5, ymax=0.2, yticks=[-4, -2, 0],
                                                                 yticklabels=['4', '2', '0'],
                                                                 vspans=[[50, 200, 'lightgray', 1.0], ])
            plot_contributions_examples_black_left[i] = subfig.create_plot(xpos=8.5 + 1.7*i, ypos=1.3, plot_height=0.7, plot_width=1.4, errorbar_area=True,
                                                                 xmin=0, xmax=250, ymin=-0.2, ymax=4.5, yticks=[0, 2, 4],
                                                                 yticklabels=['0', '2', '4'],
                                                                 vspans=[[50, 200, 'lightgray', 1.0], ])
            plot_contributions_examples_black_right[i] = subfig.create_plot(xpos=8.5 + 1.7*i, ypos=0.5, plot_height=0.7, plot_width=1.4, errorbar_area=True,
                                                                 xmin=0, xmax=250, ymin=-4.5, ymax=0.2, yticks=[-4, -2, 0],
                                                                 yticklabels=['4', '2', '0'],
                                                                 vspans=[[50, 200, 'lightgray', 1.0], ])
        else:
            plot_contributions_examples_white_left[i] = subfig.create_plot(xpos=1 + 1.7*i, ypos=1.3, plot_height=0.7, plot_width=1.4, errorbar_area=True,
                                                                 xmin=0, xmax=250, ymin=-0.2, ymax=4.5,
                                                                 vspans=[[50, 200, 'lightgray', 1.0], ])
            plot_contributions_examples_white_right[i] = subfig.create_plot(xpos=1 + 1.7*i, ypos=0.5, plot_height=0.7, plot_width=1.4, errorbar_area=True,
                                                                 xmin=0, xmax=250, ymin=-4.5, ymax=0.2,
                                                                 vspans=[[50, 200, 'lightgray', 1.0], ])
            plot_contributions_examples_black_left[i] = subfig.create_plot(xpos=8.5 + 1.7*i, ypos=1.3, plot_height=0.7, plot_width=1.4, errorbar_area=True,
                                                                 xmin=0, xmax=250, ymin=-0.2, ymax=4.5,
                                                                 vspans=[[50, 200, 'lightgray', 1.0], ])
            plot_contributions_examples_black_right[i] = subfig.create_plot(xpos=8.5 + 1.7*i, ypos=0.5, plot_height=0.7, plot_width=1.4, errorbar_area=True,
                                                                 xmin=0, xmax=250, ymin=-4.5, ymax=0.2,
                                                                 vspans=[[50, 200, 'lightgray', 1.0], ])

    # Fig. 2h
    plot_mse_overview_bottom = fig.create_plot(xpos=9, ypos=7.5, plot_height=1.25, plot_width=2.25,
                                        xmin=-1, xmax=5, ymin=0, ymax=75, yticks=[10, 20, 30, 40, 50], yl='MSE',
                                        xticks=[0, 1, 2, 3, 4], xticklabels=['multifeature\ntesting', 'unifeature\ntesting', 'X lumi change',
                                                                             'X lumi level', 'X motion'], xticklabels_rotation=90)
    plot_mse_overview_top = fig.create_plot(xpos=9, ypos=8.75, plot_height=1.25, plot_width=2.25,
                                        xmin=-1, xmax=5, ymin=75, ymax=210, yticks=[100, 150, 200])

    # Fig. S2d
    plot_mse_overview_sup = subfig.create_plot(xpos=16, ypos=2.5, plot_height=1.6, plot_width=2,
                                        xmin=0, xmax=4, ymin=0, ymax=65, yticks=[10, 20, 30, 40, 50], yl='MSE',
                                        xticks=[1, 2, 3], xticklabels=['unifeature\ntesting', 'X MF int', '+ Mot inhi'], xticklabels_rotation=90)

    # Fig. 2i
    plot_timeconstants_overview = fig.create_plot(xpos=12, ypos=7.5, plot_height=2.5, plot_width=2.25,
                                                  xmin=-1, xmax=8, ymin=0, ymax=31, yticks=[0, 5, 10, 15, 20, 25, 30],
                                                  yticklabels=['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'], yl='Time constant (s)',
                                                  xticks=[0.5, 2.5, 4.5, 6.5], xticklabels=['\u03C4 motion', '\u03C4 lumi level', '\u03C4 lumi change', '\u03C4 multifeature'],
                                                  xticklabels_rotation=90)
    # Fig. 2j
    plot_weights_overview = fig.create_plot(xpos=15, ypos=7.5, plot_height=2.5, plot_width=2.25,
                                            xmin=-1, xmax=6, ymin=0, ymax=3.5, yticks=[0, 1, 2, 3], yl='Weight',
                                            xticks=[0.5, 2.5, 4.5], xticklabels=['w motion', 'w lumi level', 'w lumi change'],
                                            xticklabels_rotation=90)

    print('Plot supplemental figure example data with model fit and node contributions.')
    sub_plot_example_traces_model_fit(path_to_analysed, plot_white_example, plot_black_example, plot_contributions_examples_white_left, plot_contributions_examples_white_right, plot_contributions_examples_black_left, plot_contributions_examples_black_right, path_to_folders)

    print('Plotting temporal data')
    sub_plot_temp_dyn_data(path_to_analysed, [plot_motion, plot_lumi, plot_same, plot_oppo])

    print('Plotting example traces')
    sub_plot_modelling_example_traces(plot_model_ADD_example, plot_model_ADD_only_example, plot_model_data_example, path_to_folders)

    print('Plotting model overview')
    sub_plot_modelling_overview_mse_tau_w(n_training_rounds=25, subfig_mse_btm=plot_mse_overview_bottom, subfig_mse_top=plot_mse_overview_top, subfig_mse_sup=plot_mse_overview_sup, subfig_tau=plot_timeconstants_overview, subfig_w=plot_weights_overview, path_to_folders=path_to_folders)

    fig.save(fig_save_path)
    subfig.save(supfig_save_path)

    # Note that Figure 2e-f and S2a only contain explanatory cartoons without actual data and therefore are not part of this code.