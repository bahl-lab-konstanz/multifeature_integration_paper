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

    drive_to_left = w_mot * left_integrated_mot + baseline + w_attractor_pos * left_integrated_ph
    drive_to_right = w_mot * right_integrated_mot + baseline + w_attractor_pos * right_integrated_ph

    # For integration of the drive
    exp_kernel_drive = np.concatenate((np.zeros(150), 1 / tau_drive * np.exp(-np.linspace(0, 150, 151) / tau_drive)))
    exp_kernel_drive = exp_kernel_drive / np.sum(exp_kernel_drive)
    left_integrated_drive = convolve1d(drive_to_left, exp_kernel_drive)
    right_integrated_drive = convolve1d(drive_to_right, exp_kernel_drive)

    swims_to_left_tot = ((left_integrated_drive - right_integrated_drive) / (
                left_integrated_drive + right_integrated_drive) + 1) / 2

    swims_to_left_tot_rw = rolling_end_window(swims_to_left_tot, 20)
    swims_to_left_tot_rw = swims_to_left_tot_rw * 100
    swims_to_left_tot_rw[np.isinf(swims_to_left_tot_rw)] = 50
    swims_to_left_tot_rw[np.isnan(swims_to_left_tot_rw)] = 50

    return swims_to_left_tot_rw

def avg_mot_change(model_input, tau_mot=4.26, tau_ph_rep=12.66, tau_drive=6.46, w_mot=2.783, w_repulsor_pos=1.857):
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

    drive_to_left = w_mot * left_integrated_mot + w_repulsor_pos * repulsion_from_right + baseline
    drive_to_right = w_mot * right_integrated_mot + w_repulsor_pos * repulsion_from_left + baseline

    # For integration of the drive
    exp_kernel_drive = np.concatenate((np.zeros(150), 1 / tau_drive * np.exp(-np.linspace(0, 150, 151) / tau_drive)))
    exp_kernel_drive = exp_kernel_drive / np.sum(exp_kernel_drive)
    left_integrated_drive = convolve1d(drive_to_left, exp_kernel_drive)
    right_integrated_drive = convolve1d(drive_to_right, exp_kernel_drive)

    swims_to_left_tot = ((left_integrated_drive - right_integrated_drive) / (
                left_integrated_drive + right_integrated_drive) + 1) / 2

    swims_to_left_tot_rw = rolling_end_window(swims_to_left_tot, 20)
    swims_to_left_tot_rw = swims_to_left_tot_rw * 100
    swims_to_left_tot_rw[np.isinf(swims_to_left_tot_rw)] = 50
    swims_to_left_tot_rw[np.isnan(swims_to_left_tot_rw)] = 50

    return swims_to_left_tot_rw

def avg_lumi_change(model_input, tau_ph_eye=12.66, tau_ph_rep=12.66, tau_drive=6.46, w_attractor_pos=0.126, w_repulsor_pos=1.857):
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

    drive_to_left = w_repulsor_pos * repulsion_from_right + baseline + w_attractor_pos * left_integrated_ph
    drive_to_right = w_repulsor_pos * repulsion_from_left + baseline + w_attractor_pos * right_integrated_ph

    # For integration of the drive
    exp_kernel_drive = np.concatenate((np.zeros(150), 1 / tau_drive * np.exp(-np.linspace(0, 150, 151) / tau_drive)))
    exp_kernel_drive = exp_kernel_drive / np.sum(exp_kernel_drive)
    left_integrated_drive = convolve1d(drive_to_left, exp_kernel_drive)
    right_integrated_drive = convolve1d(drive_to_right, exp_kernel_drive)

    swims_to_left_tot = ((left_integrated_drive - right_integrated_drive) / (
                left_integrated_drive + right_integrated_drive) + 1) / 2

    swims_to_left_tot_rw = rolling_end_window(swims_to_left_tot, 20)
    swims_to_left_tot_rw = swims_to_left_tot_rw * 100
    swims_to_left_tot_rw[np.isinf(swims_to_left_tot_rw)] = 50
    swims_to_left_tot_rw[np.isnan(swims_to_left_tot_rw)] = 50

    return swims_to_left_tot_rw

def avg_mot_lumi_change(model_input, tau_mot=4.26, tau_ph_eye=12.66, tau_ph_rep=12.66, tau_drive=6.46,
                        w_mot=2.783, w_attractor_pos=0.126, w_repulsor_pos=1.857):
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

    drive_to_left = w_mot * left_integrated_mot + w_repulsor_pos * repulsion_from_right + baseline + w_attractor_pos * left_integrated_ph
    drive_to_right = w_mot * right_integrated_mot + w_repulsor_pos * repulsion_from_left + baseline + w_attractor_pos * right_integrated_ph

    # For integration of the drive
    exp_kernel_drive = np.concatenate((np.zeros(150), 1 / tau_drive * np.exp(-np.linspace(0, 150, 151) / tau_drive)))
    exp_kernel_drive = exp_kernel_drive / np.sum(exp_kernel_drive)
    left_integrated_drive = convolve1d(drive_to_left, exp_kernel_drive)
    right_integrated_drive = convolve1d(drive_to_right, exp_kernel_drive)

    swims_to_left_tot = ((left_integrated_drive - right_integrated_drive) / (
                left_integrated_drive + right_integrated_drive) + 1) / 2

    swims_to_left_tot_rw = rolling_end_window(swims_to_left_tot, 20)
    swims_to_left_tot_rw = swims_to_left_tot_rw * 100
    swims_to_left_tot_rw[np.isinf(swims_to_left_tot_rw)] = 50
    swims_to_left_tot_rw[np.isnan(swims_to_left_tot_rw)] = 50

    return swims_to_left_tot_rw

def get_data(folder_name, stim_len_timepoints, stim_name, nsplits=3, debug=False):
    path_to_local_folder = Path(rf'X:\ZT6 Multi-fish behavior setup\Freely swimming larvae\Katja\{folder_name}\Analysis')
    df_path = f'{path_to_local_folder}/data_analysed.hdf5'
    df = pd.read_hdf(df_path)
    gkf = GroupShuffleSplit(n_splits=nsplits)
    datasplitss = {}
    datasplitss.update({f"{folder_name}": [(train_ids, test_ids) for train_ids, test_ids in
                                           gkf.split(df.index.unique('experiment_ID'),
                                                     groups=df.index.unique('experiment_ID').tolist())]})

    train_df = df[np.isin(df.index.get_level_values('experiment_ID'),
                          df.index.unique('experiment_ID')[datasplitss[f'{folder_name}'][0][0]])]
    test_df = df[np.isin(df.index.get_level_values('experiment_ID'),
                         df.index.unique('experiment_ID')[datasplitss[f'{folder_name}'][0][1]])]

    if debug:
        print(f'{folder_name} - {stim_name}: ')
        print(f'Train N FISH: {len(train_df.index.unique("experiment_ID"))}')
        print(f'Test N FISH: {len(test_df.index.unique("experiment_ID"))}\n')

    stimulus_data = train_df.xs(stim_name, level='stimulus_name')
    data_mean = stimulus_data.groupby('window_time').mean()['percentage_left']
    data_sem = stimulus_data.groupby('window_time').std()['percentage_left'] / np.sqrt(
        len(train_df.index.unique('experiment_ID')))
    stimulus_data_test = test_df.xs(stim_name, level='stimulus_name')
    data_mean_test = stimulus_data_test.groupby('window_time').mean()['percentage_left']
    data_sem_test = stimulus_data_test.groupby('window_time').std()['percentage_left'] / np.sqrt(
        len(test_df.index.unique('experiment_ID')))

    # This is needed in case there were no bouts in the first 0.1 s (this happens).
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

    data_mean = fill_nans(data_mean)
    data_sem = fill_nans(data_sem)
    data_mean_test = fill_nans(data_mean_test)
    data_sem_test = fill_nans(data_sem_test)

    return data_mean, data_sem, data_mean_test, data_sem_test

def train_model_once(model_func, folder_names, stim_len_timepoints, stim_names, subfig=None):
    n_loops = len(stim_len_timepoints) * len(stim_names)

    list_of_folder_ids = []
    list_of_stim_ids = []
    for i in range(n_loops):
        random_folder_id = np.random.randint(len(folder_names))
        random_stim_id = np.random.randint(len(stim_names))

        list_of_folder_ids = np.append(list_of_folder_ids, random_folder_id)
        list_of_stim_ids = np.append(list_of_stim_ids, random_stim_id)

        data_mean, data_sem, data_mean_test, data_sem_test = get_data(folder_names[random_folder_id],
                                                                      stim_len_timepoints[random_folder_id],
                                                                      stim_names[random_stim_id])
        left_mot_input, right_mot_input, left_lumi_input, right_lumi_input = get_stim_input(
            folder_names[random_folder_id], stim_len_timepoints[random_folder_id], stim_names[random_stim_id])

        time = np.linspace(0, (len(data_mean) - 1) / 10, len(data_mean))

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

        if i < n_loops - 1:
            model_input = np.hstack((model_input, [np.zeros(50), np.zeros(50), np.zeros(50), np.ones(50), np.ones(50)]))
            data_full = np.append(data_full, 50 * np.ones(50))
            data_full_test = np.append(data_full_test, 50 * np.ones(50))
            data_sem_full = np.append(data_sem_full, 0 * np.ones(50))
            data_sem_full_test = np.append(data_sem_full_test, 0 * np.ones(50))
            folder_ids = np.append(folder_ids, -1 * np.ones(50))
            stim_ids = np.append(stim_ids, -1 * np.ones(50))

    best_mse = 1000
    for c in range(5):
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

        try:
            popt, pcov = curve_fit(model_func, model_input, data_full, p0=p0, bounds=bounds)
        except:
            print('Warning optimal params not found, using initial params instead.')
            popt = p0
        test_mse = np.nanmean(np.square(model_func(model_input, *popt) - data_full_test))

        if test_mse < best_mse:
            best_mse = test_mse
            best_popt = popt

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

def train_model_full(model_func, folder_names, stim_len_timepoints, stim_names, subfig=None, train_loops=10, debug=False):
    best_test_mse = 1000
    all_params = []
    all_mse = []
    list_of_lists_of_folder_ids = []
    for t in range(train_loops):
        print(f'Training round {t}')
        popt, test_mse, folder_ids = train_model_once(model_func,
                                                      folder_names,
                                                      stim_len_timepoints,
                                                      stim_names,
                                                      subfig=subfig)
        list_of_lists_of_folder_ids = np.append(list_of_lists_of_folder_ids, folder_ids)

        all_params = np.append(all_params, popt)
        all_mse = np.append(all_mse, test_mse)
        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_params = popt

    if debug:
        print(best_params)
        print(all_params)
        print(all_mse)

    return all_params, all_mse, list_of_lists_of_folder_ids

def test_model_once(model_func, params, folder_names, stim_len_timepoints, stim_names, subfig_data=None, subfig_model=None, training_folder_ids=None, pick_stim_ids=None, pl_color='k', debug=False):
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

        list_of_stim_ids = np.append(list_of_stim_ids, stim_id)

        _, _, data_mean, data_sem = get_data(folder_names[folder_id],
                                             stim_len_timepoints[folder_id],
                                             stim_names[stim_id],
                                             nsplits=3)
        left_mot_input, right_mot_input, left_lumi_input, right_lumi_input = get_stim_input(folder_names[folder_id],
                                                                                            stim_len_timepoints[
                                                                                                folder_id],
                                                                                            stim_names[stim_id])

        time = np.linspace(0, (len(data_mean) - 1) / 10, len(data_mean))

        if i == 0:
            model_input = [time, left_mot_input, right_mot_input, left_lumi_input, right_lumi_input]
            data_full = data_mean
            data_sem_full = data_sem
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

        if i < len(stim_len_timepoints) * len(stim_names) - 1:
            model_input = np.hstack((model_input, [np.zeros(50), np.zeros(50), np.zeros(50), np.ones(50), np.ones(50)]))
            data_full = np.append(data_full, 50 * np.ones(50))
            data_sem_full = np.append(data_sem_full, 0 * np.ones(50))
            in_stim_tracker = np.append(in_stim_tracker, np.ones(50))
            folder_ids = np.append(folder_ids, -1 * np.ones(50))
            stim_ids = np.append(stim_ids, -1 * np.ones(50))

    test_mse = np.nanmean(np.square(model_func(model_input, *params) - data_full))
    if debug:
        print(test_mse)

    if subfig_data is not None:
        print('length of data', len(data_full))
        data_full[in_stim_tracker.astype(bool)] = np.nan
        data_sem_full[in_stim_tracker.astype(bool)] = np.nan
        subfig_data.draw_line(np.arange(len(data_full)), data_full, yerr=data_sem_full, lc='tab:blue', lw=1,
                         eafc='#AEC7E8', eaalpha=1.0, ealw=1, eaec='#AEC7E8')
        subfig_data.draw_line([len(data_full)-150, len(data_full)], [21, 21], lc='k')
        subfig_data.draw_text(len(data_full)-75, 15, '15s')

    if subfig_model is not None:
        model_output = model_func(model_input, *params)
        model_output[in_stim_tracker.astype(bool)] = np.nan
        subfig_model.draw_line(np.arange(len(data_full)), model_output, lc=pl_color, lw=1.)

    return test_mse

def sub_plot_temp_dyn_data(path_to_analysed_data, subfigs):
    analysed_df = pd.read_hdf(path_to_analysed_data)

    stim_names = ['Mot', 'Lumi', 'Same', 'Oppo']
    plot_names = ['only-motion (M)', 'only-luminance (L)', 'congruent (M=L)', 'conflicting (M\u2260L)']
    for st, pl_name, plot in zip(stim_names, plot_names, subfigs):
        stim = f'{st} W'
        stim_df = analysed_df.xs(stim, level='stimulus_name')
        n_fish = len(stim_df.index.unique('experiment_ID'))
        grouped_df = stim_df.groupby('window_time')

        mean = grouped_df['percentage_left'].mean()[0.5:]
        std = grouped_df['percentage_left'].std()[0.5:]
        sem = std / np.sqrt(n_fish)
        binned_time = mean.index.unique('window_time')

        plot.draw_line(x=binned_time, y=mean, yerr=sem, lc='#676767', eafc='#989898', eaalpha=1.0, lw=1, ealw=1, eaec='#989898')

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

def sub_plot_modelling_example_traces(plot_model_ADD_example, plot_model_ADD_only_example, plot_data_example):
    folder_names_half = ['converted_phototaxis_dotmotion_integration_simultaneous_white', # Simultaneous
                        'phototaxis_dotmotion_white_Sep', # White
                        'phototaxis_dotmotion_simultaneous_low_Sep', # Simultaneous low
                         'converted_phototaxis_dotmotion_integration_peppersalt', # peppersalt
                        'converted_phototaxis_dotmotion_integration_long']  # long
    stim_len_timepoints_half = np.array([250, 300, 450, 350, 600])
    stim_names_half = ['Motion', 'Photo', 'Same', 'Oppo']

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

    folder_names_test = ['beh_simultaneous_blackwhite_white',  # simultaneous white
                        'phototaxis_dotmotion_reverse_Sep',  # reverse
                         'beh_simultaneous_blackwhite_black',  # simultaneous black
                        'phototaxis_dotmotion_integration_halfoverlap',  # Halfoverlap
                        'beh_simultaneous_10only']  #simultaneous very low
    stim_len_timepoints_test = np.array([250, 300, 250, 350, 250])
    stim_names_test = ['Same', 'Oppo']

    testing_folder_ids = np.array([0, 1, 2, 3, 4, 3, 1, 0, 4, 2])
    testing_stim_ids = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 1])

    total_time = 450 + np.sum(stim_len_timepoints_test[testing_folder_ids.astype(int)])
    print('total time example traces is:, ', total_time)

    model_params, model_train_mse, _ = train_model_full(model_func=avg_mot_lumi_change,
                                                        folder_names=folder_names_half,
                                                        stim_len_timepoints=stim_len_timepoints_half,
                                                        stim_names=stim_names_half,
                                                        subfig=None,
                                                        train_loops=1)

    model_params = np.array(model_params).reshape(-1, 7)

    model_test_mse = []
    for params in model_params:
        model_test_mse = np.append(model_test_mse, test_model_once(model_func=avg_mot_lumi_change,
                                                                   params=params,
                                                                   folder_names=folder_names_test,
                                                                   stim_len_timepoints=stim_len_timepoints_test,
                                                                   stim_names=stim_names_test,
                                                                   subfig_data=None,
                                                                   subfig_model=plot_model_ADD_example,
                                                                   training_folder_ids=testing_folder_ids,
                                                                   pick_stim_ids=testing_stim_ids,
                                                                   pl_color='cadetblue'))

    model_params, model_train_mse, _ = train_model_full(model_func=avg_mot_lumi_change,
                                                        folder_names=folder_names_onlyML,
                                                        stim_len_timepoints=stim_len_timepoints_onlyML,
                                                        stim_names=stim_names_onlyML,
                                                        subfig=None,
                                                        train_loops=1)

    model_params = np.array(model_params).reshape(-1, 7)

    model_test_mse = []
    for params in model_params:
        model_test_mse = np.append(model_test_mse, test_model_once(model_func=avg_mot_lumi_change,
                                                                   params=params,
                                                                   folder_names=folder_names_test,
                                                                   stim_len_timepoints=stim_len_timepoints_test,
                                                                   stim_names=stim_names_test,
                                                                   subfig_data=plot_data_example,
                                                                   subfig_model=plot_model_ADD_only_example,
                                                                   training_folder_ids=testing_folder_ids,
                                                                   pick_stim_ids=testing_stim_ids,
                                                                   pl_color='cyan'))

    return

def sub_plot_example_traces_model_fit(path_to_analysed_data, subfiga, subfigb):
    model_params = [8, 52, 13, 11, 3.9, 0.9, 4]
    subfiga.draw_line([0, 1150], [50, 50], lc='w', lw=1.5)
    subfigb.draw_line([0, 1150], [50, 50], lc='w', lw=1.5)

    analysed_df = pd.read_hdf(path_to_analysed_data)

    stim_names = ['Mot', 'Lumi', 'Same', 'Oppo']
    x_starts = [0, 300, 600, 900, 1200, 1500, 1800, 2100]
    for st, x_start in zip(stim_names, x_starts):
        stim = f'{st} W'
        stim_df = analysed_df.xs(stim, level='stimulus_name')
        n_fish = len(stim_df.index.unique('experiment_ID'))
        grouped_df = stim_df.groupby('window_time')

        mean = grouped_df['percentage_left'].mean()[0.5:]
        std = grouped_df['percentage_left'].std()[0.5:]
        sem = std / np.sqrt(n_fish)
        binned_time = mean.index.unique('window_time') * 10 + x_start

        subfiga.draw_line(x=binned_time, y=mean*100, yerr=sem*100, lc='#676767', eafc='#989898', eaalpha=1.0, lw=1, ealw=1, eaec='#989898')

        stim = f'{st} B'
        stim_df = analysed_df.xs(stim, level='stimulus_name')
        n_fish = len(stim_df.index.unique('experiment_ID'))
        grouped_df = stim_df.groupby('window_time')

        mean = grouped_df['percentage_left'].mean()[0.5:]
        std = grouped_df['percentage_left'].std()[0.5:]
        sem = std / np.sqrt(n_fish)
        binned_time = mean.index.unique('window_time') * 10 + x_start

        subfigb.draw_line(x=binned_time, y=mean*100, yerr=sem*100, lc='k', eafc='#404040', alpha=0.5, eaalpha=0.5, lw=1, ealw=1, eaec='#404040')

    folder_names_test = ['beh_simultaneous_blackwhite_white',  # simultaneous white
                         ]
    stim_len_timepoints_test = np.array([250])
    stim_names_test = ['Motion', 'Photo', 'Same', 'Oppo']

    testing_folder_ids = np.array([0, 0, 0, 0])
    testing_stim_ids = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    # model_params, model_train_mse, _ = train_model_full(model_func=avg_mot_lumi_change,
    #                                                     folder_names=folder_names_test,
    #                                                     stim_len_timepoints=stim_len_timepoints_test,
    #                                                     stim_names=stim_names_test,
    #                                                     subfig=None,
    #                                                     train_loops=10)
    #
    # model_params = np.array(model_params).reshape(-1, 7)
    # print(np.nanmedian(model_params, axis=0))
    test_model_once(model_func=avg_mot_lumi_change,
                    params=model_params,
                    folder_names=folder_names_test,
                    stim_len_timepoints=stim_len_timepoints_test,
                    stim_names=stim_names_test,
                    subfig_data=None,
                    subfig_model=subfiga,
                    training_folder_ids=testing_folder_ids,
                    pick_stim_ids=testing_stim_ids,
                    pl_color='cyan')

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
                    subfig_data=None,
                    subfig_model=subfigb,
                    training_folder_ids=testing_folder_ids,
                    pick_stim_ids=testing_stim_ids,
                    pl_color='cyan')

    subfigb.draw_line([1100, 1150], [22, 22], lc='k')
    subfigb.draw_text(1125, 15, '5s')
    return

def sub_plot_modelling_overview_mse_tau_w(n_training_rounds, subfig_mse_btm, subfig_mse_top, subfig_tau, subfig_w):
    folder_names_half = ['converted_phototaxis_dotmotion_integration_simultaneous_white',  # Simultaneous
                        'phototaxis_dotmotion_white_Sep',  # White
                        'phototaxis_dotmotion_simultaneous_low_Sep',  # simul low
                         'converted_phototaxis_dotmotion_integration_peppersalt',  # peppersalt
                        'converted_phototaxis_dotmotion_integration_long']   # long
    stim_len_timepoints_half = np.array([250, 300, 450, 350, 600])
    stim_names_half = ['Motion', 'Photo', 'Same', 'Oppo']

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

    folder_names_half_test = ['beh_simultaneous_blackwhite_white',  # simultaneous white
                        'phototaxis_dotmotion_reverse_Sep',  # Reverse
                         'beh_simultaneous_blackwhite_black',  # simultaneous black
                        'phototaxis_dotmotion_integration_halfoverlap',  # Halfoverlap
                        'beh_simultaneous_10only']  #simulataneous very low
    stim_len_timepoints_half_test = np.array([250, 300, 250, 350, 250])
    stim_names_half_test = ['Motion', 'Photo', 'Same', 'Oppo']

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

    model_test_mses = [[]] * 5
    print(f'Running model avg_mot_lumi_change multifeature training')

    # Average model with change repulsion control
    model_params, model_train_mse, all_folder_ids = train_model_full(model_func=avg_mot_lumi_change,
                                                                     folder_names=folder_names_half,
                                                                     stim_len_timepoints=stim_len_timepoints_half,
                                                                     stim_names=stim_names_half,
                                                                     subfig=None,
                                                                     train_loops=n_training_rounds)

    model_params = np.array(model_params).reshape(-1, 7)
    for param in range(7):
        print('Final param means: ')
        print(f'param {param}: {model_params[:, param]}')

    all_folder_ids = np.array(all_folder_ids).reshape(-1, 20)
    for params, folder_ids in zip(model_params, all_folder_ids):
        model_test_mses[0] = np.append(model_test_mses[0],
                                       test_model_once(model_func=avg_mot_lumi_change, params=params,
                                                       folder_names=folder_names_half_test,
                                                       stim_len_timepoints=stim_len_timepoints_half_test,
                                                       stim_names=stim_names_half_test,
                                                       subfig_data=None, subfig_model=None,
                                                       training_folder_ids=folder_ids))

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

    for param in range(7):
        if param < 4:
            subfig_tau.draw_scatter(2*param * np.ones(len(model_params[:, param])) + np.random.uniform(-0.4, 0.4, len(model_params[:, param])), model_params[:, param], pc='cadetblue', ec='cadetblue')
            subfig_tau.draw_line([2*param - 0.4, 2*param + 0.4], [np.nanmedian(model_params[:, param]), np.nanmedian(model_params[:, param])], lc='k', lw='1')
        else:
            subfig_w.draw_scatter((2*param - 8) * np.ones(len(model_params[:, param])) + np.random.uniform(-0.4, 0.4, len(model_params[:, param])), model_params[:, param], pc='cadetblue', ec='cadetblue')
            subfig_w.draw_line([2*param - 8 - 0.4, 2*param - 8 + 0.4], [np.nanmedian(model_params[:, param]), np.nanmedian(model_params[:, param])], lc='k', lw='1')

    for model_counter, (modelfunction, pl_color, param_size) in enumerate(zip(
            [avg_mot_lumi_change, avg_mot_lumi, avg_mot_change, avg_lumi_change],
            ['cyan', '#D55E00', '#E69F00', '#359B73'],
            [7, 5, 5, 5])):

        print(f'Running model {model_counter} unifeature training')
        model_params, model_train_mse, all_folder_ids = train_model_full(model_func=modelfunction,
                                                                         folder_names=folder_names_onlyML,
                                                                         stim_len_timepoints=stim_len_timepoints_onlyML,
                                                                         stim_names=stim_names_onlyML,
                                                                         subfig=None,
                                                                         train_loops=n_training_rounds)

        model_params = np.array(model_params).reshape(-1, param_size)
        for param in range(param_size):
            print('Final param means: ')
            print(f'param {param}: {model_params[:, param]}')

        all_folder_ids = np.array(all_folder_ids).reshape(-1, 20)
        for params, folder_ids in zip(model_params, all_folder_ids):
            model_test_mses[model_counter+1] = np.append(model_test_mses[model_counter+1],
                                                         test_model_once(model_func=modelfunction, params=params,
                                                                         folder_names=folder_names_onlyML_test,
                                                                         stim_len_timepoints=stim_len_timepoints_onlyML_test,
                                                                         stim_names=stim_names_onlyML_test,
                                                                         subfig_data=None, subfig_model=None,
                                                                         training_folder_ids=folder_ids))

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


    for model1, model2, h in zip([0, 1, 1, 1],
                                 [1, 2, 3, 4],
                                 [80, 120, 160, 200]):
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
    return


if __name__ == '__main__':
    path_to_analysed = Path(r'X:\ZT6 Multi-fish behavior setup\Freely swimming larvae\Katja\beh_simultaneous_blackwhite\Analysis\data_analysed.hdf5')

    xticks = [0, 5, 20, 25]
    vspans = [[5, 20, 'lightgray', 1.0], ]
    fig = Figure(fig_width=18, fig_height=17)
    subfig = Figure(fig_width=18, fig_height=6)
    plot_motion = fig.create_plot(xpos=1, ypos=13.5, plot_height=2, plot_width=3.5, errorbar_area=True,
                                  xmin=0, xmax=25, yl='Left swims (%)', ymin=0.25,
                                  ymax=0.9, yticks=[0.30, 0.50, 0.70, 0.90], yticklabels=['30', '50', '70', '90'], hlines=[0.5],
                                  helper_lines_lc='w', helper_lines_dashes=(2, 0), helper_lines_lw=1, vspans=vspans)

    plot_lumi = fig.create_plot(xpos=5, ypos=13.5, plot_height=2, plot_width=3.5, errorbar_area=True,
                                  xmin=0, xmax=25, ymin=0.25,
                                  ymax=0.9, yticks=[0.30, 0.50, 0.70, 0.90], yticklabels=['', '', '', ''], hlines=[0.5],
                                  helper_lines_lc='w', helper_lines_dashes=(2, 0), helper_lines_lw=1, vspans=vspans)

    plot_same = fig.create_plot(xpos=9, ypos=13.5, plot_height=2, plot_width=3.5, errorbar_area=True,
                                  xmin=0, xmax=25, ymin=0.25,
                                  ymax=0.9, yticks=[0.30, 0.50, 0.70, 0.90], yticklabels=['', '', '', ''], hlines=[0.5],
                                  helper_lines_lc='w', helper_lines_dashes=(2, 0), helper_lines_lw=1, vspans=vspans)

    plot_oppo = fig.create_plot(xpos=13, ypos=13.5, plot_height=2, plot_width=3.5, errorbar_area=True,
                                  xmin=0, xmax=25, ymin=0.25,
                                  ymax=0.9, yticks=[0.30, 0.50, 0.70, 0.90], yticklabels=['', '', '', ''], hlines=[0.5],
                                  helper_lines_lc='w', helper_lines_dashes=(2, 0), helper_lines_lw=1, vspans=vspans)

    plot_model_ADD_example = fig.create_plot(xpos=1, ypos=9, plot_height=1.4, plot_width=7, errorbar_area=True,
                                               xmin=0, xmax=3250, ymin=20, ymax=90, yticks=[50, 75], yl='Left swims (%)')
    plot_model_ADD_only_example = fig.create_plot(xpos=1, ypos=8., plot_height=1.4, plot_width=7, errorbar_area=True,
                                               xmin=0, xmax=3250, ymin=20, ymax=90, yticks=[50, 75], yl='Left swims (%)')
    plot_model_data_example = fig.create_plot(xpos=1, ypos=6.5, plot_height=1.4, plot_width=7, errorbar_area=True,
                                               xmin=0, xmax=3250, ymin=20, ymax=90, yticks=[25, 50, 75], yl='Left swims (%)')


    plot_white_example = subfig.create_plot(xpos=1, ypos=0.5, plot_height=1.4, plot_width=7.5, errorbar_area=True,
                                                 xmin=0, xmax=1150, ymin=20, ymax=90, yticks=[25, 50, 75], yl='Left swims (%)',
                                                 vspans=[[50, 200, 'lightgray', 1.0], [350, 500, 'lightgray', 1.0], [650, 800, 'lightgray', 1.0], [950, 1100, 'lightgray', 1.0]])
    plot_black_example = subfig.create_plot(xpos=9.5, ypos=0.5, plot_height=1.4, plot_width=7.5, errorbar_area=True,
                                                 xmin=0, xmax=1150, ymin=20, ymax=90, yticks=[25, 50, 75],
                                                 vspans=[[50, 200, 'lightgray', 1.0], [350, 500, 'lightgray', 1.0], [650, 800, 'lightgray', 1.0], [950, 1100, 'lightgray', 1.0]])

    sub_plot_example_traces_model_fit(path_to_analysed, plot_white_example, plot_black_example)

    plot_mse_overview_bottom = fig.create_plot(xpos=9, ypos=7.5, plot_height=1.25, plot_width=2.25,
                                        xmin=-1, xmax=5, ymin=0, ymax=75, yticks=[10, 20, 30, 40, 50], yl='MSE',
                                        xticks=[0, 1, 2, 3, 4], xticklabels=['multifeature\ntesting', 'unifeature\ntesting', 'X lumi change',
                                                                             'X lumi level', 'X motion'], xticklabels_rotation=90)
    plot_mse_overview_top = fig.create_plot(xpos=9, ypos=8.75, plot_height=1.25, plot_width=2.25,
                                        xmin=-1, xmax=5, ymin=75, ymax=210, yticks=[100, 150, 200])

    plot_timeconstants_overview = fig.create_plot(xpos=12, ypos=7.5, plot_height=2.5, plot_width=2.25,
                                                  xmin=-1, xmax=8, ymin=0, ymax=31, yticks=[0, 5, 10, 15, 20, 25, 30],
                                                  yticklabels=['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'], yl='Time constant (s)',
                                                  xticks=[0.5, 2.5, 4.5, 6.5], xticklabels=['\u03C4 motion', '\u03C4 lumi level', '\u03C4 lumi change', '\u03C4 multifeature'],
                                                  xticklabels_rotation=90)
    plot_weights_overview = fig.create_plot(xpos=15, ypos=7.5, plot_height=2.5, plot_width=2.25,
                                            xmin=-1, xmax=6, ymin=0, ymax=3.5, yticks=[0, 1, 2, 3], yl='Weight',
                                            xticks=[0.5, 2.5, 4.5], xticklabels=['w motion', 'w lumi level', 'w lumi change'],
                                            xticklabels_rotation=90)

    print('Plotting temporal data')
    sub_plot_temp_dyn_data(path_to_analysed, [plot_motion, plot_lumi, plot_same, plot_oppo])

    print('Plotting example traces')
    sub_plot_modelling_example_traces(plot_model_ADD_example, plot_model_ADD_only_example, plot_model_data_example)

    print('Plotting model overview')
    sub_plot_modelling_overview_mse_tau_w(n_training_rounds=25, subfig_mse_btm=plot_mse_overview_bottom, subfig_mse_top=plot_mse_overview_top, subfig_tau=plot_timeconstants_overview, subfig_w=plot_weights_overview)  #25 rounds

    fig.save('C:/users/katja/Desktop/fig2.pdf')
    subfig.save('C:/users/katja/Desktop/figS2.pdf')