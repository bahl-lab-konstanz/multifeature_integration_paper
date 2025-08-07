import numpy as np
import h5py

def cohens_d(group1, group2):
    # Calculating means of the two groups
    mean1, mean2 = np.mean(group1), np.mean(group2)

    # Calculating pooled standard deviation
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

    # Calculating Cohen's d
    d = (mean1 - mean2) / pooled_std

    return d

def rolling_end_window(input, window):
    output = np.zeros(len(input))
    for i in range(len(input)):
        if i < int(window):
            output[i] = np.mean(input[:i + int(window)])
        else:
            output[i] = np.mean(input[i - int(window):i])
    return output

def fill_nans(input_arr):
    input_arr = np.array(input_arr)
    if np.isnan(input_arr[0]):
        input_arr[0] = input_arr[1]
    if np.isnan(input_arr[-1]):
        input_arr[-1] = input_arr[-2]
    for i in range(1, len(input_arr) - 1):
        if np.isnan(input_arr[i]):
            input_arr[i] = (input_arr[i - 1] + input_arr[i + 1]) / 2
    return input_arr

def create_combined_region_npy_mask(masks_path, regions=['Diencephalon', 'Dorsal Thalamus', 'Ventral Thalamus', 'Pretectum']):
    masks = h5py.File(masks_path)
    mask = np.zeros((621, 1406, 138))
    for region_count, region_name in enumerate(regions):
        print(f'Filling in {region_name}: n pixels = {masks[f"{region_name}/ind_mask_volume"].shape[1]}')
        mask[masks[f'{region_name}/ind_mask_volume'][2, :], masks[f'{region_name}/ind_mask_volume'][1, :], masks[f'{region_name}/ind_mask_volume'][0, :]] = region_count + 1
    return mask