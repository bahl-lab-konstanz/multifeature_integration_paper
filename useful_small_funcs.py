import numpy as np
import h5py

def cohens_d(group1, group2):
    '''
    This function calculates the cohens-d effect size between two groups of data.
    :param group1: First group of data (list or array)
    :param group2: Second group of data (list or array)
    :return: Effect size (between -1 and 1).
    '''
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
    '''
    This function applies a rolling average window (of size window) to the input data.
    :param input: Array of input data
    :param window: Integer window size
    :return: Output array of the same size as the input.
    '''
    output = np.zeros(len(input))
    for i in range(len(input)):
        # For the first entries of the input, the window size is not reached yet and the average is taken among the smaller dataset.
        if i < int(window):
            output[i] = np.mean(input[:i + int(window)])
        else:
            output[i] = np.mean(input[i - int(window):i])
    return output

def fill_nans(input_arr):
    '''
    This function fills the NaN values in the input array by taking the average value of the 2 neighbours
    :param input_arr: Array with input data.
    :return: Array of the same size as input with all NaN values substituted by the average of the neighbouring values.
    '''
    input_arr = np.array(input_arr)
    # If the first entry is NaN fill with the same value as the right neighbour
    if np.isnan(input_arr[0]):
        input_arr[0] = input_arr[1]
    # If the last entry is NaN fill with the same value as the left neighbour
    if np.isnan(input_arr[-1]):
        input_arr[-1] = input_arr[-2]
    # If any other entry is NaN fill with the average of both neighbours.
    for i in range(1, len(input_arr) - 1):
        if np.isnan(input_arr[i]):
            input_arr[i] = (input_arr[i - 1] + input_arr[i + 1]) / 2
    return input_arr

def create_combined_region_npy_mask(masks_path, regions=['Diencephalon', 'Dorsal Thalamus', 'Ventral Thalamus', 'Pretectum']):
    '''
    This function creates a mask in which pixel gets a number depending on the region it is part of. 0 means not part of any of the given regions.
    Note that if the regions overlap (e.g. a pixel can be part of multiple regions), the pixel will get the ID of the last region in the list.
    :param masks_path: The path to the hdf5 file with all brain regions.
    :param regions: A list of all region names to be added to the region_mask
    :return: The mask with all regions.
    '''
    # Load the region masks
    masks = h5py.File(masks_path)
    # Create the array and initialize each pixel to 0 (Not part of any region).
    mask = np.zeros((621, 1406, 138))
    # Loop through all brain regions and update the pixels with the current region ID if they are part of the region.
    for region_count, region_name in enumerate(regions):
        print(f'Filling in {region_name}: n pixels = {masks[f"{region_name}/ind_mask_volume"].shape[1]}')
        mask[masks[f'{region_name}/ind_mask_volume'][2, :], masks[f'{region_name}/ind_mask_volume'][1, :], masks[f'{region_name}/ind_mask_volume'][0, :]] = region_count + 1
    return mask