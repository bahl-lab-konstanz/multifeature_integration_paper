import numpy as np

def logic_regression_left_motion_3stim(traces_df, thresh_resp=0.2, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify leftward motion integrators based on a minimal stimulus set of 3 stimuli: lumi_off_dots_left, lumi_right_dots_right, and lumi_left_dots_off.
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during motion c.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only leftward motion integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lo_dl = 'lumi_off_dots_left'
    ll_do = 'lumi_left_dots_off'
    lr_dr = 'lumi_right_dots_right'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        lo_dl = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        ll_do = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        lr_dr = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second statement checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &

                  # Next check for temporal integration: late (c) higher than early (b) stimulus.
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{b}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check that there is no post-motion response to the contralateral stimulus: early post (d) < threshold + late post (e).
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{e}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_left_drive_3stim(traces_df, thresh_resp=0.2, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify leftward multifeature integrators based on a minimal stimulus set of 3 stimuli: lumi_off_dots_left, lumi_right_dots_right, and lumi_left_dots_off.
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during motion c.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only leftward multifeature integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lo_dl = 'lumi_off_dots_left'
    lr_dr = 'lumi_right_dots_right'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        lo_dl = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        lr_dr = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second statement checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &

                  # Next check that the luminance peak responses are higher than the persistent response: early (b) and early post (d) are higher than late (c).
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_left_lumi_3stim(traces_df, thresh_resp=0.2, thresh_below=0.9, thresh_min=0.1,
                                     shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify leftward luminance integrators based on a minimal stimulus set of 3 stimuli: lumi_off_dots_left, lumi_right_dots_right, and lumi_left_dots_off.
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance c.
    :param thresh_below: Threshold in ratio that the activity should be below during decrease in activity.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only leftward luminance integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    lo_dl = 'lumi_off_dots_left'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        lr_dr = np.random.choice(
            ['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        ll_do = np.random.choice(
            ['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        lo_dl = np.random.choice(
            ['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second statement checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &

                  # Next check that the decreased responses are low enough: late stimulus (c) < ratio_threshold * pre (a) and late post (e).
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check for temporal integration: late (c) higher than early (b) stimulus.
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{b}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_left_diff_3stim(traces_df, thresh_resp=0.2, thresh_peaks=1.25, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify leftward luminance change detectors based on a minimal stimulus set of 3 stimuli: lumi_off_dots_left, lumi_right_dots_right, and lumi_left_dots_off.
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance b.
    :param thresh_peaks: Threshold in ratio of how much higher the strong contrast peaks should be then the weak contrast peaks.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only leftward luminance change detectors.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second statement checks that the minimal activity level during the relevant stimulus is met.
    if shuffle_stim_idx:
        lr_dr = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        ll_do = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > thresh_resp) &

                  # Next check that the response during early stimulus (b) is higher than pre (a) and late stimulus (c) stimulus.
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &

                  # Next check that the response during early post stimulus (d) is higher than during late stimulus (c) and late post (e) stimulus.
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check that the weak contrast responses early (b) and early post (d) are higher than pre (a) and late post (e)
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &

                  # Next check that the high contrast peak is higher than ratio_threshold * weak contrast peak.
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > thresh_peaks * traces_df[f'{ll_do}_resp_{b}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_left_bright_3stim(traces_df, thresh_resp=0.2, thresh_peaks=1.5, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify leftward luminance increase detectors based on a minimal stimulus set of 3 stimuli: lumi_off_dots_left, lumi_right_dots_right, and lumi_left_dots_off.
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance b.
    :param thresh_peaks: Threshold in ratio of how much higher the strong contrast peaks should be then the weak contrast peaks.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only leftward luminance increase detectors.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        lr_dr = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        ll_do = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second statement checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > thresh_resp) &

                  # Next check that the response during early stimulus (b) is higher than pre (a), late (c), and both post (d-e) stimulus.
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{d}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: early stimulus (b) < threshold + pre (a)
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &

                  # Next check that the weak contrast response early post (d) is higher than late stimulus (c)
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &

                  # Next check that the high contrast peak is higher than ratio_threshold * weak contrast peak.
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > thresh_peaks * traces_df[f'{ll_do}_resp_{d}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: early post stimulus (d) < threshold + pre (a)
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_left_dark_3stim(traces_df, thresh_resp=0.2, thresh_peaks=1.5, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify leftward luminance decrease detectors based on a minimal stimulus set of 3 stimuli: lumi_off_dots_left, lumi_right_dots_right, and lumi_left_dots_off.
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance d.
    :param thresh_peaks: Threshold in ratio of how much higher the strong contrast peaks should be then the weak contrast peaks.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only leftward luminance decrease detectors.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        lr_dr = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        ll_do = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second statement checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > thresh_resp) &

                  # Next check that the response during early post stimulus (d) is higher than pre (a), early (b), late (c), and late post (e) stimulus.
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: early post stimulus (d) < threshold + pre (a)
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &

                  # Next check that the weak contrast responses early (b) is higher than pre (a)
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &

                  # Next check that the high contrast peak is higher than ratio_threshold * weak contrast peak.
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > thresh_peaks * traces_df[f'{ll_do}_resp_{b}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: early stimulus (b) < threshold + pre (a)
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)))

    return traces_df[good_cells]


def logic_regression_left_motion(traces_df, thresh_resp=0.2, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify leftward motion integrators based on the full stimulus set of 9 stimuli: lumi_[off/left/right]_dots_[off/left/right].
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during motion c.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only leftward motion integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lo_dl = 'lumi_off_dots_left'
    ll_dl = 'lumi_left_dots_left'
    lr_dl = 'lumi_right_dots_left'
    ll_dr = 'lumi_left_dots_right'
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    lr_do = 'lumi_right_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        lo_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        ll_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        ll_do = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_do = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &

                  # Next check for temporal integration: late (c) higher than early (b) stimulus.
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{b}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check that there is no post-motion response to the contralateral stimulus: early post (d) < threshold + late post (e).
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{e}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_left_drive(traces_df, thresh_resp=0.2, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify leftward multifeature integrators based on the full stimulus set of 9 stimuli: lumi_[off/left/right]_dots_[off/left/right].
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during motion c.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only leftward multifeature integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lo_dl = 'lumi_off_dots_left'
    ll_dl = 'lumi_left_dots_left'
    lr_dl = 'lumi_right_dots_left'
    lo_dr = 'lumi_off_dots_right'
    lr_dr = 'lumi_right_dots_right'
    lo_do = 'lumi_off_dots_off'
    lr_do = 'lumi_right_dots_off'
    ll_dr = 'lumi_left_dots_right'
    ll_do = 'lumi_left_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        lo_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        ll_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lo_dr = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_dr = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lo_do = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_do = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        ll_dr = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        ll_do = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &

                  # Next check that the luminance peak responses are higher than the persistent response: early (b) and early post (d) are higher than late (c).
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > traces_df[f'{lr_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > traces_df[f'{lr_do}_resp_{c}'].astype(float)) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &

                  # Next check that the luminance peak responses are higher than the persistent response: early (b) and early post (d) are higher than pre (a) and late post(e).
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > traces_df[f'{lr_do}_resp_{e}'].astype(float))
                  )
    return traces_df[good_cells]

def logic_regression_left_lumi(traces_df, thresh_resp=0.2, thresh_below=0.9, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify leftward luminance integrators based on the full stimulus set of 9 stimuli: lumi_[off/left/right]_dots_[off/left/right].
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance c.
    :param thresh_below: Threshold in ratio that the activity should be below during decrease in activity.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only leftward luminance integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    ll_dl = 'lumi_left_dots_left'
    lr_dl = 'lumi_right_dots_left'
    ll_dr = 'lumi_left_dots_right'
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    lr_do = 'lumi_right_dots_off'
    lo_dl = 'lumi_off_dots_left'
    lo_dr = 'lumi_off_dots_right'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        ll_dl = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_dl = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        ll_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        ll_do = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_do = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lo_dl = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lo_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{a}'].astype( float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &

                  # Next check that the decreased responses are low enough: late stimulus (c) < ratio_threshold * pre (a) and late post (e).
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_do}_resp_{e}'].astype(float)) &

                  # Next check for temporal integration: late (c) higher than early (b) stimulus.
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{b}'].astype(float))

                  )

    return traces_df[good_cells]

def logic_regression_left_diff(traces_df, thresh_resp=0.2, thresh_peaks=1.25, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify leftward luminance change detectors based on the full stimulus set of 9 stimuli: lumi_[off/left/right]_dots_[off/left/right].
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance b.
    :param thresh_peaks: Threshold in ratio of how much higher the strong contrast peaks should be then the weak contrast peaks.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only leftward luminance change detectors.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    ll_dl = 'lumi_left_dots_left'
    lr_dl = 'lumi_right_dots_left'
    ll_dr = 'lumi_left_dots_right'
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    lr_do = 'lumi_right_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        ll_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        ll_dr = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_dr = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        ll_do = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_do = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > thresh_resp) &

                  # Next check that the response during early stimulus (b) is higher than pre (a) and late stimulus (c) stimulus.
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > traces_df[f'{lr_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > traces_df[f'{lr_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) > traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) > traces_df[f'{lr_dl}_resp_{c}'].astype(float)) &

                  # Next check that the response during early post stimulus (d) is higher than during late (c) and late post (e) stimulus.
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) > traces_df[f'{lr_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) > traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check that the weak contrast responses early (b) and early post (d) are higher than pre (a) and late (c)
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) > traces_df[f'{ll_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) > traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) > traces_df[f'{ll_dr}_resp_{c}'].astype(float)) &

                  # Next check that the high contrast peak is higher than ratio_threshold * weak contrast peak.
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > thresh_peaks * traces_df[f'{ll_do}_resp_{b}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_left_bright(traces_df, thresh_resp=0.2, thresh_peaks=1.5, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify leftward luminance increase detectors based on the full stimulus set of 9 stimuli: lumi_[off/left/right]_dots_[off/left/right].
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance b.
    :param thresh_peaks: Threshold in ratio of how much higher the strong contrast peaks should be then the weak contrast peaks.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only leftward luminance increase detectors.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    ll_dl = 'lumi_left_dots_left'
    lr_dl = 'lumi_right_dots_left'
    ll_dr = 'lumi_left_dots_right'
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    lr_do = 'lumi_right_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        ll_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        ll_dr = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_dr = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        ll_do = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_do = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > thresh_resp) &

                  # Next check that the response during early stimulus (b) is higher than pre (a), late (c), early post (d) and late post (e) stimulus.
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > traces_df[f'{lr_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > traces_df[f'{lr_do}_resp_{d}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) > traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) > traces_df[f'{lr_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) > traces_df[f'{lr_dl}_resp_{d}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) > traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{d}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: early stimulus (b) < threshold + pre (a)
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &

                  # Next check that the weak contrast responses early post (d) is higher than late (c)
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) > traces_df[f'{ll_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) > traces_df[f'{ll_dr}_resp_{c}'].astype(float)) &

                  # Next check that the high contrast peak is higher than ratio_threshold * weak contrast peak.
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > thresh_peaks * traces_df[f'{ll_do}_resp_{d}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: early post stimulus (d) < threshold + pre (a)
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_left_dark(traces_df, thresh_resp=0.2, thresh_peaks=1.5, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify leftward luminance decrease detectors based on the full stimulus set of 9 stimuli: lumi_[off/left/right]_dots_[off/left/right].
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance d.
    :param thresh_peaks: Threshold in ratio of how much higher the strong contrast peaks should be then the weak contrast peaks.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only leftward luminance decrease detectors.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    ll_dl = 'lumi_left_dots_left'
    lr_dl = 'lumi_right_dots_left'
    ll_dr = 'lumi_left_dots_right'
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    lr_do = 'lumi_right_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        ll_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        ll_dr = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_dr = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        ll_do = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_do = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > thresh_resp) &

                  # Next check that the response during early post stimulus (d) is higher than pre (a), early (b), late (c) and late post (e) stimulus.
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > traces_df[f'{lr_do}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > traces_df[f'{lr_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) > traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) > traces_df[f'{lr_dl}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) > traces_df[f'{lr_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) > traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: early post stimulus (d) < threshold + pre (a)
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &

                  # Next check that the weak contrast responses early (b) is higher than pre (a)
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) > traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &

                  # Next check that the high contrast peak is higher than ratio_threshold * weak contrast peak.
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > thresh_peaks * traces_df[f'{ll_do}_resp_{b}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: early stimulus (b) < threshold + pre (a)
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_right_motion_3stim(traces_df, thresh_resp=0.2, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify rightward motion integrators based on a minimal stimulus set of 3 stimuli: lumi_off_dots_left, lumi_right_dots_right, and lumi_left_dots_off.
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during motion c.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only rightward motion integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lo_dl = 'lumi_off_dots_left'
    ll_do = 'lumi_left_dots_off'
    lr_dr = 'lumi_right_dots_right'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        lo_dl = np.random.choice(
            ['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        ll_do = np.random.choice(
            ['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        lr_dr = np.random.choice(
            ['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second statement checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check for temporal integration: late (c) higher than early (b) stimulus.
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{b}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &

                  # Next check that there is no post-motion response to the contralateral stimulus: early post (d) < threshold + late post (e).
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_right_drive_3stim(traces_df, thresh_resp=0.2, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify rightward multifeature integrators based on a minimal stimulus set of 3 stimuli: lumi_off_dots_left, lumi_right_dots_right, and lumi_left_dots_off.
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during motion c.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only rightward multifeature integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        ll_do = np.random.choice(
            ['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        lr_dr = np.random.choice(
            ['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second statement checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check that the luminance peak responses are higher than the persistent response: early (b) and early post (d) are higher than late (c).
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_right_lumi_3stim(traces_df, thresh_resp=0.2, thresh_below=0.9, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify rightward luminance integrators based on a minimal stimulus set of 3 stimuli: lumi_off_dots_left, lumi_right_dots_right, and lumi_left_dots_off.
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance c.
    :param thresh_below: Threshold in ratio that the activity should be below during decrease in activity.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only rightward luminance integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    lo_dl = 'lumi_off_dots_left'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        lr_dr = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        ll_do = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        lo_dl = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second statement checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &

                  # Next check that the decreased responses are low enough: late stimulus (c) < ratio_threshold * pre (a) and late post (e).
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_do}_resp_{e}'].astype(float)) &

                  # Next check for temporal integration: late (c) higher than early (b) stimulus.
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{b}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_right_diff_3stim(traces_df, thresh_resp=0.2, thresh_peaks=1.25, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify rightward luminance change detectors based on a minimal stimulus set of 3 stimuli: lumi_off_dots_left, lumi_right_dots_right, and lumi_left_dots_off.
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance b.
    :param thresh_peaks: Threshold in ratio of how much higher the strong contrast peaks should be then the weak contrast peaks.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only rightward luminance change detectors.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        lr_dr = np.random.choice(
            ['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        ll_do = np.random.choice(
            ['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second statement checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > thresh_resp) &

                  # Next check that the response during early stimulus (b) is higher than pre (a) and late stimulus (c) stimulus.
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &

                  # Next check that the response during early post stimulus (d) is higher than late (c) and late post (e) stimulus.
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &

                  # Next check that the weak contrast responses early (b) and early post (d) are higher than pre (a) and late (c)
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &

                  # Next check that the high contrast peak is higher than ratio_threshold * weak contrast peak.
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > thresh_peaks * traces_df[f'{lr_dr}_resp_{b}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_right_bright_3stim(traces_df, thresh_resp=0.2, thresh_peaks=1.5, thresh_min=0.1,
                                       shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify rightward luminance increase detectors based on a minimal stimulus set of 3 stimuli: lumi_off_dots_left, lumi_right_dots_right, and lumi_left_dots_off.
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance b.
    :param thresh_peaks: Threshold in ratio of how much higher the strong contrast peaks should be then the weak contrast peaks.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only rightward luminance increase detectors.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        lr_dr = np.random.choice(
            ['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        ll_do = np.random.choice(
            ['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second statement checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > thresh_resp) &

                  # Next check that the response during early stimulus (b) is higher than pre (a), late (c), early post (d) and late post (e) stimulus.
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{d}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: early stimulus (b) < threshold + pre (a)
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &

                  # Next check that the weak contrast responses early post (d) is higher than late (c)
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &

                  # Next check that the high contrast peak is higher than ratio_threshold * weak contrast peak.
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > thresh_peaks * traces_df[f'{lr_dr}_resp_{d}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: early post stimulus (d) < threshold + pre (a)
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype( float)))

    return traces_df[good_cells]

def logic_regression_right_dark_3stim(traces_df, thresh_resp=0.2, thresh_peaks=1.5, thresh_min=0.1,
                                     shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify rightward luminance decrease detectors based on a minimal stimulus set of 3 stimuli: lumi_off_dots_left, lumi_right_dots_right, and lumi_left_dots_off.
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance d.
    :param thresh_peaks: Threshold in ratio of how much higher the strong contrast peaks should be then the weak contrast peaks.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only rightward luminance decrease detectors.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        lr_dr = np.random.choice(
            ['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        ll_do = np.random.choice(
            ['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second statement checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > thresh_resp) &

                  # Next check that the response during early post stimulus (d) is higher than pre (a), early (b), late (c) and late post (e) stimulus.
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: early post stimulus (d) < threshold + pre (a)
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &

                  # Next check that the weak contrast responses early (b) is higher than pre (a)
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &

                  # Next check that the high contrast peak is higher than ratio_threshold * weak contrast peak.
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > thresh_peaks * traces_df[f'{lr_dr}_resp_{b}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: early stimulus (b) < threshold + pre (a)
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_right_motion(traces_df, thresh_resp=0.2, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify rightward motion integrators based on the full stimulus set of 9 stimuli: lumi_[off/left/right]_dots_[off/left/right].
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during motion c.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only rightward motion integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lo_dr = 'lumi_off_dots_right'
    ll_dr = 'lumi_left_dots_right'
    lr_dr = 'lumi_right_dots_right'
    ll_dl = 'lumi_left_dots_left'
    lr_dl = 'lumi_right_dots_left'
    ll_do = 'lumi_left_dots_off'
    lr_do = 'lumi_right_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        lo_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        ll_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        ll_do = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_do = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check for temporal integration: late (c) higher than early (b) stimulus.
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{b}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &

                  # Next check that there is no post-motion response to the contralateral stimulus: early post (d) < threshold + late post (e).
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_right_drive(traces_df, thresh_resp=0.2, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify rightward multifeature integrators based on the full stimulus set of 9 stimuli: lumi_[off/left/right]_dots_[off/left/right].
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during motion c.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only rightward multifeature integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lo_dl = 'lumi_off_dots_left'
    ll_dl = 'lumi_left_dots_left'
    lr_dl = 'lumi_right_dots_left'
    lo_dr = 'lumi_off_dots_right'
    lr_dr = 'lumi_right_dots_right'
    lo_do = 'lumi_off_dots_off'
    lr_do = 'lumi_right_dots_off'
    ll_dr = 'lumi_left_dots_right'
    ll_do = 'lumi_left_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        lo_dl = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        ll_dl = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_dl = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lo_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lo_do = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_do = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        ll_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        ll_do = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check that the luminance peak responses are higher than the persistent response: early (b) and early post (d) are higher than late (c).
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > traces_df[f'{ll_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) > traces_df[f'{ll_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{e}'].astype(float)) &

                  # Next check that the luminance peak responses are higher than the persistent response: early (b) and early post (d) are higher than pre (a) and late post(e).
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float))
                  )
    return traces_df[good_cells]

def logic_regression_right_lumi(traces_df, thresh_resp=0.2, thresh_below=0.9, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify rightward luminance integrators based on the full stimulus set of 9 stimuli: lumi_[off/left/right]_dots_[off/left/right].
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance c.
    :param thresh_below: Threshold in ratio that the activity should be below during decrease in activity.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only rightward luminance integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    ll_dl = 'lumi_left_dots_left'
    lr_dl = 'lumi_right_dots_left'
    ll_dr = 'lumi_left_dots_right'
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    lr_do = 'lumi_right_dots_off'
    lo_dl = 'lumi_off_dots_left'
    lo_dr = 'lumi_off_dots_right'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        ll_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        ll_dr = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_dr = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        ll_do = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_do = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lo_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lo_dr = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &

                  # Next check that the decreased responses are low enough: late stimulus (c) < ratio_threshold * pre (a) and late post (e).
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_do}_resp_{e}'].astype(float)) &

                  # Next check for temporal integration: late (c) higher than early (b) stimulus.
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{b}'].astype(float))

                  )

    return traces_df[good_cells]

def logic_regression_right_diff(traces_df, thresh_resp=0.2, thresh_peaks=1.25, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify rightward luminance change detectors based on the full stimulus set of 9 stimuli: lumi_[off/left/right]_dots_[off/left/right].
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance b.
    :param thresh_peaks: Threshold in ratio of how much higher the strong contrast peaks should be then the weak contrast peaks.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only rightward luminance change detectors.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    ll_dl = 'lumi_left_dots_left'
    lr_dl = 'lumi_right_dots_left'
    ll_dr = 'lumi_left_dots_right'
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    lr_do = 'lumi_right_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        ll_dl = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_dl = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        ll_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        ll_do = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_do = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > thresh_resp) &

                  # Next check that the response during early stimulus (b) is higher than pre (a) and late stimulus (c) stimulus.
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > traces_df[f'{ll_dl}_resp_{c}'].astype(float)) &

                  # Next check that the response during early post stimulus (d) is higher than late (c) and late post (e) stimulus.
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) > traces_df[f'{ll_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) > traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) > traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) > traces_df[f'{ll_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) > traces_df[f'{ll_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) > traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &

                  # Next check that the weak contrast responses early (b) and early post (d) are higher than pre (a) and late (c)
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > traces_df[f'{lr_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) > traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) > traces_df[f'{lr_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &

                  # Next check that the high contrast peak is higher than ratio_threshold * weak contrast peak.
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > thresh_peaks * traces_df[f'{lr_do}_resp_{b}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_right_bright(traces_df, thresh_resp=0.2, thresh_peaks=1.5, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify rightward luminance increase detectors based on the full stimulus set of 9 stimuli: lumi_[off/left/right]_dots_[off/left/right].
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance b.
    :param thresh_peaks: Threshold in ratio of how much higher the strong contrast peaks should be then the weak contrast peaks.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only rightward luminance increase detectors.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    ll_dl = 'lumi_left_dots_left'
    lr_dl = 'lumi_right_dots_left'
    ll_dr = 'lumi_left_dots_right'
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    lr_do = 'lumi_right_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        ll_dl = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_dl = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        ll_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        ll_do = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_do = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > thresh_resp) &

                  # Next check that the response during early stimulus (b) is higher than pre (a), late (c), early post (d) and late post (e) stimulus.
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{d}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > traces_df[f'{ll_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > traces_df[f'{ll_dl}_resp_{d}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) > traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) > traces_df[f'{ll_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) > traces_df[f'{ll_dr}_resp_{d}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) > traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: early stimulus (b) < threshold + pre (a)
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &

                  # Next check that the weak contrast responses early post (d) is higher than pre (c)
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > traces_df[f'{lr_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) > traces_df[f'{lr_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &

                  # Next check that the high contrast peak is higher than ratio_threshold * weak contrast peak.
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > thresh_peaks * traces_df[f'{lr_do}_resp_{d}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: early post stimulus (d) < threshold + pre (a)
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_right_dark(traces_df, thresh_resp=0.2, thresh_peaks=1.5, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify rightward luminance decrease detectors based on the full stimulus set of 9 stimuli: lumi_[off/left/right]_dots_[off/left/right].
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance d.
    :param thresh_peaks: Threshold in ratio of how much higher the strong contrast peaks should be then the weak contrast peaks.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only rightward luminance decrease detectors.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    ll_dl = 'lumi_left_dots_left'
    lr_dl = 'lumi_right_dots_left'
    ll_dr = 'lumi_left_dots_right'
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    lr_do = 'lumi_right_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        ll_dl = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_dl = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        ll_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        ll_do = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_do = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) > thresh_resp) &

                  # Next check that the response during early post stimulus (d) is higher than pre (a), early (b), late (c) and late post (e) stimulus.
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) > traces_df[f'{ll_dl}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) > traces_df[f'{ll_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) > traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) > traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) > traces_df[f'{ll_dr}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) > traces_df[f'{ll_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) > traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: early post stimulus (d) < threshold + pre (a)
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &

                  # Next check that the weak contrast responses early post (d) is higher than pre (c)
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) > traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &

                  # Next check that the high contrast peak is higher than ratio_threshold * weak contrast peak.
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > thresh_peaks * traces_df[f'{lr_do}_resp_{b}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: early stimulus (b) < threshold + pre (a)
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_left_motion_wta(traces_df, thresh_resp=0.2, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify leftward WTA motion integrators based on the full stimulus set of 9 stimuli: lumi_[off/left/right]_dots_[off/left/right].
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during motion c.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only leftward WTA motion integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lo_dl = 'lumi_off_dots_left'
    ll_dl = 'lumi_left_dots_left'
    lr_dl = 'lumi_right_dots_left'
    ll_dr = 'lumi_left_dots_right'
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    lr_do = 'lumi_right_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        lo_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        ll_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        ll_do = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_do = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &

                  # Next check for temporal integration: late (c) higher than early (b) stimulus.
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{b}'].astype(float)) &

                  # Next check that the WTA non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check that there is no post-motion response to the contralateral stimulus: early post (d) < threshold + late post (e).
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{e}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_right_motion_wta(traces_df, thresh_resp=0.2, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify rightward WTA motion integrators based on the full stimulus set of 9 stimuli: lumi_[off/left/right]_dots_[off/left/right].
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during motion c.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only rightward WTA motion integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    lo_dr = 'lumi_off_dots_right'
    ll_dr = 'lumi_left_dots_right'
    lr_dr = 'lumi_right_dots_right'
    ll_dl = 'lumi_left_dots_left'
    lr_dl = 'lumi_right_dots_left'
    ll_do = 'lumi_left_dots_off'
    lr_do = 'lumi_right_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        lo_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        ll_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        ll_do = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_do = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check for temporal integration: late (c) higher than early (b) stimulus.
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{b}'].astype(float)) &

                  # Next check that the WTA non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &

                  # Next check that there is no post-motion response to the contralateral stimulus: early post (d) < threshold + late post (e).
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float))
                  )

    return traces_df[good_cells]
def logic_regression_left_lumi_wta(traces_df, thresh_resp=0.2, thresh_below=0.9, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify leftward WTA luminance integrators based on a minimal stimulus set of 3 stimuli: lumi_off_dots_left, lumi_right_dots_right, and lumi_left_dots_off.
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance c.
    :param thresh_below: Threshold in ratio that the activity should be below during decrease in activity.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only leftward WTA luminance integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    ll_dl = 'lumi_left_dots_left'
    lr_dl = 'lumi_right_dots_left'
    ll_dr = 'lumi_left_dots_right'
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    lr_do = 'lumi_right_dots_off'
    lo_dl = 'lumi_off_dots_left'
    lo_dr = 'lumi_off_dots_right'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        ll_dl = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_dl = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        ll_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        ll_do = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lr_do = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lo_dl = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        lo_dr = np.random.choice(
            ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
             'lumi_off_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &

                  # Next check that the WTA non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &

                  # Next check that the decreased responses are low enough: late stimulus (c) < ratio_threshold * pre (a) and late post (e).
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_do}_resp_{e}'].astype(float)) &

                  # Next check for temporal integration: late (c) higher than early (b) stimulus.
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{b}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_right_lumi_wta(traces_df, thresh_resp=0.2, thresh_below=0.9, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify rightward WTA luminance integrators based on the full stimulus set of 9 stimuli: lumi_[off/left/right]_dots_[off/left/right].
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance c.
    :param thresh_below: Threshold in ratio that the activity should be below during decrease in activity.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only rightward WTA luminance integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    ll_dl = 'lumi_left_dots_left'
    lr_dl = 'lumi_right_dots_left'
    ll_dr = 'lumi_left_dots_right'
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    lr_do = 'lumi_right_dots_off'
    lo_dl = 'lumi_off_dots_left'
    lo_dr = 'lumi_off_dots_right'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        ll_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        ll_dr = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_dr = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        ll_do = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lr_do = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lo_dl = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        lo_dr = np.random.choice(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &

                  # Next check that the WTA non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &

                  # Next check that the decreased responses are low enough: late stimulus (c) < ratio_threshold * pre (a) and late post (e).
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_do}_resp_{e}'].astype(float)) &

                  # Next check for temporal integration: late (c) higher than early (b) stimulus.
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{b}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_left_lumi_single(traces_df, thresh_resp=0.2, thresh_below=0.9, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify leftward luminance integrators based on the luminance integrator experiment stimulus set: lumi_[left/right]_weak_dots_off and lumi_off_dots_[left/right].
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance c.
    :param thresh_below: Threshold in ratio that the activity should be below during decrease in activity.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only leftward luminance integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    ll_do = 'lumi_left_weak_dots_off'
    lr_do = 'lumi_right_weak_dots_off'
    lo_dl = 'lumi_off_dots_left'
    lo_dr = 'lumi_off_dots_right'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        ll_do = np.random.choice(
            ['lumi_left_weak_dots_off',  'lumi_right_weak_dots_off', 'lumi_off_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right'])
        lr_do = np.random.choice(
            ['lumi_left_weak_dots_off',  'lumi_right_weak_dots_off', 'lumi_off_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right'])
        lo_dl = np.random.choice(
            ['lumi_left_weak_dots_off',  'lumi_right_weak_dots_off', 'lumi_off_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right'])
        lo_dr = np.random.choice(
            ['lumi_left_weak_dots_off',  'lumi_right_weak_dots_off', 'lumi_off_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{a}'].astype( float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &

                  # Next check that the decreased responses are low enough: late stimulus (c) < ratio_threshold * pre (a) and late post (e).
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_do}_resp_{e}'].astype(float)) &

                  # Next check for temporal integration: late (c) higher than early (b) stimulus.
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{b}'].astype(float))

                  )

    return traces_df[good_cells]

def logic_regression_right_lumi_single(traces_df, thresh_resp=0.2, thresh_below=0.9, thresh_min=0.1, shuffle_stim_idx=False):
    '''
    This function contains the logical statements to identify rightward WTA luminance integrators based on the full stimulus set of 9 stimuli: lumi_[off/left/right]_dots_[off/left/right].
    :param traces_df: The dataframe containing for each neuron the average functional activity including the average response pre-stimulus (a), early during stimulus (b), late during stimulus (c), early post-stimulus (d), late post-stimulus (e).
    :param thresh_resp: Threshold of minimum response in dF/F0 during luminance c.
    :param thresh_below: Threshold in ratio that the activity should be below during decrease in activity.
    :param thresh_min: Threshold in dF/F0 that the activity cannot cross during stimuli where no activity is expected.
    :param shuffle_stim_idx: If TRUE the stimulus indexes and responses (a-e) are randomly picked (useful as shuffled control).
    :return: subset of traces_df containing only rightward WTA luminance integrators.
    '''
    # Create stimulus name shorthands to increase readability and to make stimulus shuffling easier.
    ll_do = 'lumi_left_weak_dots_off'
    lr_do = 'lumi_right_weak_dots_off'
    lo_dl = 'lumi_off_dots_left'
    lo_dr = 'lumi_off_dots_right'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    # If shuffle_stim_idx is True, shuffle the stimuli and response idxes.
    if shuffle_stim_idx:
        ll_do = np.random.choice(
            ['lumi_left_weak_dots_off',  'lumi_right_weak_dots_off', 'lumi_off_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right'])
        lr_do = np.random.choice(
            ['lumi_left_weak_dots_off',  'lumi_right_weak_dots_off', 'lumi_off_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right'])
        lo_dl = np.random.choice(
            ['lumi_left_weak_dots_off',  'lumi_right_weak_dots_off', 'lumi_off_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right'])
        lo_dr = np.random.choice(
            ['lumi_left_weak_dots_off',  'lumi_right_weak_dots_off', 'lumi_off_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    # Select the good cells based on a list of logical statements. The first statement sanity checks that the neuron is in the brain,
    # the second block of statements checks that the minimal activity level during the relevant stimulus is met.
    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > thresh_resp) &

                  # Next check that the response during late stimulus (c) is higher than pre (a) and late post (e) stimulus.
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{e}'].astype(float)) &

                  # Next check that the non-responses are actually non-responses: late stimulus (c) < threshold + pre (a) and threshold + post(e)
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &

                  # Next check that the decreased responses are low enough: late stimulus (c) < ratio_threshold * pre (a) and late post (e).
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_do}_resp_{e}'].astype(float)) &

                  # Next check for temporal integration: late (c) higher than early (b) stimulus.
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{b}'].astype(float))

                  )

    return traces_df[good_cells]
