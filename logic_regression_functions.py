import numpy as np

def logic_regression_left_motion_3stim(traces_df, thresh_resp=0.2, thresh_min=0.1, shuffle_stim_idx=False):
    lo_dl = 'lumi_off_dots_left'
    ll_do = 'lumi_left_dots_off'
    lr_dr = 'lumi_right_dots_right'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

    if shuffle_stim_idx:
        lo_dl = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        ll_do = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        lr_dr = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{b}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{e}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_left_drive_3stim(traces_df, thresh_resp=0.2, shuffle_stim_idx=False):
    lo_dl = 'lumi_off_dots_left'
    lr_dr = 'lumi_right_dots_right'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'


    if shuffle_stim_idx:
        lo_dl = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        lr_dr = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_left_lumi_3stim(traces_df, thresh_resp=0.2, thresh_below=0.9, thresh_min=0.1,
                                     shuffle_stim_idx=False):
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    lo_dl = 'lumi_off_dots_left'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{b}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_left_diff_3stim(traces_df, thresh_resp=0.2, thresh_peaks=1.25, shuffle_stim_idx=False):
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'


    if shuffle_stim_idx:
        lr_dr = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        ll_do = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &

                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > thresh_peaks * traces_df[f'{ll_do}_resp_{b}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_left_bright_3stim(traces_df, thresh_resp=0.2, thresh_peaks=1.5, thresh_min=0.1, shuffle_stim_idx=False):
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'


    if shuffle_stim_idx:
        lr_dr = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        ll_do = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{d}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &

                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > thresh_peaks * traces_df[f'{ll_do}_resp_{d}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_left_dark_3stim(traces_df, thresh_resp=0.2, thresh_peaks=1.5, thresh_min=0.1, shuffle_stim_idx=False):
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'


    if shuffle_stim_idx:
        lr_dr = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        ll_do = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &

                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > thresh_peaks * traces_df[f'{ll_do}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)))

    return traces_df[good_cells]


def logic_regression_left_motion(traces_df, thresh_resp=0.2, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{b}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{e}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_left_drive(traces_df, thresh_resp=0.2, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > traces_df[f'{lr_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > traces_df[f'{lr_do}_resp_{c}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > traces_df[f'{lr_do}_resp_{e}'].astype(float))
                  )
    return traces_df[good_cells]

def logic_regression_left_lumi(traces_df, thresh_resp=0.2, thresh_below=0.9, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{a}'].astype( float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_do}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{b}'].astype(float))

                  )

    return traces_df[good_cells]

def logic_regression_left_diff(traces_df, thresh_resp=0.2, thresh_peaks=1.25, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > traces_df[f'{lr_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > traces_df[f'{lr_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) > traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) > traces_df[f'{lr_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) > traces_df[f'{lr_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) > traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) > traces_df[f'{ll_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) > traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) > traces_df[f'{ll_dr}_resp_{c}'].astype(float)) &

                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > thresh_peaks * traces_df[f'{ll_do}_resp_{b}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_left_bright(traces_df, thresh_resp=0.2, thresh_peaks=1.5, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > thresh_resp) &

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

                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) > traces_df[f'{ll_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) > traces_df[f'{ll_dr}_resp_{c}'].astype(float)) &

                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > thresh_peaks * traces_df[f'{ll_do}_resp_{d}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_left_dark(traces_df, thresh_resp=0.2, thresh_peaks=1.5, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > thresh_resp) &

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

                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) > traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &

                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > thresh_peaks * traces_df[f'{ll_do}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_right_motion_3stim(traces_df, thresh_resp=0.2, thresh_min=0.1, shuffle_stim_idx=False):
    lo_dl = 'lumi_off_dots_left'
    ll_do = 'lumi_left_dots_off'
    lr_dr = 'lumi_right_dots_right'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{b}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_right_drive_3stim(traces_df, thresh_resp=0.2, shuffle_stim_idx=False):
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_right_lumi_3stim(traces_df, thresh_resp=0.2, thresh_below=0.9, thresh_min=0.1, shuffle_stim_idx=False):
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    lo_dl = 'lumi_off_dots_left'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'


    if shuffle_stim_idx:
        lr_dr = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        ll_do = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        lo_dl = np.random.choice(['lumi_off_dots_left', 'lumi_right_dots_right', 'lumi_left_dots_off'])
        a = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        b = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        c = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        d = np.random.choice(['a', 'b', 'c', 'd', 'e'])
        e = np.random.choice(['a', 'b', 'c', 'd', 'e'])

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{b}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_right_diff_3stim(traces_df, thresh_resp=0.2, thresh_peaks=1.25, shuffle_stim_idx=False):
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > thresh_resp) &

                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > thresh_peaks * traces_df[f'{lr_dr}_resp_{b}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_right_bright_3stim(traces_df, thresh_resp=0.2, thresh_peaks=1.5, thresh_min=0.1,
                                       shuffle_stim_idx=False):
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > thresh_resp) &

                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{d}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > thresh_peaks * traces_df[f'{lr_dr}_resp_{d}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype( float)))

    return traces_df[good_cells]

def logic_regression_right_dark_3stim(traces_df, thresh_resp=0.2, thresh_peaks=1.5, thresh_min=0.1,
                                     shuffle_stim_idx=False):
    lr_dr = 'lumi_right_dots_right'
    ll_do = 'lumi_left_dots_off'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > thresh_resp) &

                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > thresh_peaks * traces_df[f'{lr_dr}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_right_motion(traces_df, thresh_resp=0.2, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{b}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_right_drive(traces_df, thresh_resp=0.2, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > traces_df[f'{ll_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) > traces_df[f'{ll_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &

                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float))
                  )
    return traces_df[good_cells]

def logic_regression_right_lumi(traces_df, thresh_resp=0.2, thresh_below=0.9, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_do}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{b}'].astype(float))

                  )

    return traces_df[good_cells]

def logic_regression_right_diff(traces_df, thresh_resp=0.2, thresh_peaks=1.25, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > thresh_resp) &

                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > traces_df[f'{ll_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) > traces_df[f'{ll_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) > traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) > traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) > traces_df[f'{ll_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) > traces_df[f'{ll_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) > traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > traces_df[f'{lr_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) > traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) > traces_df[f'{lr_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > thresh_peaks * traces_df[f'{lr_do}_resp_{b}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_right_bright(traces_df, thresh_resp=0.2, thresh_peaks=1.5, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dr}_resp_{b}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > thresh_resp) &

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

                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > traces_df[f'{lr_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) > traces_df[f'{lr_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > thresh_peaks * traces_df[f'{lr_do}_resp_{d}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_right_dark(traces_df, thresh_resp=0.2, thresh_peaks=1.5, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) > thresh_resp) &

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

                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{b}'].astype(float) > traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > thresh_peaks * traces_df[f'{lr_do}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)))

    return traces_df[good_cells]

def logic_regression_left_motion_wta(traces_df, thresh_resp=0.2, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{b}'].astype(float)) &

                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{e}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_right_motion_wta(traces_df, thresh_resp=0.2, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{b}'].astype(float)) &

                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float))
                  )

    return traces_df[good_cells]
def logic_regression_left_lumi_wta(traces_df, thresh_resp=0.2, thresh_below=0.9, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_do}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{b}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_right_lumi_wta(traces_df, thresh_resp=0.2, thresh_below=0.9, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_do}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{b}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_left_drive_wta(traces_df, thresh_resp=0.2, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) > traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_dr}_resp_{b}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) > traces_df[f'{lr_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{b}'].astype(float) > traces_df[f'{lr_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) > traces_df[f'{lr_do}_resp_{c}'].astype(float))
                  )
    return traces_df[good_cells]

def logic_regression_right_drive_wta(traces_df, thresh_resp=0.2, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) > traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_dl}_resp_{b}'].astype(float) > traces_df[f'{ll_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) > traces_df[f'{ll_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{b}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) > traces_df[f'{ll_do}_resp_{c}'].astype(float))
                  )
    return traces_df[good_cells]

def logic_regression_left_motion_wta_ctrl(traces_df, thresh_resp=0.2, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) > traces_df[f'{lo_dl}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{b}'].astype(float)) &

                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_dr}_resp_{e}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_right_motion_wta_ctrl(traces_df, thresh_resp=0.2, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) > traces_df[f'{lo_dr}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{b}'].astype(float)) &

                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{d}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{e}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_left_lumi_wta_ctrl(traces_df, thresh_resp=0.2, thresh_below=0.9, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lr_do}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) > traces_df[f'{ll_do}_resp_{b}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) > traces_df[f'{ll_dr}_resp_{b}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_right_lumi_wta_ctrl(traces_df, thresh_resp=0.2, thresh_below=0.9, thresh_min=0.1, shuffle_stim_idx=False):
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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dl}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{ll_dl}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lr_dr}_resp_{c}'].astype(float) < thresh_min + traces_df[f'{lr_dr}_resp_{e}'].astype(float)) &

                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_do}_resp_{a}'].astype(float)) &
                  (traces_df[f'{ll_do}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{ll_do}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lr_do}_resp_{c}'].astype(float) > traces_df[f'{lr_do}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lr_dl}_resp_{c}'].astype(float) > traces_df[f'{lr_dl}_resp_{b}'].astype(float))
                  )

    return traces_df[good_cells]

def logic_regression_left_motion_detector(traces_df, thresh_resp=0.2, thresh_below=0.9, shuffle_stim_idx=False):
        lo_dl = 'lumi_off_dots_left'
        lo_dr = 'lumi_off_dots_right'
        a = 'a'
        b = 'b'
        c = 'c'
        d = 'd'
        e = 'e'

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

        good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                      (traces_df[f'{lo_dl}_resp_{b}'].astype(float) > thresh_resp) &
                      (traces_df[f'{lo_dr}_resp_{d}'].astype(float) > thresh_resp) &

                      (traces_df[f'{lo_dl}_resp_{b}'].astype(float) > traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                      (traces_df[f'{lo_dl}_resp_{b}'].astype(float) > traces_df[f'{lo_dl}_resp_{c}'].astype(float)) &
                      (traces_df[f'{lo_dl}_resp_{b}'].astype(float) > traces_df[f'{lo_dl}_resp_{d}'].astype(float)) &
                      (traces_df[f'{lo_dl}_resp_{b}'].astype(float) > traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &
                      (traces_df[f'{lo_dr}_resp_{d}'].astype(float) > traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                      (traces_df[f'{lo_dr}_resp_{d}'].astype(float) > traces_df[f'{lo_dr}_resp_{b}'].astype(float)) &
                      (traces_df[f'{lo_dr}_resp_{d}'].astype(float) > traces_df[f'{lo_dr}_resp_{c}'].astype(float)) &
                      (traces_df[f'{lo_dr}_resp_{d}'].astype(float) > traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &

                      (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                      (traces_df[f'{lo_dr}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lo_dr}_resp_{e}'].astype(float)))

        return traces_df[good_cells]

def logic_regression_right_motion_detector(traces_df, thresh_resp=0.2, thresh_below=0.9,
                                          shuffle_stim_idx=False):
    lo_dl = 'lumi_off_dots_left'
    lo_dr = 'lumi_off_dots_right'
    a = 'a'
    b = 'b'
    c = 'c'
    d = 'd'
    e = 'e'

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

    good_cells = ((traces_df['ZB_z'].astype(float) > 0) &
                  (traces_df[f'{lo_dr}_resp_{b}'].astype(float) > thresh_resp) &
                  (traces_df[f'{lo_dl}_resp_{d}'].astype(float) > thresh_resp) &

                  (traces_df[f'{lo_dr}_resp_{b}'].astype(float) > traces_df[f'{lo_dr}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{b}'].astype(float) > traces_df[f'{lo_dr}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{b}'].astype(float) > traces_df[f'{lo_dr}_resp_{d}'].astype(float)) &
                  (traces_df[f'{lo_dr}_resp_{b}'].astype(float) > traces_df[f'{lo_dr}_resp_{e}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{d}'].astype(float) > traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{d}'].astype(float) > traces_df[f'{lo_dl}_resp_{b}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{d}'].astype(float) > traces_df[f'{lo_dl}_resp_{c}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{d}'].astype(float) > traces_df[f'{lo_dl}_resp_{e}'].astype(float)) &

                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lo_dl}_resp_{a}'].astype(float)) &
                  (traces_df[f'{lo_dl}_resp_{c}'].astype(float) < thresh_below * traces_df[f'{lo_dl}_resp_{e}'].astype(float)))

    return traces_df[good_cells]
