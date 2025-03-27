import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter, argrelextrema
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import gaussian
from matplotlib.collections import LineCollection
from test_data_analysis.read_neware_file import read_neware_xls
import os
import matplotlib as mpl
from test_data_analysis.tesla_half_cell import gauss_win_kf

# plt.style.use('chalmers_KF')
mpl.rc('lines', linewidth=0.8)


def find_ica_step(char_df, cell):
    '''

    :param char_df:
    :param cell:
    :return:
    '''
    ica_step_list = []
    for stp in char_df.step_nbr.unique():
        if abs(char_df[char_df.step_nbr == stp].maxV.iloc[0] - cell['Umax']) < 0.05 and \
                abs(char_df[char_df.step_nbr == stp].duration.iloc[0]) > 15*3600 and \
                abs(char_df[char_df.step_nbr == stp].curr.iloc[0])  < 0.3:
            ica_step_list.append(stp)
    return ica_step_list


def make_ica_dva_plots(df, name='Def'):
    '''

    :param df:
    :param name:
    :return:
    '''
    df_ica = df[df.curr < 0]
    df_ica.set_index(pd.to_timedelta(df_ica.float_time - df_ica.float_time.min(), unit='s'), inplace=True)

    if '30' in name or '50' in name:
        df_ica_ds = df_ica.resample('180s').mean()
        df_ica_ds['ica'] = np.gradient(df_ica_ds.mAh, df_ica_ds.volt) / 1000  # Compensate from mAh to Ah
        df_ica_ds['ica_filt'] = savgol_filter(df_ica_ds.ica, 35, 2)
    elif '100' in name:
        df_ica_ds = df_ica.resample('300s').mean()
        df_ica_ds['ica'] = np.gradient(df_ica_ds.mAh, df_ica_ds.volt) / 1000  # Compensate from mAh to Ah
        df_ica_ds['ica_filt'] = savgol_filter(df_ica_ds.ica, 55, 2)
    elif 'by3' in name:
        df_ica_ds = df_ica.resample('120s').mean()
        df_ica_ds['ica'] = np.gradient(df_ica_ds.mAh, df_ica_ds.volt) / 1000  # Compensate from mAh to Ah
        df_ica_ds['ica_filt'] = savgol_filter(df_ica_ds.ica, 15, 2)
    df_ica_ds['ica_max'] = df_ica_ds.iloc[argrelextrema(df_ica_ds.ica.values, np.greater, order=50)[0]]['ica_filt']

    fig1, ax1 = plt.subplots(2, 1, sharex=True)
    ax1[1].plot(df_ica_ds.volt, df_ica_ds.ica)
    ax1[1].plot(df_ica_ds.volt, df_ica_ds.ica_filt, linewidth=0.5)
    ax1[1].scatter(df_ica_ds.volt, df_ica_ds.ica_max, color='r', s=50, zorder=1)
    ax1[0].plot(df_ica_ds.volt, (df_ica_ds.mAh.max() - df_ica_ds.mAh) / 1000)
    ax1[0].set_title('Capacity v Voltage for {0}'.format(name))
    ax1[0].set_ylabel('Capacity [Ah]')
    ax1[1].set_xlabel('Voltage [V]')
    ax1[1].set_ylabel('ICA [d(Ah)/dV]')
    plt.ylim([0, 15])
    plt.xlim([3, 4.2])
    plt.tight_layout()
    # fig = plt.figure()
    # plt.plot(df_ica.mAh.max() - df_ica.mAh, df_ica.volt)

    df_ica_ds['dva'] = np.gradient(df_ica_ds.volt, df_ica_ds.mAh) * 1000
    df_ica_ds['dva_filt'] = savgol_filter(df_ica_ds.dva, 11, 2)
    df_ica_ds['dva_max'] = df_ica_ds.iloc[argrelextrema(df_ica_ds.dva.values, np.greater, order=50)[0]]['dva_filt']

    fig2, ax2 = plt.subplots(2, 1, sharex=True)
    xval = (df_ica_ds.mAh - df_ica_ds.mAh.min()) / max(df_ica_ds.mAh - df_ica_ds.mAh.min())
    # ax2[1].plot(xval, df_ica_ds.dva)
    ax2[0].plot(xval, df_ica_ds.volt)
    ax2[1].plot(xval, df_ica_ds.dva_filt, linewidth=0.7)
    ax2[1].scatter(xval, df_ica_ds.dva_max, color='r', s=15)
    ax2[0].set_title('SOC v Voltage for {0}'.format(name))
    ax2[0].set_ylabel('Voltage [V]')
    ax2[1].set_xlabel('SOC [-]')
    ax2[1].set_ylabel('DVA [dV/d(Ah)]')
    plt.ylim([0, 1])
    plt.tight_layout()
    return fig1, fig2, df_ica_ds


def calc_ica_dva(df):
    from scipy.signal import butter, filtfilt, savgol_filter
    '''
    
    :param df: Dataframe with input data on standard Neware format
    :return: 
    '''
    b, a = butter(5, 0.01, 'low')
    volt_butt = filtfilt(b, a, df.volt)
    volt_savgol = savgol_filter(df.volt, 25, 1)
    df.loc[:, 'volt_butt'] = volt_butt
    df.loc[:, 'volt_savgol'] = volt_savgol
    df.loc[:, 'ica'] = np.gradient(df.mAh/1000, volt_savgol)
    df.loc[:, 'dva'] = np.gradient(volt_savgol, df.mAh/1000)
    return df


def simplified_ica_dva(df):
    '''

    :param df:  DataFrame containing ica/dva measurement on standard neware format
    :return:
    '''
    df.loc[:, 'ica'] = np.gradient(df.cap/1000, df.volt)
    df.loc[:, 'dva'] = np.gradient(df.volt, df.cap/1000)
    return df


def large_span_ica_dva(df):
    '''
    Function that utilises larger span (five samples wide) to calculate the derivative
    :param df:
    :return:
    '''
    df.loc[:, 'ica'] = df.mAh.diff(5)/1000 / (df.volt.diff(5)*np.sign(df.curr))
    df.loc[:, 'dva'] = (df.volt.diff(5) * np.sign(df.curr)) / (df.mAh.diff(5) / 1000)
    return df


def smooth(a, WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation

    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(a[:WSZ - 1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def smooth_with_span(df, col='volt', prespan=0.0055):
    # df is dataframe with data to be smoothed
    smoothing_span = int(prespan * df.shape[0])
    # smoothing span must be odd
    smoothing_span = smoothing_span - 1 + smoothing_span % 2
    df.loc[:, col] = smooth(df[col], smoothing_span)
    return df


def gaussianfilter(df, prespan=0.0055, gausspan=0.03, gausswin=True):
    # Remove inconsistent data, ie where voltage is decreasing in charge or increasing in discharge
    df = df[(df['volt'].diff() * np.sign(df['curr'])).fillna(method='bfill') > 0]
    df = smooth_with_span(df, 'volt', prespan)
    df = smooth_with_span(df, 'mAh', prespan)
    Es = df['volt']
    dV = np.diff(Es * np.sign(df['curr']))
    Qs = (df['mAh'] - df['mAh'].min()) / 1000
    dQ = np.diff(Qs)
    # Es_conv, dqdv_conv = gaussianfilterconvolution(Es, -dQ / dV, len(Es) * gausspan)
    # Qs_conv, dvdq_conv = gaussianfilterconvolution(Qs, -dV / dQ, len(Qs) * gausspan)
    if gausswin:
        dQdV = gaussianfilterint(Es, dQ / dV, len(Es) * gausspan)
        dVdQ = gaussianfilterint(Qs, dV / dQ, len(Es) * gausspan)
    else:
        dQdV = dQ / dV
        dVdQ = dV / dQ
    df_out = pd.DataFrame({'volt': Es[1:],
                           'cap': Qs[1:],
                           'ica': dQdV,
                           'dva': dVdQ})
    return Es[1:], Qs[1:], dQdV, dVdQ


def gaussianfilterint(x, y, gausspan=0.03):
    ynan = pd.isna(y)
    span = len(x) * gausspan
    span = np.floor(span)
    n = len(y)
    span = min(span, n)
    width = span - 1 + span % 2
    xreps = any(np.diff(x) == 0)
    if width == 1 and not xreps and not any(ynan):
        c = y
    c = np.zeros_like(y)
    G = gauss_win_kf(width)
    j = 0
    h = int((width - 1) / 2)
    for k in range(h):
        c[k] = sum(G[(h - k):] * y[:(1 + h + k)]) / sum(G[(h - k):])
        j += 1
    for k in range(h, n - (h + 1)):
        c[k] = sum(G * y[k-h:k+h + 1]) / sum(G)
    for k in range(n - (h + 1), n):
        c[k] = sum(G[:(1 + h + j)] * y[k-h:]) / sum(G[:(h + j)])
        j -= 1
    return c


def gaussianfilterconvolution(x, y, span):
    ynan = np.isnan(y)
    span = np.floor(span)
    n = len(y)
    span = min(span, n)
    width = span - 1 + span % 2
    xreps = any(np.diff(x) == 0)
    if width == 1 and not xreps and not any(ynan):
        c = y
        print('Data not complete')
        return x, c
    G = gaussian(width, 3.8 * width / 10)
    # plt.plot(G)
    print('Current span = {} with size of gaussian window {}'.format(span, G.shape[0]))
    c = np.convolve(y, G, 'same') / sum(G)
    return x, c


def remove_faulty_points(df):
    """
    Function that removes noise points that have been incorrectly measured
    :param df: Dataframe containing ICA measurement
    :return:
    """
    # Split in two cases based on expected polarisation
    chrg_df = df[df.curr > 0]
    dchg_df = df[df.curr < 0]
    # Find points with faulty polarisation to be dropped for the charge part of IC test
    chrg_to_drop = np.array(chrg_df[chrg_df.volt.diff() <= 0].index)
    # Remove the next point after the faulty one as an extra measurement is triggered
    # by the faulty measurement.
    chrg_to_drop = np.append(chrg_to_drop, chrg_to_drop + 1)
    # Repeat process with inverted polarisation for the discharge case.
    dchg_to_drop = np.array(dchg_df[dchg_df.volt.diff() >= 0].index)
    dchg_to_drop = np.append(dchg_to_drop, dchg_to_drop + 1)
    # Concatenate all points to be removed and drop from the dataframe.
    df = df.drop(np.append(dchg_to_drop, chrg_to_drop))
    return df


def perform_ica(df, volt_col='volt', prespan=0.0055, gausspan=0.03):
    """
    Summary function that performs all necessary steps to perform ICA/DVA
    :param df:
    :param volt_col:
    :param prespan:
    :return:
    """
    df = remove_faulty_points(df)
    chrg_df = df[df.curr > 0]
    dchg_df = df[df.curr < 0]
    if not chrg_df.empty:
        chrg_df = smooth_with_span(chrg_df, col=volt_col, prespan=prespan)
        chrg_df = simplified_ica_dva(chrg_df)
        chrg_df.loc[:, 'ica_gauss'] = gaussianfilterint(chrg_df['volt'], chrg_df['ica'], gausspan=gausspan)
        chrg_df.loc[:, 'dva_gauss'] = gaussianfilterint(chrg_df['volt'], chrg_df['dva'], gausspan=gausspan)
    if not dchg_df.empty:
        dchg_df = smooth_with_span(dchg_df, col=volt_col, prespan=prespan)
        dchg_df = simplified_ica_dva(dchg_df)
        dchg_df.loc[:, 'ica_gauss'] = gaussianfilterint(dchg_df['volt'], dchg_df['ica'], gausspan=gausspan)
        dchg_df.loc[:, 'dva_gauss'] = gaussianfilterint(dchg_df['volt'], dchg_df['dva'], gausspan=gausspan)
    df = pd.concat([chrg_df, dchg_df])
    return df


def ica_on_arb_data(x_data, y_data, prespan=0.0055, gausspan=0.03):
    """
    Perform ICA on two np arrays with capacity and voltage data
    :param x_data: Capaity data of IC run
    :param y_data: Voltage data of IC run
    :param prespan:
    :return:
    """
    # Find the points where voltage different is consistent with charge/discharge case to remove noisy points
    curr_case = np.sign(np.diff(y_data).mean())
    rel_points = np.sign(np.diff(y_data)) == curr_case
    rel_points = np.insert(rel_points, -1, True)
    y_data = y_data[rel_points]
    x_data = x_data[rel_points]
    # y_data is measurement data to be smoothed
    smoothing_span = int(prespan * y_data.shape[0])
    # smoothing span must be odd
    smoothing_span = smoothing_span - 1 + smoothing_span % 2
    y_data = smooth(y_data, smoothing_span)
    ica = np.gradient(x_data, y_data * curr_case)
    dva = np.gradient(y_data * curr_case, x_data)
    ica = gaussianfilterint(x_data, ica, gausspan=gausspan)
    dva = gaussianfilterint(x_data, dva, gausspan=gausspan)
    output = {
        'Qs': x_data,
        'Es': y_data,
        'dqdv': ica,
        'dvdq': dva
    }
    return output


if __name__ == '__main__':
    pkl_dir = r"\\sol.ita.chalmers.se\groups\batt_lab_data\20200923_pkl\pickle_files_channel_2_8"
    processed_data = {}
    short_fig, sax = plt.subplots(1, 1)
    long_fig, lax = plt.subplots(1, 1)
    for root, dir, files in os.walk(pkl_dir):
        for name in files:
            if '_ica_dump_rpt_1.' in name:
                print('Analysing data in file: {}'.format(name))
                ica_df = pd.read_pickle(os.path.join(root, name))
                if not ica_df.empty:
                    # sub_df = ica_df[ica_df.step == ica_df.step.unique()[0]]
                    # Perform rough cleaning of data
                    # ica_df = smooth_with_span(ica_df)
                    gb = ica_df.groupby('step')
                    res_list = [ica_on_arb_data(gb.get_group(x)['cap'].values, gb.get_group(x)['volt'].values) for x in gb.groups]
                    # ica_df = ica_df[(ica_df['volt'].diff() * np.sign(ica_df['curr'])) > 0]

                    ica_df_long = ica_df.copy()
                    # ica_df = simplified_ica_dva(ica_df)
                    # ica_df_long = large_span_ica_dva(ica_df_long)
                    gauss_span = int(ica_df.shape[0] * 0.03)
                    # ica_df.loc[:, 'ica_gauss'] = gaussianfilterint(ica_df.volt, ica_df.ica, gauss_span)
                    # ica_df_long.loc[:, 'ica_gauss'] = gaussianfilterint(ica_df_long.volt, ica_df_long.ica, gauss_span)
                    gb = ica_df.groupby('step')
                    ica_dch = [perform_ica(gb.get_group(x), prespan=0.015)
                               for x in gb.groups if gb.get_group(x).curr.mean() < 0][0]
                    ica_chg = [perform_ica(gb.get_group(x), prespan=0.015)
                               for x in gb.groups if gb.get_group(x).curr.mean() > 0][0]
                    comb_ica = pd.concat([ica_chg, ica_dch])
                    # plt.plot(sub_df['float_time'] - sub_df['float_time'].iloc[0], sub_df['volt'], label=name)
                    processed_data['{}_{}'.format(name, 'dch')] = gaussianfilter(ica_dch)
                    processed_data['{}_{}'.format(name, 'chg')] = gaussianfilter(ica_chg, prespan=0.012)
                    processed_data['{}_{}'.format(name, 'full')] = gaussianfilter(ica_df, prespan=0.008)
                    # Es, Qs, dVdQ, dQdV = processed_data['{}_{}'.format(name, 'full')]
                    sax.plot(ica_dch.volt, ica_dch.ica_gauss, label=name)
                    sax.plot(ica_chg.volt, ica_chg.ica_gauss, label=name)
                    lax.plot(comb_ica.mAh - comb_ica.mAh.min(), comb_ica.volt)
    sax.legend()
    sax.set_title('ICA 128s pulse duration')
