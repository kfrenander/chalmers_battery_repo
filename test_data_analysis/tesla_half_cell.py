import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz, trapz
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import gaussian
from backend_fix import fix_mpl_backend
import os
# plt.rcParams['axes.grid'] = True
#plt.style.use('chalmers_kf')


def smooth(a, WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation

    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))


def gauss_win_kf(L, a=2.5):
    """
    Function that mirrors matlab's function to create a gaussian window since implementation in scipy doesn't seem to be
    analogous
    :param L: Length of window (int)
    :param a: Standard deviation of window, matlab standard 2.5
    :return:
    """
    N = L - 1
    n = np.arange(N + 1) - N / 2
    w = np.exp(-(1/2)*(a*n/(N / 2))**2)
    return w


def data_reader(path):
    """
    Function that reads in data and gives it standardised format.
    :param path:
    :return:
    """
    df = pd.read_csv(path, sep='\t', names=['time', 'curr', 'pot'])
    df.loc[:, 'cap'] = cumtrapz(df.curr, df.time / 3600, initial=0)
    nbr_of_switches = df.loc[abs(df.curr.diff()) > 1e-4].shape[0]
    df.loc[abs(df.curr.diff()) > 1e-4, 'step'] = range(1, nbr_of_switches + 1)
    df.loc[:, 'step'] = df.loc[:, 'step'].fillna(method='ffill').fillna(0)
    return df


def visualise_hc_test(df):
    """
    Standardised visualisation
    :param df:
    :return:
    """
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(df.time, df.pot)
    ax[1].plot(df.time, df.curr)
    cap_fig = plt.figure()
    plt.plot((df.cap - df_neg.cap.min())*1000, df.pot, alpha=0.8, linewidth=0.6)
    plt.xlabel('Capacity [mAh]')
    plt.ylabel('Potential v Li [V]')
    plt.grid(True)
    return {'subplots': fig, 'cap_fig': cap_fig}


def visualise_differentials(df):
    """
    Standardised visualisation
    :param df:
    :return:
    """
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(df.pot, df.dqdv, label='Unfiltered ICA')
    ax[0].plot(df.pot, df.dqdv_filt, label='Filtered ICA')
    ax[1].plot(df.cap, df.dvdq, label='Unfiltered DVA')
    ax[1].plot(df.cap, df.dvdq_filt, label='Filtered DVA')
    ax[0].grid(True)
    ax[1].grid(True)
    return ax


def calc_dqdv(df, stp_size=15):
    df_ds = df.iloc[::stp_size]
    df.loc[:, 'dqdv'] = (df_ds.cap.diff() / df_ds.pot.diff())
    df.loc[:, 'dvdq'] = (df_ds.pot.diff() / df_ds.cap.diff())
    df.fillna(method='bfill', inplace=True)
    df.loc[:, 'dqdv_filt'] = savgol_filter(df.dqdv, 45, 0)
    df.loc[:, 'dvdq_filt'] = savgol_filter(df.dvdq, 45, 0)
    return df


def gaussianfilter(df,
                   prespan=0.0055,
                   gausspan=0.03,
                   gausswin=True,
                   pot_col='pot',
                   cap_col='cap'):
    """
    Function to 
    :param df:
    :param prespan:
    :param gausspan:
    :param gausswin:
    :param pot_col:
    :return:
    """
    # Define span to use for smoothing
    smooth_span = int(prespan * df[pot_col].shape[0])
    # Span must be odd
    smooth_span = smooth_span - 1 + smooth_span % 2
    Es = smooth(df[pot_col], smooth_span)
    dV = np.diff(Es)
    if cap_col == 'cap':
        Qs = smooth(df[cap_col] - df[cap_col].min(), smooth_span)
    else:
        Qs = smooth(df[cap_col], smooth_span)
    dQ = np.diff(Qs)
    test = gaussian_filter1d(np.gradient(Qs, Es), sigma=smooth_span)
    Es_conv, dqdv_conv = gaussianfilterconvolution(Es, np.gradient(Qs, Es), len(Es) * gausspan)
    Qs_conv, dvdq_conv = gaussianfilterconvolution(Qs, np.gradient(Es, Qs), len(Qs) * gausspan)
    if gausswin:
        Es, dQdV = gaussianfilterint(Es, np.gradient(Qs, Es), len(Es) * gausspan)
        Qs, dVdQ = gaussianfilterint(Qs, np.gradient(Es, Qs), len(Es) * gausspan)
    else:
        dQdV = dQ / dV
        dVdQ = dV / dQ
    df_out = pd.DataFrame({'volt': Es_conv,
                           'cap': Qs_conv,
                           'ica': dqdv_conv,
                           'dva': dvdq_conv})
    return df_out


def gaussianfilterint(x, y, span):
    ynan = np.isnan(y)
    span = np.floor(span)
    n = len(y)
    span = min(span, n)
    width = span - 1 + span % 2
    xreps = any(np.diff(x)==0)
    if width == 1 and not xreps and not any(ynan):
        c = y
    c = np.zeros_like(y)
    G = gaussian(width, width / 5)
    j = 0
    h = int((width - 1) / 2)
    # FAULTTRACING NEEDED FOR GAUSSIAN FILTER TO WORK!!!
    for k in range(h):
        c[k] = sum(G[(h - k):] * y[:(1 + h + k)]) / sum(G[(h - k):])
        j += 1
    for k in range(h, n - (h + 1)):
        c[k] = sum(G * y[k-h:k+h + 1]) / sum(G)
    for k in range(n - (h + 1), n):
        c[k] = sum(G[:(1 + h + j)] * y[k-h:]) / sum(G[:(1 + h + j)])
        j -= 1
    return x, c


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
    G = gaussian(width, width / 5)
    c = np.convolve(y, G, 'same') / sum(G)
    return x, c


def find_hysteresis(df, rng):
    from scipy.interpolate import interp1d
    df_sub = df[df.step.isin(rng)]
    df_sub.loc[:, 'soc'] = (df_sub.cap - df_sub.cap.min()) / (df_sub.cap - df_sub.cap.min()).max()
    df_chg = df_sub[df_sub.curr > 0]
    df_dch = df_sub[df_sub.curr < 0]
    pot_int_chg = interp1d(df_chg.soc, df_chg.pot)
    pot_int_dch = interp1d(df_dch.soc, df_dch.pot)
    x_int = np.linspace(max(df_dch.soc.min(), df_chg.soc.min()),
                        min(df_dch.soc.max(), df_chg.soc.max()),
                        num=200)
    hyst = pot_int_chg(x_int) - pot_int_dch(x_int)
    return x_int.max() - x_int, hyst, df_sub


def calc_step_caps(df):
    cap_arr = np.array(df.groupby('step').apply(lambda x: trapz(x.curr, x.time) / 3.6))
    cap_chrg = cap_arr[cap_arr > 1e-3]
    cap_dchg = cap_arr[cap_arr < -1e-3]
    # Filter out deviations larger than one standard deviation to remove outliers
    cap_chrg = cap_chrg[abs(cap_chrg - cap_chrg.mean()) < cap_chrg.std()]
    cap_dchg = cap_dchg[abs(cap_dchg - cap_dchg.mean()) < cap_dchg.std()]
    # As several steps are not full charge discharge steps this is done twice
    cap_chrg = cap_chrg[abs(cap_chrg - cap_chrg.mean()) < cap_chrg.std()]
    cap_dchg = cap_dchg[abs(cap_dchg - cap_dchg.mean()) < cap_dchg.std()]
    return cap_dchg, cap_chrg


def plot_hysteresis(df, x, hyst, electrode_name='negative'):
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(df.soc.max() - df.soc, df.pot,
               label='Voltage', linewidth='0.8')
    ax[1].set_xlabel('SOC [-]')
    fig.text(0.04, 0.5, 'Voltage [V]',
             va='center',
             rotation='vertical',
             fontsize=14)
    fig.suptitle('Voltage and hysteresis of {} electrode halfcell'.format(electrode_name))
    ax[0].legend()
    ax[1].plot(x, hyst, label='Hysteresis', linewidth=0.8)
    ax[1].legend()
    fig.savefig(
        r'Z:\Images\HalfcellIca\{}_electrode_hysteresis.png'.format(electrode_name),
        dpi=800)
    return fig


def plot_ica_dva(df, title=''):
    fig, ax = plt.subplots(2, 2, sharex='col')
    soc = (df.cap.max() - df.cap) / df.cap.max()
    ax[0, 0].plot(soc, df.volt)
    ax[0, 0].set_title('DVA')
    ax[1, 0].plot(soc, df.dva)
    # ax[1, 0].set_title('DVA')
    ax[1, 0].set_ylim(0, 500)
    ax[1, 0].set_xlabel('SOC')
    ax[0, 1].plot(df.volt, df.cap)
    ax[0, 1].set_title('ICA')
    ax[1, 1].plot(df.volt, df.ica)
    ax[1, 1].set_ylim(0, 0.3)
    ax[1, 1].set_xlim(0, 0.5)
    # ax[1, 1].set_title('ICA')
    # ax[1, 1].set_ylim(0, 500)
    ax[1, 1].set_xlabel('Voltage [V]')
    fig.suptitle(title)
    return fig


def plot_ica(df_dlth, df_lith):
    if df_dlth.volt.max() > 3:
        pos_case = True
        neg_case = False
    else:
        neg_case = True
        pos_case = False
    fig, ax = plt.subplots(2, 2, sharex='col')
    ax[0, 0].plot(df_dlth.volt, df_dlth.cap)
    ax[0, 0].set_title('Delithiation')
    ax[0, 1].plot(df_lith.volt, df_lith.cap)
    ax[0, 1].set_title('Lithiation')
    if neg_case:
        ax[1, 0].plot(df_dlth.volt, df_dlth.ica)
        ax[1, 0].set_ylim(1e-3, 0.3)
        ax[1, 0].set_xlim(0, 0.5)
        ax[1, 1].plot(df_lith.volt, df_lith.ica)
        ax[1, 1].set_ylim(1e-3, 0.3)
        ax[1, 1].set_xlim(0, 0.5)
    elif pos_case:
        ax[1, 0].plot(df_dlth.volt, df_dlth.ica)
        ax[1, 0].set_ylim(0, 0.03)
        ax[1, 0].set_xlim(3.3, 4.4)
        ax[1, 1].plot(df_lith.volt, df_lith.ica)
        ax[1, 1].set_ylim(0, 0.03)
        ax[1, 1].set_xlim(3, 4.4)
    ax[1, 0].set_xlabel('Voltage [V]')
    ax[1, 1].set_xlabel('Voltage [V]')
    return fig


def add_step_cap(data_dict):
    for i in data_dict:
        tmp_df = data_dict[i]
        tmp_df.loc[:, 'step_cap'] = cumtrapz(tmp_df.curr.abs(), tmp_df.time / 3600, initial=0)
        data_dict[i] = tmp_df
    return data_dict


def plot_dva(df_dlth, df_lith):
    if df_dlth.volt.max() > 3:
        pos_case = True
        neg_case = False
    else:
        neg_case = True
        pos_case = False
    if neg_case:
        soc_dlth = (df_dlth.cap.max() - df_dlth.cap) / df_dlth.cap.max()
        soc_lith = (df_lith.cap.max() - df_lith.cap) / df_lith.cap.max()
    elif pos_case:
        soc_dlth = 1 - (df_dlth.cap.max() - df_dlth.cap) / df_dlth.cap.max()
        soc_lith = 1 - (df_lith.cap.max() - df_lith.cap) / df_lith.cap.max()
    fig, ax = plt.subplots(2, 2, sharex='col')
    ax[0, 0].plot(soc_dlth, df_dlth.volt)
    ax[0, 0].set_title('Delithiation')
    ax[1, 0].plot(soc_dlth, df_dlth.dva)
    ax[1, 0].set_ylim(0, 500)
    # ax[1, 0].set_xlim(0, 0.5)
    ax[1, 0].set_xlabel('SOC [-]')
    ax[0, 1].plot(soc_lith, df_lith.volt)
    ax[0, 1].set_title('Lithiation')
    ax[1, 1].plot(soc_lith, df_lith.dva)
    ax[1, 1].set_ylim(0, 500)
    # ax[1, 1].set_xlim(0, 0.5)
    ax[1, 1].set_xlabel('SOC [-]')
    return fig


if __name__ == '__main__':
    fix_mpl_backend()
    tesla_data_file_neg = r"Z:\Provning\Halvcellsdata\20200910-AJS-NH0S05-Tes-C10-BB2.txt"
    df_neg = data_reader(tesla_data_file_neg)
    gb_neg = df_neg.groupby('step')
    lithiation_lst_neg = [gb_neg.get_group(x) for x in gb_neg.groups
                          if gb_neg.get_group(x).curr.mean() < 0]
    delithiation_lst_neg = [gb_neg.get_group(x) for x in gb_neg.groups
                            if gb_neg.get_group(x).curr.mean() > 1e-4]
    best_lith_case_neg = {'{:.0f}'.format(df.step.mean()): df for df in lithiation_lst_neg if df.pot.max() > 1}
    best_dlth_case_neg = {'{:.0f}'.format(df.step.mean()): df for df in delithiation_lst_neg if df.pot.max() > 1.3}
    span_neg = 0.06
    res_lith_neg = gaussianfilter(best_lith_case_neg['11'], gausspan=span_neg)
    res_delith_neg = gaussianfilter(best_dlth_case_neg['21'], gausspan=span_neg)
    fig_lith = plot_ica_dva(res_lith_neg, title='Lithiation')
    fig_dlth = plot_ica_dva(res_delith_neg, title='Delithiation')
    fig_ica = plot_ica(res_delith_neg, res_lith_neg)
    fig_dva = plot_dva(res_delith_neg, res_lith_neg)
    cap_hc_neg = (df_neg.cap - df_neg.cap.min()).max()
    stepcap_neg_dchg, stepcap_neg_chrg = calc_step_caps(df_neg)
    r_hc = 1.5 / 2
    A_hc = r_hc**2 * math.pi
    spec_cap_neg = abs(stepcap_neg_dchg.mean()) / A_hc

    tesla_data_file_pos = r"Z:\Provning\Halvcellsdata\20200910-AJS-PH0S06-Tes-C10-BB5.txt"
    df_pos = data_reader(tesla_data_file_pos)
    gb_pos = df_pos.groupby('step')
    stepcap_pos_dchg, stepcap_pos_chrg = calc_step_caps(df_pos)
    delithiation_lst_pos = [gb_pos.get_group(x) for x in gb_pos.groups if gb_pos.get_group(x).curr.mean() > 1e-4]
    lithiation_lst_pos = [gb_pos.get_group(x) for x in gb_pos.groups if gb_pos.get_group(x).curr.mean() < 0]
    best_dlth_case_pos = {'{:.0f}'.format(df.step.mean()): df for df in delithiation_lst_pos
                        if df.pot.max() > 4.2 and df.pot.min() < 3.6}
    best_lith_case_pos = {'{:.0f}'.format(df.step.mean()): df for df in lithiation_lst_pos
                        if df.pot.max() > 4.1 and df.pot.min() < 3.2}
    span_pos = 0.03
    res_delith_pos = gaussianfilter(best_dlth_case_pos['7'], gausspan=span_pos)
    res_lith_pos = gaussianfilter(best_lith_case_pos['9'], gausspan=span_pos)
    pos_ica = plot_ica(res_delith_pos, res_lith_pos)
    pos_dva = plot_dva(res_delith_pos, res_lith_pos)
    cap_hc_pos = (df_pos.cap - df_pos.cap.min()).max()
    spec_cap_pos = abs(stepcap_pos_dchg.mean()) / A_hc

    x_neg, hyst_neg, part_df_neg = find_hysteresis(df_neg, range(5, 9))
    fig_neg = plot_hysteresis(part_df_neg, x_neg, hyst_neg, 'neg')

    x_pos, hyst_pos, part_df_pos = find_hysteresis(df_pos, range(7, 11))
    fig_pos = plot_hysteresis(part_df_pos, x_pos, hyst_pos, 'pos')

    # Section to create image for article on Tesla small DoD degradation
    plt.rcParams['figure.figsize'] = 9, 6
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['lines.linewidth'] = 1.7
    plt.rcParams['xtick.labelsize'] = 'x-large'
    plt.rcParams['ytick.labelsize'] = 'x-large'

    output_dir_article = r"Z:\Provning\Analysis\ALINE_plots\small_soc\final_versions"
    if not os.path.isdir(output_dir_article):
        os.mkdir(output_dir_article)
    mrk_font_neg = {'family': 'Times New Roman',
                    'weight': 'bold',
                    'size': 15,
                    'color': 'white'}
    box_props_neg = dict(boxstyle='circle', facecolor='black', alpha=0.9, edgecolor='black')
    A_neg = 1000
    neg_scale = A_neg / A_hc
    best_lith_case_neg = add_step_cap(best_lith_case_neg)
    best_dlth_case_neg = add_step_cap(best_dlth_case_neg)
    dfa_neg_lth = gaussianfilter(best_lith_case_neg['23'], cap_col='step_cap', gausspan=0.06)
    dfa_neg_dlt = gaussianfilter(best_dlth_case_neg['9'], cap_col='step_cap', gausspan=0.06)
    alex_data_chrg = r"\\sol.ita.chalmers.se\groups\batt_lab_data\ALINE_data\DataFromAlex\GrSiOxchg_E_V_dqdv_mAhcm-2V-1.csv"
    alex_data_dchg = r"\\sol.ita.chalmers.se\groups\batt_lab_data\ALINE_data\DataFromAlex\GrSiOxdchg_E_V_dqdv_mAhcm-2V-1.csv"
    alex_df_chrg = pd.read_csv(alex_data_chrg, names=['volt', 'ica'])
    alex_df_dchg = pd.read_csv(alex_data_dchg, names=['volt', 'ica'])
    art_fig_neg, nax = plt.subplots(1, 1)
    y_loc = 60
    peak_coords_neg = [(0.112, y_loc), (0.15, y_loc), (0.232, y_loc), (0.28, y_loc)]
    x_coords_neg = [x for x, y in peak_coords_neg]
    y_coords_neg = [y for x, y in peak_coords_neg]
    peaks_neg = [str(x) for x in range(1, 5)]
    nax.plot(alex_df_chrg.volt, alex_df_chrg.ica, color='blue', label='Si-Gr half cell')
    nax.plot(alex_df_dchg.volt, alex_df_dchg.ica, color='blue')
    for p, x, y in zip(peaks_neg, x_coords_neg, y_coords_neg):
        plt.text(x, y, p, fontdict=mrk_font_neg, bbox=box_props_neg, horizontalalignment='center')
    plt.vlines(x_coords_neg, ymin=0, ymax=y_loc - 10, color='black', linewidth=0.8)
    nax.set_xlim(0.05, 0.6)
    nax.set_xlabel('Half cell potential vs Li/Li$^{+}$, V', fontsize=20)
    nax.set_ylabel(r'IC, dQ dV$^{-1}$, mAh cm$^{-2}$ V$^{-1}$', fontsize=20)
    nax.legend(loc='center left', fontsize=18)
    nax.text(0.44, y_loc, 'Si-Gr reaction', color='blue', fontsize=15, weight='bold')
    nax.invert_xaxis()
    nax.set_ylim(-60, 80)
    nax.grid(False)
    art_fig_neg.savefig(os.path.join(output_dir_article, 'negative_halfcell_ica_updated_labels_cropped.pdf'))
    art_fig_neg.savefig(os.path.join(output_dir_article, 'negative_halfcell_ica_updated_labels_cropped.png'), dpi=400)

    plt.yscale('log')
    nax.set_ylim(7e-4, 6)
    art_fig_neg.savefig(os.path.join(output_dir_article, 'negative_halfcell_ica_logy.pdf'))
    art_fig_neg.savefig(os.path.join(output_dir_article, 'negative_halfcell_ica_logy.png'), dpi=400)



    A_pos = 1000
    pos_scale = A_pos / A_hc
    best_dlth_case_pos = add_step_cap(best_dlth_case_pos)
    best_lith_case_pos = add_step_cap(best_lith_case_pos)
    dfa_pos_dlt = gaussianfilter(best_dlth_case_pos['7'], cap_col='step_cap', gausspan=span_pos)
    dfa_pos_lth = gaussianfilter(best_lith_case_pos['9'], cap_col='step_cap', gausspan=span_pos)
    mrk_font_pos = {'family': 'Times New Roman',
                    'weight': 'normal',
                    'size': 15,
                    'color': 'black'}
    box_props_pos = dict(boxstyle='circle', facecolor='white', alpha=1, edgecolor='black')
    y_loc_pos = 20.5
    peak_coords = [(3.566, y_loc_pos), (3.72, y_loc_pos), (4.01, y_loc_pos), (4.197, y_loc_pos)]
    x_coords = [x for x, y in peak_coords]
    y_coords = [y for x, y in peak_coords]
    peaks = reversed([str(x) for x in range(1, 5)])

    art_fig_pos, pax = plt.subplots(1, 1)
    pax.plot(dfa_pos_lth.volt, dfa_pos_lth.ica * pos_scale, color='red', label='NCA half cell')
    pax.plot(dfa_pos_dlt.volt, dfa_pos_dlt.ica * pos_scale, color='red')
    for p, x, y in zip(peaks, x_coords, y_coords):
        plt.text(x, y, p, fontdict=mrk_font_pos, bbox=box_props_pos, horizontalalignment='center')
    plt.vlines(x_coords, ymin=0, ymax=y_loc_pos - 2, color='black', linewidth=0.8)
    pax.set_xlabel('Half cell potential vs Li/Li$^{+}$, V', fontsize=20)
    pax.set_ylabel(r'IC dQ dV$^{-1}$, mAh cm$^{-2}$ V$^{-1}$', fontsize=20)
    pax.legend(loc='lower left', fontsize=18)
    pax.set_ylim(-20, 25)
    pax.text(3.2, y_loc_pos, 'NCA reaction', color='red', fontsize=15, weight='bold')
    pax.grid(False)
    art_fig_pos.savefig(os.path.join(output_dir_article, 'positive_halfcell_ica_updated_labels.pdf'))
    art_fig_pos.savefig(os.path.join(output_dir_article, 'positive_halfcell_ica_updated_labels.png'), dpi=400)
