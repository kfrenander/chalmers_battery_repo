from rpt_data_analysis.ReadRptClass import OrganiseRpts, look_up_fce
from bda_data_plots import yield_dataset, find_peak_coords
from backend_fix import fix_mpl_backend
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import numpy as np
from scipy.signal import find_peaks
import os
import re
x_width = 8
aspect_rat = 3 / 4
plt.rcParams['figure.figsize'] = x_width, aspect_rat * x_width
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 1.7
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams["text.usetex"]
lbl_font = {'weight': 'normal',
            'size': 20}
plt.rc('legend', fontsize=14)
mark_size = 5
cap_size = 6
plt.rc('font', **{"family": 'sans-serif', 'sans-serif': 'Helvetica'})


if __name__ == '__main__':
    fix_mpl_backend()
    data_dir = r"\\sol.ita.chalmers.se\groups\batt_lab_data\20200923"
    hc_data = r"\\sol.ita.chalmers.se\groups\batt_lab_data\analysis_directory\TeslaPulseAgeingPaper\half_cell_processed_data"
    o_dir = r"\\sol.ita.chalmers.se\groups\batt_lab_data\analysis_directory\TeslaPulseAgeingPaper\optimisation_from_hc"
    fig_dir = r"\\sol.ita.chalmers.se\groups\batt_lab_data\analysis_directory\TeslaPulseAgeingPaper"
    hc_data_set = {k.split('.')[0]: pd.read_pickle(os.path.join(hc_data, k)) for k in os.listdir(hc_data)}
    neg_df = hc_data_set['neg_lithiation']
    pos_df = hc_data_set['pos_delithiation']
    neg_df.loc[:, 'cell_cap'] = (neg_df.cap.max() - neg_df.cap) * 1000
    pos_df.loc[:, 'cell_cap'] = pos_df.cap * 1000
    data_set = OrganiseRpts(data_dir, proj='BDA')
    init_data = data_set.ica_dict['128s_2_8']['rpt_1']
    init_data_chrg = init_data[init_data.curr > 0]
    dchg_neg_df = hc_data_set['neg_delithiation']
    dchg_pos_df = hc_data_set['pos_lithiation']
    chrg_dva = {
        'fc': init_data_chrg,
        'neg': neg_df,
        'pos': pos_df
    }

    dchg_dva = {
        'fc': init_data[init_data.curr < 0],
        'neg': dchg_neg_df * 1000,
        'pos': (dchg_pos_df.cap.max() - dchg_pos_df.cap) * 1000
    }

    fc_offset = 0.08
    neg_scale = lambda x, scale, offset: interp1d(scale * neg_df['cell_cap'] + offset,
                                                  neg_df['dva'] / 1000, fill_value='extrapolate')(x)
    pos_scale = lambda x, scale, offset: interp1d(scale * pos_df['cell_cap'] + offset,
                                                  pos_df['dva'] / 1000, fill_value='extrapolate')(x)
    full_cell = lambda x, ne_scale, ne_offset, \
                       pe_scale, pe_offset: neg_scale(x, ne_scale, ne_offset) + \
                                            pos_scale(x, pe_scale, pe_offset) + fc_offset
    p_init = [698, -100, 630, -650]
    lo_b = (300, -500, 300, -700)
    hi_b = (700, 0, 700, -500)
    popt, pcov = curve_fit(full_cell, init_data_chrg.cap, init_data_chrg.dva_gauss,
                           p0=p_init, bounds=(lo_b, hi_b))
    plt.figure()
    plt.plot(init_data_chrg.cap, init_data_chrg.dva_gauss, label='raw data')
    plt.plot(init_data_chrg.cap, full_cell(init_data_chrg.cap, *popt), label='fitted')
    plt.legend()
    plt.ylim(0, 0.5)

    plt.close('all')
    plt.figure()
    # my_p = [680, -70, 625, -630]
    my_p = [675, -50, 625, -630]
    plt.plot(init_data_chrg.cap, init_data_chrg.dva_gauss, label='target', color='k')
    plt.plot(init_data_chrg.cap, full_cell(init_data_chrg.cap, *my_p), label='fit fc', color='grey')
    plt.plot(init_data_chrg.cap, neg_scale(init_data_chrg.cap, *my_p[:2]), label='neg fit', color='blue')
    plt.plot(init_data_chrg.cap, pos_scale(init_data_chrg.cap, *my_p[2:4]), label='pos fit', color='red')
    plt.ylim(0, 0.5)
    plt.legend()

    pretty_dva_fig, pax = plt.subplots(1, 1, figsize=(8, 6))
    pax.plot(init_data_chrg.cap / 1000, init_data_chrg.dva_gauss, label='Full cell')
    pax.plot(init_data_chrg.cap / 1000, neg_scale(init_data_chrg.cap, *my_p[:2]),
             label='Negative Electrode', linestyle='dashed')
    pax.plot(init_data_chrg.cap / 1000, pos_scale(init_data_chrg.cap, *my_p[2:4]),
             label='Positive Electrode', linestyle='dotted')
    pax.legend(prop={"size": 18})
    pax.set_ylabel('dV/dQ (V/Ah)', fontsize=22)
    pax.set_xlabel('Full cell Capacity (Ah)', fontsize=22)
    pax.set_ylim((0, 0.5))
    pretty_dva_fig.savefig(os.path.join(fig_dir, 'fullcell_and_halfcells_dva.png'), dpi=800)
    selected_data_pts = yield_dataset(20, 1, dataset=data_set, operation='chrg', cell='128s_2_8', start=1)
    pks = {k: find_peak_coords(selected_data_pts[k], md='chrg') for k in selected_data_pts}
    for x in pks['rpt_1'].index:
        if 'NCA' in pks['rpt_1'].loc[x, 'peak_id']:
            pax.axvline(pks['rpt_1'].loc[x, 'cap'] / 1000, ls='dashed', color='red')
        else:
            pax.axvline(pks['rpt_1'].loc[x, 'cap'] / 1000, ls='dashed', color='maroon')
    pretty_dva_fig.savefig(os.path.join(fig_dir, 'fullcell_and_halfcells_dva_with_peaks.png'), dpi=300)
    pretty_dva_fig.savefig(os.path.join(fig_dir, 'fullcell_and_halfcells_dva_with_peaks.eps'))

    neg_scaled_x_axis = (my_p[0] * neg_df['cell_cap'] + my_p[1])
    pos_scaled_x_axis = (my_p[2] * pos_df['cell_cap'] + my_p[3])
    pr_full_fig, pv_ax = plt.subplots(1, 1, figsize=(8, 6))
    pv_ax.plot(init_data_chrg.cap / 1000, init_data_chrg.volt, label='Full cell')
    pv_ax.plot(neg_scaled_x_axis / 1000, neg_df['volt'], label='Negative Electrode', linestyle='dashed')
    pv_ax.plot(pos_scaled_x_axis / 1000, pos_df['volt'], label='Positive Electrode', linestyle='dotted')
    pv_ax.set_ylabel('Voltage (V)', fontsize=22)
    pv_ax.set_xlabel('Capacity (Ah)', fontsize=22)
    pv_ax.legend(prop={'size': 18})
    pr_full_fig.savefig(os.path.join(fig_dir, 'fullcell_and_halvcell_voltage.png'), dpi=300)
    pr_full_fig.savefig(os.path.join(fig_dir, 'fullcell_and_halvcell_voltage.eps'))

    # Plot volt and DVA together on subplots
    combined_plot, cax = plt.subplots(2, 1, figsize=(8, 12), sharex=True)
    cax[0].plot(init_data_chrg.cap / 1000, init_data_chrg.volt, label='Full cell')
    cax[0].plot(neg_scaled_x_axis / 1000, neg_df['volt'], label='Negative Electrode', linestyle='dashed')
    cax[0].plot(pos_scaled_x_axis / 1000, pos_df['volt'], label='Positive Electrode', linestyle='dotted')
    cax[0].set_ylabel('Voltage / V', fontsize=18)
    # cax[0].set_xlabel('Capacity (Ah)', fontsize=22)
    cax[1].plot(init_data_chrg.cap / 1000, init_data_chrg.dva_gauss, label='Full cell')
    cax[1].plot(neg_scaled_x_axis / 1000, neg_scale(neg_scaled_x_axis, *my_p[:2]),
                label='Negative Electrode', linestyle='dashed')
    cax[1].plot(pos_scaled_x_axis / 1000, pos_scale(pos_scaled_x_axis, *my_p[2:4]),
                label='Positive Electrode', linestyle='dotted')
    cax[0].legend(prop={"size": 14})
    cax[1].set_ylabel('dV/dQ (V/Ah)', fontsize=18)
    cax[1].set_xlabel('Capacity (Ah)', fontsize=18)
    cax[1].set_ylim((0, 0.5))
    cax[1].set_xlim((-0.5, 5))
    combined_plot.savefig(os.path.join(fig_dir, 'combined_plot_volt_and_dva.png'), dpi=800)

    col = ['nes', 'neo', 'pes', 'peo']
    for cell_set in data_set.ica_dict:
        plt.figure(figsize=(14, 9))
        opt_df = pd.DataFrame(columns=col)
        for rpt in data_set.ica_dict[cell_set]:
            df = data_set.ica_dict[cell_set][rpt]
            df = df[df.curr > 0]
            try:
                popt, pcov = curve_fit(full_cell, df.cap, df.dva_gauss,
                                       p0=p_init, bounds=(lo_b, hi_b))
                opt_df.loc[look_up_fce(rpt), :] = popt
                if look_up_fce(rpt) < 300 and look_up_fce(rpt) % 100 == 0:
                    l = plt.plot(df.cap, df.dva_gauss, label=f'Raw_data at FCE {look_up_fce(rpt)}')
                    act_col = l[0].get_color()
                    plt.plot(df.cap, full_cell(df.cap, *popt), label=f'Fitted at FCE {look_up_fce(rpt)}',
                             color=act_col, linestyle='dashed')
                    plt.xlabel('Capacity [mAh]')
                    plt.ylabel('DVA [V/Ah]')
                    plt.legend(prop={"size": 16})
            except RuntimeError:
                print(f'No optimal solution found for {cell_set} at {rpt}')
        plt.legend()
        plt.ylim(0, 0.5)
        op = opt_df.astype('float').sort_index()
        op.to_csv(os.path.join(o_dir, cell_set + '_fit_values.csv'))
        op.to_pickle(os.path.join(o_dir, cell_set + '_fit_values.pkl'))
        # plt.savefig(os.path.join(o_dir, cell_set + '_figure.png'))

