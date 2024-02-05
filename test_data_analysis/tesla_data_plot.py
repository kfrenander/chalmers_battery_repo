import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
from rpt_data_analysis.ReadRptClass import OrganiseRpts
import os
import natsort
plt.rcParams['axes.grid'] = True
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['legend.fontsize'] = 20


def look_up_fce_nrc(rpt_str):
    rpt_num = int(re.search(r'\d+', rpt_str).group())
    x_rpt = np.arange(1, 35)
    y_fce = np.arange(0, 34*150, 150)
    fce = int(interp1d(x_rpt, y_fce)(rpt_num))
    return fce

def my_plot_fun(data_dict, fig_name):
    hyst_fig, h_ax = plt.subplots(1, 1, figsize=(8, 6))
    volt_fig, v_ax = plt.subplots(1, 1, figsize=(8, 6))
    h_ax.grid(True)
    v_ax.grid(True)
    soc_lvls = re.findall(r'\d+', fig_name)
    for key in data_dict:
        tmp_ica = data_dict[key]
        soc = (tmp_ica.mAh - tmp_ica.mAh.min()) / (tmp_ica.mAh.max() - tmp_ica.mAh.min())
        tmp_ica.loc[:, 'soc'] = soc
        u_int_chrg = interp1d(tmp_ica[tmp_ica.curr > 0].soc, tmp_ica[tmp_ica.curr > 0].volt)
        u_int_dchg = interp1d(tmp_ica[tmp_ica.curr < 0].soc, tmp_ica[tmp_ica.curr < 0].volt)
        x_low = max(tmp_ica[tmp_ica.curr > 0].soc.min(), tmp_ica[tmp_ica.curr < 0].soc.min())
        x_hi = min(tmp_ica[tmp_ica.curr > 0].soc.max(), tmp_ica[tmp_ica.curr < 0].soc.max())
        x_int = np.linspace(x_low, x_hi, 400)
        v_ax.plot(tmp_ica.soc * 100, tmp_ica.volt,
                  label='{}-{}% SOC {} FCE'.format(*soc_lvls, look_up_fce_nrc(key)),
                  linewidth=0.85)
        h_ax.plot(x_int * 100, u_int_chrg(x_int) - u_int_dchg(x_int),
                  label='{}-{}% SOC {} FCE'.format(*soc_lvls, look_up_fce_nrc(key)),
                  linewidth=0.85)


    h_ax.set_xlabel('SOC [%]', weight='bold')
    h_ax.set_ylabel('Voltage hysteresis [V]', weight='bold')
    h_ax.lines[0].set_label('Fresh cell')
    h_ax.legend()
    v_ax.set_xlabel('SOC [%]', weight='bold')
    v_ax.set_ylabel('Voltage [V]', weight='bold')
    v_ax.lines[0].set_label('Fresh cell')
    v_ax.legend()
    hyst_fig.savefig(os.path.join(fig_dir, '{}_hysteresis_updated_FCE.eps'.format(fig_name)),
                     dpi=hyst_fig.dpi)
    hyst_fig.savefig(os.path.join(fig_dir, '{}_hysteresis_updated_FCE.png'.format(fig_name)),
                     dpi=hyst_fig.dpi)
    volt_fig.savefig(os.path.join(fig_dir, '{}_voltage_updated_FCE.eps'.format(fig_name)),
                     dpi=hyst_fig.dpi)
    volt_fig.savefig(os.path.join(fig_dir, '{}_voltage_updated_FCE.png'.format(fig_name)),
                     dpi=hyst_fig.dpi)

if __name__ == '__main__':
    # data_location = r"\\sol.ita.chalmers.se\groups\batt_lab_data\20210216"
    data_location = r"\\sol.ita.chalmers.se\groups\batt_lab_data\ALINE_data\10_dod_data"
    fig_dir = r"\\sol.ita.chalmers.se\groups\batt_lab_data\analysis_directory\TeslaSocAgeingPaper_updated"
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    data_set = OrganiseRpts(data_location, proj='aline')

    # data_set.plot_soc_volt(test_name='5 to 15 SOC', rpt_num=['rpt_1', 'rpt_10'], cell_nbr=['1'])
    # data_set.plot_hysteresis(test_name='3600s', rpt_num=['rpt_1', 'rpt_4', 'rpt_8'], cell_nbr='1')

    data_pts_0_100 = ['rpt_1', 'rpt_4', 'rpt_8']
    data_pts_5_15 = ['rpt_1', 'rpt_4', 'rpt_8']
    data_pts_85_95 = ['rpt_1', 'rpt_4', 'rpt_8', 'rpt_11']
    plt_data_0100 = {k: data_set.ica_dict['3600s_4_3'][k] for k in data_pts_0_100}
    plt_data_5_15 = {k: data_set.ica_dict['5 to 15 SOC_2_1'][k] for k in data_pts_5_15}
    plt_data_85_95 = {k: data_set.ica_dict['85 to 95 SOC_4_2'][k] for k in data_pts_5_15}
    plt.style.use('seaborn-bright')
    my_plot_fun(plt_data_5_15, '5_15_SOC')
    my_plot_fun(plt_data_0100, '0_100_SOC')
    my_plot_fun(plt_data_85_95, '85_95_SOC')
    # plt.close('all')


    hyst_fig, h_ax = plt.subplots(1, 1, figsize=(8, 6))
    volt_fig, v_ax = plt.subplots(1, 1, figsize=(8, 6))
    gauss_fig, g_ax = plt.subplots(1, 1, figsize=(14, 9))
    comb_data = {
        '5to15_rpt_1': data_set.ica_dict['5 to 15 SOC_2_1']['rpt_1'],
        '5to15_rpt_8': data_set.ica_dict['5 to 15 SOC_2_1']['rpt_8'],
        '85to95_rpt_8': data_set.ica_dict['85 to 95 SOC_4_2']['rpt_8']
    }
    for key in comb_data:
        tmp_ica = comb_data[key]
        soc_lvls = re.findall(r'\d+', key)
        rpt_str = re.search(r'rpt_\d', key).group()
        soc = (tmp_ica.mAh - tmp_ica.mAh.min()) / (tmp_ica.mAh.max() - tmp_ica.mAh.min())
        tmp_ica.loc[:, 'soc'] = soc
        u_int_chrg = interp1d(tmp_ica[tmp_ica.curr > 0].soc, tmp_ica[tmp_ica.curr > 0].volt)
        u_int_dchg = interp1d(tmp_ica[tmp_ica.curr < 0].soc, tmp_ica[tmp_ica.curr < 0].volt)
        x_low = max(tmp_ica[tmp_ica.curr > 0].soc.min(), tmp_ica[tmp_ica.curr < 0].soc.min())
        x_hi = min(tmp_ica[tmp_ica.curr > 0].soc.max(), tmp_ica[tmp_ica.curr < 0].soc.max())
        x_int = np.linspace(x_low, x_hi, 400)
        v_ax.plot(tmp_ica.soc * 100, tmp_ica.volt,
                  label='{}-{}% SOC {} FCE'.format(*soc_lvls[:2], look_up_fce_nrc(rpt_str)),
                  linewidth=0.85)
        h_ax.plot(x_int * 100, u_int_chrg(x_int) - u_int_dchg(x_int),
                  label='{}-{}% SOC {} FCE'.format(*soc_lvls[:2], look_up_fce_nrc(rpt_str)),
                  linewidth=0.85)
        g_ax.plot(tmp_ica.volt, tmp_ica.ica_gauss,
                  label='{}-{}% SOC {} FCE'.format(*soc_lvls[:2], look_up_fce_nrc(rpt_str)),
                  linewidth=0.85)
    v_ax.set_xlabel('SOC [%]', weight='bold')
    v_ax.set_ylabel('Voltage [V]', weight='bold')
    v_ax.lines[0].set_label('Fresh cell')
    v_ax.legend()
    h_ax.set_xlabel('SOC [%]', weight='bold')
    h_ax.set_ylabel('Voltage hysteresis [V]', weight='bold')
    h_ax.lines[0].set_label('Fresh cell')
    h_ax.legend()
    g_ax.set_xlabel('Voltage [V]', weight='bold')
    g_ax.set_ylabel('Incremental capacity dQ/dV [V/Ah]', weight='bold')
    g_ax.lines[0].set_label('Fresh cell')
    g_ax.legend()

    res_fig, r_ax = plt.subplots(1, 1, figsize=(10, 8))
    for test_case in data_set.summary_dict:
        data_set.summary_dict[test_case].data.loc[:, 'FCE'] = [look_up_fce_nrc(k)
                                                               for k in data_set.summary_dict[test_case].data.index]
        if 'SOC' in test_case:
            soc_lvls = re.findall(r'\d+', test_case)
            df_temp = data_set.summary_dict[test_case].data
            if '15' in soc_lvls:
                c = 'red'
            elif '85' in soc_lvls:
                c = 'blue'
            else:
                c = 'maroon'
            r_ax.plot(df_temp[df_temp['FCE'] < 1900]['FCE'],
                      df_temp[df_temp['FCE'] < 1900]['res_dchg_50_relative'] * 100,
                      label='{}-{}% SOC'.format(*soc_lvls[:2]),
                      color=c,
                      marker='o',
                      markersize=9,
                      fillstyle='none')
    r_ax.set_xlabel('Number of Full Cycle Equivalents (FCE)', weight='bold')
    r_ax.set_ylabel('Relative 10s discharge resistance (%)', weight='bold')
    r_ax.legend()
    r_ax.set_xticks(np.arange(0, 1900, step=200))


    hyst_fig.savefig(os.path.join(fig_dir, 'SOC5-15_and_SOC85-95_hysteresis_comparison.eps'),
                     dpi=hyst_fig.dpi)
    hyst_fig.savefig(os.path.join(fig_dir, 'SOC5-15_and_SOC85-95_hysteresis_comparison.png'),
                     dpi=hyst_fig.dpi)
    volt_fig.savefig(os.path.join(fig_dir, 'SOC5-15_and_SOC85-95_voltage_comparison.eps'),
                     dpi=hyst_fig.dpi)
    volt_fig.savefig(os.path.join(fig_dir, 'SOC5-15_and_SOC85-95_voltage_comparison.png'),
                     dpi=hyst_fig.dpi)
    gauss_fig.savefig(os.path.join(fig_dir, 'SOC5-15_and_SOC85-95_ica.png'),
                      dpi=gauss_fig.dpi)
    res_fig.savefig(os.path.join(fig_dir, 'dchg_res_50SOC_10s_test_comparison.png'),
                    dpi=res_fig.dpi)
    res_fig.savefig(os.path.join(fig_dir, 'dchg_res_50SOC_10s_test_comparison.eps'),
                    dpi=res_fig.dpi)


