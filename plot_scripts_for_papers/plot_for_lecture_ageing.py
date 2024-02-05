import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rpt_data_analysis.ReadRptClass import OrganiseRpts
from test_data_analysis.ica_analysis import gaussianfilterint, remove_faulty_points
from plot_scripts_for_papers.aline_tesla_plots_anastasiia import build_df_from_tmp_dict
import datetime as dt
import scipy.constants
import os


def cell_prop_dict_lrg_soc():
    cell_dict_upd = {
        '644': ['0 to 50 SOC high temp_5_1', '0-50% SOC', '45$^\circ$C', 'H0-50', '#966B9D', 1],
        '644_bk': ['0 to 50 SOC high temp_5_3', '0-50% SOC', '45$^\circ$C', 'H0-50', '#966B9D', 1],
        '648': ['0 to 100 SOC high temp_5_6', '0-100% SOC', '45$^\circ$C', '0-100% SOC', '#26021B', 1],
        '648_bk': ['0 to 100 SOC high temp_5_2', '0-100% SOC', '45$^\circ$C', 'H0-100', '#26021B', 1],
        '673': ['50 to 100 SOC room temp_4_6', '50-100% SOC', 'room temp', 'R50-100', '#1E7099', 1],
        '673_bk': ['50 to 100 SOC room temp_4_5', '50-100% SOC', 'room temp', 'R50-100', '#1E7099', 1],
        '643': ['50 to 100 SOC high temp_5_4', '50-100% SOC', '45$^\circ$C', '50-100% SOC', '#5C2751', 1],
        '643_bk': ['50 to 100 SOC high temp_5_5', '50-100% SOC', '45$^\circ$C', 'H50-100', '#5C2751', 1],
        '682': ['RPT_240119_4_7', '0-50% SOC', 'room temp', 'R0-50', '#6EA4BF', 1],
        '682_bk': ['0 to 50 SOC room temp_4_8', '0-50% SOC', 'room temp', 'R0-50', '#6EA4BF', 1],
        '685': ['0 to 100 SOC room temp_4_3', '0-100% SOC', 'room temp', 'R0-100', '#274654', 1],
        '685_bk': ['0 to 100 SOC room temp_4_4', '0-100% SOC', 'room temp', 'R0-100', '#274654', 1]
    }
    return cell_dict_upd


def cell_prop_dict_sml_soc():
    cell_info_dict = {
        '5 to 15 SOC': ['5 to 15 SOC_2_1', np.array((170, 111, 158)) / 255, '5-15% SOC'],
        '5 to 15 SOC cell 2': ['5 to 15 SOC_2_2', np.array((170, 111, 158)) / 255, '5-15% SOC'],
        '15 to 25 SOC': ['15 to 25 SOC_240119_2_4', np.array((136, 46, 114)) / 255, '15-25% SOC'],
        '15 to 25 SOC cell 2': ['15 to 25 SOC_2_3', np.array((136, 46, 114)) / 255, '15-25% SOC'],
        '25 to 35 SOC': ['25 to 35 SOC_2_6', np.array((67, 125, 191)) / 255, '25-35% SOC'],
        '25 to 35 SOC cell 2': ['25 to 35 SOC_240119_2_5', np.array((67, 125, 191)) / 255, '25-35% SOC'],
        '35 to 45 SOC': ['35 to 45 SOC_2_7', np.array((123, 175, 222)) / 255, '35-45% SOC'],
        '35 to 45 SOC cell 2': ['35 to 45 SOC_2_8', np.array((123, 175, 222)) / 255, '35-45% SOC'],
        '45 to 55 SOC': ['45 to 55 SOC_3_1', np.array((144, 201, 135)) / 255, '45-55% SOC'],
        '45 to 55 SOC cell 2': ['45 to 55 SOC_3_2', np.array((144, 201, 135)) / 255, '45-55% SOC'],
        '55 to 65 SOC': ['55 to 65 SOC_3_3', np.array((247, 240, 86)) / 255, '55-65% SOC'],
        '55 to 65 SOC cell 2': ['55 to 65 SOC_3_4', np.array((247, 240, 86)) / 255, '55-65% SOC'],
        '65 to 75 SOC': ['65 to 75 SOC_3_5', np.array((244, 167, 54)) / 255, '65-75% SOC'],
        '65 to 75 SOC cell 2': ['65 to 75 SOC_3_6', np.array((244, 167, 54)) / 255, '65-75% SOC'],
        '75 to 85 SOC': ['75 to 85 SOC_3_7', np.array((230, 85, 24)) / 255, '75-85% SOC'],
        '75 to 85 SOC cell 2': ['75 to 85 SOC_3_8', np.array((230, 85, 24)) / 255, '75-85% SOC'],
        '85 to 95 SOC': ['85 to 95 SOC_4_1', np.array((165, 23, 14)) / 255, '85-95% SOC'],
        '85 to 95 SOC cell 2': ['85 to 95 SOC_4_2', np.array((165, 23, 14)) / 255, '85-95% SOC']
    }
    return cell_info_dict


if __name__ == '__main__':
    data_set_location = r"\\sol.ita.chalmers.se\groups\batt_lab_data\ALINE_data\50_dod_data"
    op_location = r"Z:\Documents\Papers\LicentiateThesis\images"
    data = OrganiseRpts(data_set_location, proj='aline')
    sml_soc = OrganiseRpts(r"\\sol.ita.chalmers.se\groups\batt_lab_data\ALINE_data\10_dod_data", proj='aline')
    plt.rcParams['figure.figsize'] = 9, 6
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['lines.linewidth'] = 1.7
    plt.rcParams['xtick.labelsize'] = 'x-large'
    plt.rcParams['ytick.labelsize'] = 'x-large'
    lbl_font = {'weight': 'normal',
                'size': 20}

    mrk_font = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 15}

    mrk_font_neg = {'family': 'Times New Roman',
                    'weight': 'bold',
                    'size': 15,
                    'color': 'white'}

    lrg_soc_prop = cell_prop_dict_lrg_soc()
    sml_soc_prop = cell_prop_dict_sml_soc()

    cell_list_lrg = ['648', '643']
    cell_list_sml = [
        '15 to 25 SOC',
        '45 to 55 SOC',
        '85 to 95 SOC'
    ]
    ar = 3 / 4
    x_width = 9.5
    qe_fig, qe_ax = plt.subplots(1, 1, figsize=(x_width, ar*x_width))
    mol_mark = 0
    all_cell_dict = {}
    for c in cell_list_lrg:
        tmp_q_dict = {}
        for c_name, lst in lrg_soc_prop.items():
            if c in c_name:
                tmp_q_dict[c_name] = data.summary_dict[lst[0]].data
        qdf = build_df_from_tmp_dict(tmp_q_dict)
        all_cell_dict[c] = qdf
        qe_ax.errorbar(qdf.loc[:, f'fce_{c}'], qdf['y_mean']*100, qdf['y_err'].abs()*100,
                    color=lrg_soc_prop[c][4],
                    alpha=lrg_soc_prop[c][5],
                    elinewidth=1.5,
                    marker='s',
                    capsize=6,
                    label=lrg_soc_prop[c][3])
    for c in cell_list_sml:
        tmp_q_dict = {}
        for c_name, lst in sml_soc_prop.items():
            if c in c_name:
                tmp_q_dict[c_name] = sml_soc.summary_dict[lst[0]].data
        df = build_df_from_tmp_dict(tmp_q_dict)
        all_cell_dict[c] = df
        qe_ax.errorbar(df.loc[:, f'FCE_{c}'], df['y_mean'] * 100, df['y_err'].abs() * 100,
                       color=sml_soc_prop[c][1],
                       elinewidth=1.5,
                       marker='s',
                       capsize=6,
                       label=sml_soc_prop[c][2])

    qe_ax.legend(loc='upper right', ncols=2)
    qe_ax.set_xlabel('Number of Full Cycle Equivalents, FCE', fontdict=lbl_font)
    qe_ax.set_ylabel('Percentage of Capacity Retained, %', fontdict=lbl_font)
    qe_ax.set_ylim(70, 105)
    qe_fig.savefig(os.path.join(op_location,
                                f'CapacityRetention_w_errorbar_small_and_large_soc_{dt.datetime.now():%Y_%m_%d}.png'),
                   dpi=400)
    qe_fig.savefig(os.path.join(op_location,
                                f'CapacityRetention_w_errorbar_small_and_large_soc_{dt.datetime.now():%Y_%m_%d}.pdf'),
                   dpi=400)

    R = scipy.constants.value('molar gas constant')
    Ea = 105e3
    A = np.exp(23)
    T_arr = np.arange(263.15, 313.15, 0.01)
    my_arr_fun = lambda T_r: A * np.exp(-Ea / (R * T_r))

    x_width = 9
    plt.figure(figsize=(x_width, ar * x_width))

    plt.style.use('kelly_colors')
    plt.plot(T_arr, my_arr_fun(T_arr) * 1e6, linewidth=3, linestyle='dashed')
    plt.xlabel('Temperature [K]')
    plt.ylabel(r'Rate $k$ [cm$^3$ mol$^{-1}$ s$^{-1}$]')
    plt.savefig(os.path.join(op_location, 'arrhenius_eq_schematic.png'), dpi=400)
