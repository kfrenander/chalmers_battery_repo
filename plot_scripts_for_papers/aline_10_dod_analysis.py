import pandas as pd
from PythonScripts.rpt_data_analysis.ReadRptClass import OrganiseRpts
from PythonScripts.backend_fix import fix_mpl_backend
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from scipy.interpolate import interp1d
fix_mpl_backend()
kth_compatible = 1
if kth_compatible:
    x_width = 3.25
    aspect_rat = 3 / 4
    plt.rcParams['figure.figsize'] = x_width, aspect_rat * x_width
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    lbl_font = {'weight': 'normal',
                'size': 9}
    plt.rc('legend', fontsize=8)
    plt.rc('font', **{"family": 'sans-serif', 'sans-serif': 'Helvetica'})
    mark_size = 2
    cap_size = 3
    peak_mark_size = 6.5
    output_dir = r"Z:\Provning\Analysis\ALINE_plots\small_soc\after_review_kth_size_dash_legend"
else:
    x_width = 8
    aspect_rat = 12 / 16
    plt.rcParams['figure.figsize'] = x_width, aspect_rat * x_width
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['lines.linewidth'] = 1.7
    plt.rcParams['xtick.labelsize'] = 'x-large'
    plt.rcParams['ytick.labelsize'] = 'x-large'
    lbl_font = {'weight': 'normal',
                'size': 20}
    plt.rc('legend', fontsize=14)
    peak_mark_size = 15
    mark_size = 5
    cap_size = 6
    output_dir = r"Z:\Provning\Analysis\ALINE_plots\small_soc\after_review_comments"


def color_coding_max_contrast(test_name, col_format='hex'):
    list_of_tests = ['5 to 15 SOC',
                        '15 to 25 SOC',
                        '25 to 35 SOC',
                        '35 to 45 SOC',
                        '45 to 55 SOC',
                        '55 to 65 SOC',
                        '65 to 75 SOC',
                        '75 to 85 SOC',
                        '85 to 95 SOC'
                    ]
    list_of_colors_hex = ['#e6194B',
                        '#3cb44b',
                        '#ffe119',
                        '#4363d8',
                        '#f58231',
                        '#42d4f4',
                        '#f032e6',
                        '#fabed4',
                        '#000000']
    list_of_colors_rgb = np.array([(230, 25, 75),
                        (60, 180, 75),
                        (255, 225, 25),
                        (0, 130, 200),
                        (245, 130, 48),
                        (70, 240, 240),
                        (240, 50, 230),
                        (250, 190, 212),
                        (0, 0, 0)
                        ]) / 255
    if col_format == 'hex':
        col_dict = dict(zip(list_of_tests, list_of_colors_hex))
    elif col_format == 'rgb':
        col_dict = dict(zip(list_of_tests, list_of_colors_rgb))
    return col_dict[test_name.split('_')[0]]


def look_up_fce_nrc(rpt_str):
    rpt_num = int(re.search(r'\d+', rpt_str).group())
    x_rpt = np.arange(1, 14)
    y_fce = np.array([0, 150, 300, 450, 600, 750, 900,
                      1200, 1350, 1500, 1650, 1800, 2100])
    fce = int(interp1d(x_rpt, y_fce)(rpt_num))
    return fce


def build_df_from_tmp_dict(tmp_dict, ageing_case='cycling'):
    if ageing_case == 'cycling':
        cols_to_include = ['cap', 'cap_relative', 'date', 'res_dchg_50_relative', 'egy_thrg', 'FCE']
    elif ageing_case == 'storage':
        cols_to_include = ['cap', 'cap_relative', 'date', 'res_dchg_50', 'res_dchg_50_relative']
    df = pd.DataFrame()
    for k in tmp_dict:
        for col_name in cols_to_include:
            df[f'{col_name}_{k}'] = tmp_dict[k][col_name]
        df.loc[:, 'time_stamp'] = pd.to_datetime(df[f'date_{k}'])
        if ageing_case == 'storage':
            df = df[df.time_stamp.diff().fillna(dt.timedelta(days=30)) > dt.timedelta(days=10)]
    df.loc[:, 'y_mean'] = df.filter(regex='cap_relative').mean(axis=1)
    df.loc[:, 'y_err'] = df.filter(regex='cap_relative').diff(axis=1).dropna(axis=1, how='all') / 2
    df = df[df['y_err'].notna()]
    return df


def calc_hysteresis(ica_data):
    soc = (ica_data.mAh - ica_data.mAh.min()) / (ica_data.mAh.max() - ica_data.mAh.min())
    ica_data.loc[:, 'soc'] = soc
    u_int_chrg = interp1d(ica_data[ica_data.curr > 0].soc, ica_data[ica_data.curr > 0].volt)
    u_int_dchg = interp1d(ica_data[ica_data.curr < 0].soc, ica_data[ica_data.curr < 0].volt)
    x_low = max(ica_data[ica_data.curr > 0].soc.min(), ica_data[ica_data.curr < 0].soc.min())
    x_hi = min(ica_data[ica_data.curr > 0].soc.max(), ica_data[ica_data.curr < 0].soc.max())
    x_int = np.linspace(x_low, x_hi, 400)
    y_hyst = u_int_chrg(x_int) - u_int_dchg(x_int)
    return x_int, y_hyst


def color_coding_kth(test_name):
    list_of_tests = ['5 to 15 SOC',
                    '15 to 25 SOC',
                    '25 to 35 SOC',
                    '35 to 45 SOC',
                    '45 to 55 SOC',
                    '55 to 65 SOC',
                    '65 to 75 SOC',
                    '75 to 85 SOC',
                    '85 to 95 SOC'
                    ]
    list_of_colors_rgb = np.array([(170, 111, 158),
                                   (136, 46, 114),
                                   (67, 125, 191),
                                   (123, 175, 222),
                                   (144, 201, 135),
                                   (247, 240, 86),
                                   (244, 167, 54),
                                   (230, 85, 24),
                                   (165, 23, 14)
                                   ]) / 255
    col_dict = dict(zip(list_of_tests, list_of_colors_rgb))
    return col_dict[test_name.split('_')[0]]


def find_soh(full_data, test_name, soh_level):
    df = full_data.summary_dict[test_name].data
    nearest = df.iloc[(df['cap_relative']-soh_level).abs().argsort()[:1]]
    print(f"Nearest capacity found was {nearest['cap_relative'][0]:.2f} with target of {soh_level}.\nCase {test_name}")
    return nearest.index[0]


def find_soh_at_fce(full_data, test_name, fce):
    df = full_data.summary_dict[test_name].data
    nearest = df.iloc[(df['FCE'] - fce).abs().argsort()[:1]]
    return nearest['cap_relative'][0]


def peak_name_def(peak_mark_switch):
    if peak_mark_switch:
        return 'peaks_marked'
    else:
        return 'no_peaks'


rpt_data = OrganiseRpts(r"\\sol.ita.chalmers.se\groups\batt_lab_data\ALINE_data\10_dod_data", proj='aline')
lrg_dod_data = OrganiseRpts(r"\\sol.ita.chalmers.se\groups\batt_lab_data\ALINE_data\50_dod_data", proj='aline')
cal_data = OrganiseRpts(r"\\sol.ita.chalmers.se\groups\batt_lab_data\ALINE_data\CalendarData", proj='aline')
bol_data = r"\\sol.ita.chalmers.se\groups\batt_lab_data\ALINE_data\bol_ref_data.pkl"
bol_data_dva = r"\\sol.ita.chalmers.se\groups\batt_lab_data\Ica_files\bol_ica_data_filtered.csv"


if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

mrk_font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': peak_mark_size}

mrk_font_neg = {'family': 'Times New Roman',
                'weight': 'bold',
                'size': peak_mark_size,
                'color': 'white'}




ica_set = ['5 to 15 SOC_2_2',
            '15 to 25 SOC_240119_2_4',
            '85 to 95 SOC_4_1']

my_tests = ['5 to 15 SOC_2_1',
            '15 to 25 SOC_240119_2_4',
            '25 to 35 SOC_2_6',
            '35 to 45 SOC_2_7',
            '45 to 55 SOC_3_1',
            '55 to 65 SOC_3_3',
            '65 to 75 SOC_3_5',
            '75 to 85 SOC_3_7',
            '85 to 95 SOC_4_1']

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

cell_info_cal = {
    '15 SOC': ['Storage 15 SOC_1_1', np.array((170, 111, 158)) / 255, '15 % SOC'],
    '15 SOC cell 2': ['Storage 15 SOC_1_2', np.array((170, 111, 158)) / 255, '15 % SOC'],
    '50 SOC': ['Storage 50 SOC_1_3', np.array((144, 201, 135)) / 255, '50 % SOC'],
    '50 SOC cell 2': ['Storage 50 SOC_1_4', np.array((144, 201, 135)) / 255, '50 % SOC'],
    '85 SOC': ['Storage 85 SOC_1_5', np.array((165, 23, 14)) / 255, '85 % SOC'],
    '85 SOC cell 2': ['Storage 85 SOC_240119_1_6', np.array((165, 23, 14)) / 255, '85 % SOC']
}

cal_test_list = [
    '15 SOC',
    '50 SOC',
    '85 SOC'
]

list_of_test_names = [
    '5 to 15 SOC',
    '15 to 25 SOC',
    '25 to 35 SOC',
    '35 to 45 SOC',
    '45 to 55 SOC',
    '55 to 65 SOC',
    '65 to 75 SOC',
    '75 to 85 SOC',
    '85 to 95 SOC'
]

list_of_test_names_mid = [
    '35 to 45 SOC',
    '45 to 55 SOC'
]


subset_tests = [
    '25 to 35 SOC',
    '35 to 45 SOC',
    '45 to 55 SOC',
    '55 to 65 SOC',
    '65 to 75 SOC',
    '75 to 85 SOC',
    '85 to 95 SOC'
]

marker_list = ['v', '.', 's', '^', '*']
style_list = ['solid', 'dashed', 'dashdot', (0, (1, 1))]
dash_list = [[4, 0], [4, 1], [1, 1, 0.5, 1]]
fig, ax = plt.subplots(1, 1)
for i, t in enumerate(my_tests):
    ax.plot(rpt_data.summary_dict[t].data['FCE'],
            rpt_data.summary_dict[t].data['cap_relative'],
            marker='*',
            linestyle=style_list[i % len(style_list)],
            linewidth=0.9,
            label=t.split('_')[0],
            color=color_coding_kth(t))
ax.legend(ncols=2)
ax.set_xlabel('Number of Full Cycle Equivalents / -', fontdict=lbl_font)
ax.set_ylabel('Percentage of Capacity Retained / -', fontdict=lbl_font)
fig.savefig(os.path.join(output_dir, 'capacity_retention.pdf'))

qe_fig, qe_ax = plt.subplots(1, 1)
all_cell_dict = {}
for c in list_of_test_names:
    tmp_q_dict = {}
    for c_name in cell_info_dict:
        if c in c_name:
            tmp_q_dict[c_name] = rpt_data.summary_dict[cell_info_dict[c_name][0]].data
    df = build_df_from_tmp_dict(tmp_q_dict)
    all_cell_dict[c] = df
    qe_ax.errorbar(df.loc[:, f'FCE_{c}'], df['y_mean'] * 100, df['y_err'].abs() * 100,
                   color=cell_info_dict[c][1],
                   elinewidth=1.5,
                   marker='s',
                   markersize=mark_size,
                   capsize=cap_size,
                   label=cell_info_dict[c][2])
# qe_ax.grid(False)
qe_ax.legend(loc='upper right', ncols=2)
qe_ax.set_xlabel('Number of Full Cycle Equivalents / -', fontdict=lbl_font, labelpad=2)
qe_ax.set_ylabel('Percentage of Capacity Retained / -', fontdict=lbl_font, labelpad=2)
qe_ax.set_ylim(77, 106)
qe_ax.set_xlim(-20, 4550)
qe_ax.grid(color='grey', alpha=0.5)
qe_fig.subplots_adjust(bottom=0.15)
qe_fig.subplots_adjust(left=0.15)
qe_fig.savefig(os.path.join(output_dir, 'capacity_retention_w_errorbar.pdf'))
qe_fig.savefig(os.path.join(output_dir, 'capacity_retention_w_errorbar.png'), dpi=300, transparent=True)
qe_fig.savefig(os.path.join(output_dir, 'capacity_retention_w_errorbar.eps'), dpi=300)


qe_sub_fig, qe_sub_ax = plt.subplots(1, 1)
for c in subset_tests:
    df = all_cell_dict[c]
    qe_sub_ax.errorbar(df.loc[:, f'FCE_{c}'], df['y_mean'] * 100, df['y_err'].abs() * 100,
                   color=cell_info_dict[c][1],
                   elinewidth=1.5,
                   marker='s',
                   markersize=mark_size,
                   capsize=cap_size,
                   label=cell_info_dict[c][2])
# qe_sub_ax.grid(False)
qe_sub_ax.legend(loc='upper right', ncols=2)
qe_sub_ax.set_xlabel('Number of Full Cycle Equivalents / -', fontdict=lbl_font)
qe_sub_ax.set_ylabel('Percentage of Capacity Retained / -', fontdict=lbl_font)
qe_sub_ax.set_ylim(77, 104)
qe_sub_ax.set_xlim(-20, 4550)
qe_sub_ax.grid(color='grey', alpha=0.5)
qe_sub_fig.savefig(os.path.join(output_dir, 'capacity_retention_subset_w_errorbar.pdf'))
qe_sub_fig.savefig(os.path.join(output_dir, 'capacity_retention_subset_w_errorbar.png'), dpi=300)
qe_sub_fig.savefig(os.path.join(output_dir, 'capacity_retention_subset_w_errorbar.eps'), dpi=300)


q_cal_fig, q_cal_ax = plt.subplots(1, 1)
cal_cell_dict = {}
for c in cal_test_list:
    tmp_q_dict = {}
    for c_name in cell_info_cal:
        if c in c_name:
            tmp_q_dict[c_name] = cal_data.summary_dict[cell_info_cal[c_name][0]].data
    df = build_df_from_tmp_dict(tmp_q_dict, ageing_case='storage')
    df.loc[:, 'time_as_date'] = pd.to_datetime(df.loc[:, f'date_{c}'])
    df.loc[:, 'rel_time'] = (df.time_as_date - df.time_as_date[0]).astype('timedelta64[D]')
    cal_cell_dict[c] = df
    q_cal_ax.errorbar(df.loc[:, f'rel_time'], df['y_mean'] * 100, df['y_err'].abs() * 100,
                   color=cell_info_cal[c][1],
                   elinewidth=1.5,
                   marker='s',
                   markersize=5,
                   capsize=6,
                   label=cell_info_cal[c][2])
q_cal_ax.legend()
q_cal_ax.set_xlabel('Time / days', fontdict=lbl_font)
q_cal_ax.set_ylabel('Percentage of Capacity Retained / -', fontdict=lbl_font)
q_cal_ax.set_ylim(88, 102)
q_cal_ax.set_xlim(-10, 750)
q_cal_ax.grid(color='grey', alpha=0.5)
q_cal_fig.savefig(os.path.join(output_dir, 'capacity_retention_calendar.pdf'))
q_cal_fig.savefig(os.path.join(output_dir, 'capacity_retention_calendar.png'), dpi=300)
q_cal_fig.savefig(os.path.join(output_dir, 'capacity_retention_calendar.eps'), dpi=300)

fig1, ax1 = plt.subplots(3, 1, figsize=(x_width, 9.5 / 4 * x_width),  gridspec_kw={'height_ratios': [3.5, 3, 3]})
for c in list_of_test_names:
    df = all_cell_dict[c]
    ax1[0].errorbar(df.loc[:, f'FCE_{c}'], df['y_mean'] * 100, df['y_err'].abs() * 100,
                   color=cell_info_dict[c][1],
                   elinewidth=1,
                   marker='s',
                   markersize=mark_size,
                   capsize=cap_size,
                   label=cell_info_dict[c][2])
#ax1[0].legend(loc='upper right', ncols=2, fontsize=7.5)
fig1.legend(ncols=2, fontsize=8.5)
ax1[0].set_xlabel('Number of Full Cycle Equivalents / -', fontdict=lbl_font, labelpad=2)
ax1[0].set_ylabel('Capacity Retention / %', fontdict=lbl_font, labelpad=2)
ax1[0].set_ylim(77, 106)
ax1[0].set_xlim(-20, 4550)
ax1[0].grid(color='grey', alpha=0.5)
ax1[0].set_yticks(np.arange(80, 105, 5))

x_bars = np.arange(10, 100, 10)
y_bars = np.array([15.25, 12.55, 9.05, 5.85, 5.3, 8.35, 9.1, 9, 9.4])
y_bar_err = np.array([0.5, 0.7, 0.5, 0.5, 0, 1.1, 1.2, 0.2, 0.6]) / 2
color_list = [cell_info_dict[c][1] for c in cell_info_dict][0::2]
ax1[1].bar(x_bars, y_bars,
           width=10,
           yerr=y_bar_err,
           color=color_list,
           edgecolor="none",
           capsize=cap_size)
ax1[1].set_xlabel('SOC / %', fontdict=lbl_font, labelpad=2)
ax1[1].set_ylabel('Capacity loss at 1200 FCE / %', fontdict=lbl_font, labelpad=2)
ax1[1].set_yticks([0, 5, 10, 15])
ax1[1].set_xticks(np.arange(5, 100, 10))
ax1[1].grid(color='grey', alpha=0.5)

for c in cal_test_list:
    df = cal_cell_dict[c]
    ax1[2].errorbar(df.loc[:, f'rel_time'], df['y_mean'] * 100, df['y_err'].abs() * 100,
                   color=cell_info_cal[c][1],
                   elinewidth=1,
                   marker='s',
                   markersize=mark_size,
                   capsize=cap_size,
                   label=cell_info_cal[c][2])
ax1[2].legend(fontsize=9)
ax1[2].set_xlabel('Time / days', fontdict=lbl_font, labelpad=2)
ax1[2].set_ylabel('Capacity Retention / %', fontdict=lbl_font, labelpad=2)
ax1[2].set_ylim(88, 102)
ax1[2].set_xlim(-10, 750)
ax1[2].grid(color='grey', alpha=0.5)
ax1[2].set_yticks(np.arange(88, 102, 2))
fig1.subplots_adjust(bottom=0.05)
fig1.subplots_adjust(top=0.95)
fig1.subplots_adjust(left=0.18)
fig1.subplots_adjust(hspace=0.25)
fig1_labels = ['(a)', '(b)', '(c)']
for k, label in enumerate(fig1_labels):
    ax1[k].text(-0.235, 1.0, label, transform=ax1[k].transAxes, fontsize=9)
fig1.savefig(os.path.join(output_dir, 'figure1_capretention.pdf'))
fig1.savefig(os.path.join(output_dir, 'figure1_capretention.png'), dpi=300)
fig1.savefig(os.path.join(output_dir, 'figure1_capretention.eps'), dpi=300)
# qe_ax.grid(False)
# Define plot positions for ICA peak on reference.

#p_crds_neg = [(3.35, yloc_neg), (3.47, yloc_neg), (3.58, yloc_neg),
#              (3.69, yloc_neg), (3.87, yloc_neg), (3.95, yloc_neg), (4.115, yloc_neg)]
#p_crds_pos = [(3.35, yloc_pos), (3.47, yloc_pos), (3.58, yloc_pos),
#              (3.69, yloc_pos), (3.87, yloc_pos), (3.95, yloc_pos), (4.115, yloc_pos)]
#peak_coords = [(3.36, 3.12), (3.46, 6.64), (3.58, 4.55), (3.66, 6.48), (3.85, 5.67), (3.93, 5.78), (4.1, 11.4)]
#x_coords = [x for x, y in peak_coords]
#y_coords = [y for x, y in peak_coords]

yloc_neg = [13.5]*7
yloc_pos = [17]*7
peaks_pos = [str(x) for x in [4, 4, 3, 3, 2, 2, 1]]
peaks_neg = [str(x) for x in [4, 3, 3, 2, 2, 1, 1]]
x_coords = [3.38, 3.47, 3.587, 3.67, 3.865, 3.95, 4.115]
offset = 0.15
box_props_neg = dict(boxstyle='circle', facecolor='black', alpha=0.9, edgecolor='black')
box_props_pos = dict(boxstyle='circle', facecolor='white', alpha=1, edgecolor='black')
rpt_soh_list = [f'rpt_{i}' for i in range(3, 32, 12)]
bol_ref = pd.read_pickle(bol_data)
bol_ref_dva = pd.read_csv(bol_data_dva, index_col=0)

# Reference figure for ICA with peak marking
ica_ref_fig, ica_ref_ax = plt.subplots(1, 1)
ica_ref_ax.plot(bol_ref[bol_ref.curr > 0]['volt'],
                bol_ref[bol_ref.curr > 0]['ica_gauss'],
                linestyle='solid',
                color='black',
                linewidth=1.5,
                label='Fresh cell')
ica_ref_ax.plot(bol_ref[bol_ref.curr < 0]['volt'],
                bol_ref[bol_ref.curr < 0]['ica_gauss'],
                linestyle='solid',
                linewidth=1.5,
                color='black')
plt.text(3.0, yloc_pos[0], 'NCA reaction', color='red', fontsize=17, weight='bold')
plt.text(3.0, yloc_neg[0], 'Si-Gr reaction', color='blue', fontsize=17, weight='bold')
plt.vlines(x_coords, 0, yloc_neg[0] - 1.5, color='black', linewidth=0.8)
for p, x, y in zip(peaks_pos, x_coords, yloc_pos):
    plt.text(x, y, p, fontdict=mrk_font, bbox=box_props_pos, horizontalalignment='center')
for p, x, y in zip(peaks_neg, x_coords, yloc_neg):
    plt.text(x, y, p, fontdict=mrk_font_neg, bbox=box_props_neg, horizontalalignment='center')
ica_ref_ax.set_xlim(2.9, 4.2)
ica_ref_ax.set_ylim(-15, 20)
ica_ref_ax.grid(False)
ica_ref_ax.set_xlabel('Cell Voltage / V', fontdict=lbl_font)
ica_ref_ax.set_ylabel(r'IC, dQ dV$^{-1}$ / Ah V$^{-1}$', fontdict=lbl_font)
ica_ref_fig.savefig(os.path.join(output_dir, 'ica_reference_marked_reaction_updated_label.pdf'))
ica_ref_fig.savefig(os.path.join(output_dir, 'ica_reference_marked_reaction_updated_label.png'), dpi=300)
ica_ref_fig.savefig(os.path.join(output_dir, 'ica_reference_marked_reaction_updated_label.eps'), dpi=300)


# ICA plot for same cell at different SOH levels, with fresh cell reference
ica_soh_fig, ica_soh_ax = plt.subplots(1, 1)
ica_to_plot = '5 to 15 SOC_2_2'
peak_marking = 0
ica_soh_ax.plot(bol_ref[bol_ref.curr > 0]['volt'],
                bol_ref[bol_ref.curr > 0]['ica_gauss'],
                linestyle='dashed',
                color='black',
                linewidth=1.5,
                label='Fresh cell')
ica_soh_ax.plot(bol_ref[bol_ref.curr < 0]['volt'],
                bol_ref[bol_ref.curr < 0]['ica_gauss'],
                linestyle='dashed',
                linewidth=1.5,
                color='black')

for i, r in enumerate(rpt_soh_list):
    d = rpt_data.ica_dict[ica_to_plot][r]
    d_pos = d[d.curr > 0]
    d_neg = d[d.curr < 0]
    fce = rpt_data.summary_dict[ica_to_plot].data.loc[r, 'FCE']
    ica_soh_ax.plot(d_pos['volt'], d_pos['ica_gauss'],
                    label=f"{fce} FCE",
                    linestyle=style_list[i],
                    linewidth=1.5,
                    color=color_coding_kth(ica_to_plot))
    ica_soh_ax.plot(d_neg['volt'], d_neg['ica_gauss'],
                    color=color_coding_kth(ica_to_plot),
                    linewidth=1.5,
                    linestyle=style_list[i])
ica_soh_ax.set_xlim(2.9, 4.2)
ica_soh_ax.set_xlabel('Cell Voltage / V', fontdict=lbl_font)
ica_soh_ax.set_ylabel(r'IC, dQ dV$^{-1}$ / Ah V$^{-1}$', fontdict=lbl_font)
ica_soh_ax.legend(loc='lower left', fontsize=20)
ica_soh_ax.grid(False)
box_props = dict(boxstyle='round', facecolor='white', alpha=1)
ica_soh_ax.text(3, 10, '5-15% SOC', fontsize=20, bbox=box_props)
ica_soh_fig.savefig(os.path.join(output_dir, 'ica_cycling_5_to_15_marked_updated_label.pdf'))


# ICA plot for different cells at similar SOH
soh_to_show = 0.88
ica_cell_fig, ica_cell_ax = plt.subplots(1, 1)
ica_cell_ax.plot(bol_ref[bol_ref.curr > 0]['volt'],
                bol_ref[bol_ref.curr > 0]['ica_gauss'],
                linestyle='dashed',
                color='black',
                linewidth=1.5,
                label='Fresh cell')
ica_cell_ax.plot(bol_ref[bol_ref.curr < 0]['volt'],
                bol_ref[bol_ref.curr < 0]['ica_gauss'],
                linestyle='dashed',
                linewidth=1.5,
                color='black')
for i, r in enumerate(ica_set):
    rpt_for_soh = find_soh(rpt_data, r, soh_to_show)
    d_pos = rpt_data.ica_dict[r][rpt_for_soh][rpt_data.ica_dict[r][rpt_for_soh]['curr'] > 0]
    d_neg = rpt_data.ica_dict[r][rpt_for_soh][rpt_data.ica_dict[r][rpt_for_soh]['curr'] < 0]
    ica_cell_ax.plot(d_pos.volt,
                    d_pos.ica_gauss,
                    linestyle='solid',
                    color=color_coding_kth(r),
                    label=f"{r.split('_')[0]}"
                    )
    ica_cell_ax.plot(d_neg.volt,
                    d_neg.ica_gauss,
                    linestyle='solid',
                    color=color_coding_kth(r)
                    )
ica_cell_ax.legend(loc='lower left')
ica_cell_ax.set_xlim(2.9, 4.2)
ica_cell_ax.text(3, 10, f"SOH {round(soh_to_show*100, -1):.0f}%", fontsize=20, bbox=box_props)
ica_cell_ax.set_xlabel('Cell Voltage / V', fontdict=lbl_font)
ica_cell_ax.set_ylabel(r'IC, dQ dV$^{-1}$ / Ah V$^{-1}$', fontdict=lbl_font)
ica_cell_ax.grid(False)

ica_cell_fig.savefig(os.path.join(output_dir, 'ica_soh90_multiple_cells_marked_updated_label.pdf'))

# Make ICA plots on single figure
fig5, ax5 = plt.subplots(2, 1, figsize=(x_width, 6 / 4 * x_width),  gridspec_kw={'height_ratios': [1, 1]})
ica_to_plot = '5 to 15 SOC_2_2'
peak_marking = 1
ax5[0].plot(bol_ref[bol_ref.curr > 0]['volt'],
                bol_ref[bol_ref.curr > 0]['ica_gauss'],
                dashes=[4, 1],
                color='black',
                linewidth=.85,
                label='Fresh cell')
ax5[0].plot(bol_ref[bol_ref.curr < 0]['volt'],
                bol_ref[bol_ref.curr < 0]['ica_gauss'],
                dashes=[4, 1],
                linewidth=.85,
                color='black')

for i, r in enumerate(rpt_soh_list):
    d = rpt_data.ica_dict[ica_to_plot][r]
    d_pos = d[d.curr > 0]
    d_neg = d[d.curr < 0]
    fce = rpt_data.summary_dict[ica_to_plot].data.loc[r, 'FCE']
    ax5[0].plot(d_pos['volt'], d_pos['ica_gauss'],
                    label=f"{fce} FCE",
                    dashes=dash_list[i],
                    linewidth=0.85,
                    color=color_coding_kth(ica_to_plot))
    ax5[0].plot(d_neg['volt'], d_neg['ica_gauss'],
                    color=color_coding_kth(ica_to_plot),
                    linewidth=0.85,
                    dashes=dash_list[i])
ax5[0].set_xlim(2.9, 4.2)
ax5[0].set_xlabel('Cell Voltage / V', fontdict=lbl_font, labelpad=1)
ax5[0].set_ylabel(r'IC, dQ dV$^{-1}$ / Ah V$^{-1}$', fontdict=lbl_font, labelpad=2)
ax5[0].legend(loc='lower left', fontsize=8)
ax5[0].grid(False)
box_props = dict(boxstyle='round', facecolor='white', alpha=1)
if peak_marking:
    ax5[0].text(2.95, 8, '5-15% SOC', fontsize=9, bbox=box_props)
    ax5[0].set_ylim(-18, 20)
else:
    ax5[0].text(3, 10, '5-15% SOC', fontsize=9, bbox=box_props)
    ax5[0].set_ylim(-18, 15)

ax5[1].plot(bol_ref[bol_ref.curr > 0]['volt'],
                bol_ref[bol_ref.curr > 0]['ica_gauss'],
                dashes=[4, 1],
                color='black',
                linewidth=0.85,
                label='Fresh cell')
ax5[1].plot(bol_ref[bol_ref.curr < 0]['volt'],
                bol_ref[bol_ref.curr < 0]['ica_gauss'],
                dashes=[4, 1],
                linewidth=0.85,
                color='black')
for i, r in enumerate(ica_set):
    rpt_for_soh = find_soh(rpt_data, r, soh_to_show)
    d_pos = rpt_data.ica_dict[r][rpt_for_soh][rpt_data.ica_dict[r][rpt_for_soh]['curr'] > 0]
    d_neg = rpt_data.ica_dict[r][rpt_for_soh][rpt_data.ica_dict[r][rpt_for_soh]['curr'] < 0]
    ax5[1].plot(d_pos.volt,
                    d_pos.ica_gauss,
                    linestyle='solid',
                    color=color_coding_kth(r),
                    label=cell_info_dict[r.split('_')[0]][2],
                    linewidth=0.8
                    )
    ax5[1].plot(d_neg.volt,
                    d_neg.ica_gauss,
                    linestyle='solid',
                    color=color_coding_kth(r),
                    linewidth=0.8
                    )
ax5[1].legend(loc='lower left', fontsize=7.5)
ax5[1].set_xlim(2.9, 4.2)
ax5[1].set_ylim(-18, 15)
ax5[1].text(3, 10, f"SOH {round(soh_to_show*100, -1):.0f}%", fontsize=9, bbox=box_props)
ax5[1].set_xlabel('Cell Voltage / V', fontdict=lbl_font, labelpad=1)
ax5[1].set_ylabel(r'IC, dQ dV$^{-1}$ / Ah V$^{-1}$', fontdict=lbl_font, labelpad=2)
ax5[1].grid(False)
if peak_marking:
    for p, x, y in zip(peaks_pos, x_coords, yloc_pos):
        ax5[0].text(x, y, p, fontdict=mrk_font, bbox=box_props_pos, horizontalalignment='center')
    for p, x, y in zip(peaks_neg, x_coords, yloc_neg):
        ax5[0].text(x, y, p, fontdict=mrk_font_neg, bbox=box_props_neg, horizontalalignment='center')
fig5.subplots_adjust(bottom=0.08)
fig5.subplots_adjust(top=0.96)
fig5.subplots_adjust(left=0.17)
fig5.subplots_adjust(hspace=0.25)
fig5_labels = ['(a)', '(b)']
for k, label in enumerate(fig5_labels):
    ax5[k].text(-0.17, 1.0, label, transform=ax5[k].transAxes, fontsize=9)
fig5.savefig(os.path.join(output_dir, f'figure5_icaplots_{peak_name_def(peak_marking)}.pdf'))
fig5.savefig(os.path.join(output_dir, f'figure5_icaplots_{peak_name_def(peak_marking)}.png'), dpi=300)
fig5.savefig(os.path.join(output_dir, f'figure5_icaplots_{peak_name_def(peak_marking)}.eps'), dpi=300)

# DVA Plot at different SOH levels for the same cell
rpt_soh_list = [f'rpt_{i}' for i in [2, 12, 22]]
dva_soh_fig, dva_soh_ax = plt.subplots(1, 1)
dva_soh_ax.plot(bol_ref[bol_ref.curr < 0]['cap'],
                -bol_ref[bol_ref.curr < 0]['dva_gauss'],
                linestyle='dashed',
                color='black',
                label='Fresh Cell')
# dva_soh_ax.plot((bol_ref[bol_ref.curr < 0]['cap'].max() - bol_ref[bol_ref.curr < 0]['cap']) / 1000,
#                 -bol_ref[bol_ref.curr < 0]['dva_gauss'],
#                 linestyle='dashed',
#                 color='black',
#                 label='Fresh Cell')
dva_to_plot = '5 to 15 SOC_2_2'
# dva_to_plot = '85 to 95 SOC_4_1'
for i, r in enumerate(rpt_soh_list):
    d = rpt_data.ica_dict[dva_to_plot][r]
    d_pos = d[d.curr > 0]
    d_neg = d[d.curr < 0]
    fce = rpt_data.summary_dict[dva_to_plot].data.loc[r, 'FCE']
    soh = rpt_data.summary_dict[dva_to_plot].data.loc[r, 'cap_relative']
    dva_soh_ax.plot(d_neg['cap'], -d_neg['dva_gauss'],
                    label=f"FCE {fce:.0f}", # SOH {round(soh * 100):.0f} %
                    linestyle=style_list[i],
                    color=color_coding_kth(dva_to_plot))
    # dva_soh_ax.plot((d_neg['cap'].max() - d_neg['cap']) / 1000, -d_neg['dva_gauss'],
    #                 label=f"FCE {fce:.0f}", # SOH {round(soh * 100):.0f} %
    #                 linestyle=style_list[i],
    #                 color=color_coding_kth(dva_to_plot))
dva_soh_ax.set_ylim(0, 0.75)
dva_soh_ax.set_xlabel('Capacity / mAh', fontdict=lbl_font)
dva_soh_ax.set_ylabel(r'DV, dV dQ$^{-1}$ / V mAh$^{-1}$', fontdict=lbl_font)
dva_soh_ax.text(1720, 0.45, '5-15% SOC', fontsize=20, bbox=box_props)
dva_soh_ax.legend(loc='upper center')
dva_soh_ax.grid(color='grey', alpha=0.5)
dva_soh_fig.subplots_adjust(bottom=0.12)
dva_soh_fig.savefig(os.path.join(output_dir, 'dva_cycling_5_to_15_fce_legend.pdf'))
dva_soh_fig.savefig(os.path.join(output_dir, 'dva_cycling_5_to_15_fce_legend.png'), dpi=300)

# DVA plot for different cells at similar SOH
dva_cell_fig, dva_cell_ax = plt.subplots(1, 1)
dva_cell_ax.plot((bol_ref[bol_ref.curr < 0]['cap'].max() - bol_ref[bol_ref.curr < 0]['cap']) / 1000,
                -bol_ref[bol_ref.curr < 0]['dva_gauss'],
                linestyle='dashed',
                color='black',
                label='Fresh Cell')
for t in ica_set:
    rpt_to_show = find_soh(rpt_data, t, soh_to_show)
    chrg_data = rpt_data.ica_dict[t][rpt_to_show][rpt_data.ica_dict[t][rpt_to_show]['curr'] > 0]
    dchg_data = rpt_data.ica_dict[t][rpt_to_show][rpt_data.ica_dict[t][rpt_to_show]['curr'] < 0]
    dva_cell_ax.plot((dchg_data['cap'].max() - dchg_data['cap']) / 1000,
                -dchg_data['dva_gauss'],
                label=f"{t.split('_')[0]}",
                linewidth=1.4,
                color=color_coding_kth(t))
dva_cell_ax.set_ylim(0, 0.75)
dva_cell_ax.set_xlabel('Capacity / Ah', fontdict=lbl_font)
dva_cell_ax.set_ylabel(r'DV, dV dQ$^{-1}$ / V Ah$^{-1}$', fontdict=lbl_font)
dva_cell_ax.text(1.85, 0.45, f"SOH {round(soh_to_show*100, -1):.0f}%", fontsize=20, bbox=box_props)
dva_cell_ax.legend(loc='upper center')
dva_cell_ax.grid(color='grey', alpha=0.5)
dva_cell_fig.savefig(os.path.join(output_dir, 'dva_soh90_multiple_cells_marked_updated_label.pdf'))
dva_cell_fig.savefig(os.path.join(output_dir, 'dva_soh90_multiple_cells_marked_updated_label.png'), dpi=300)
dva_cell_fig.savefig(os.path.join(output_dir, 'dva_soh90_multiple_cells_marked_updated_label.eps'), dpi=300)

hyst_cases = [
    '5 to 15 SOC cell 2',
    '85 to 95 SOC'
]
for h in hyst_cases:
    hyst_fig, hax1 = plt.subplots(1, 1)
    settings_515 = cell_info_dict[h]
    data_pts_5_15 = ['rpt_1', 'rpt_5', 'rpt_10']
    for i, k in enumerate(data_pts_5_15):
        tmp_data = rpt_data.ica_dict[settings_515[0]][k]
        x, y = calc_hysteresis(tmp_data)
        hax1.plot(x*100, y,
                  color=settings_515[1],
                  label=f'{settings_515[2]} FCE {look_up_fce_nrc(k)}',
                  linestyle=style_list[i - 1]
                  )
    hax1.set_xlabel('SOC / %', fontdict=lbl_font)
    hax1.set_ylabel('Voltage hysteresis / V', fontdict=lbl_font)
    hax1.lines[0].set_label('Fresh cell')
    hax1.lines[0].set_color('black')
    hax1.lines[0].set_linestyle('dashed')
    hax1.legend(loc='upper right', fontsize=19)
    hax1.set_ylim(0, 0.3)
    hax1.grid(alpha=0.5, color='grey')
    hyst_fig.savefig(os.path.join(output_dir, f'hysteresis_plot_{h.replace(" ", "_")}_FCE_number_alt.pdf'))
    hyst_fig.savefig(os.path.join(output_dir, f'hysteresis_plot_{h.replace(" ", "_")}_FCE_number_alt.png'), dpi=300)
    hyst_fig.savefig(os.path.join(output_dir, f'hysteresis_plot_{h.replace(" ", "_")}_FCE_number_alt.eps'), dpi=300)

bol_hyst_fig, bhax1 = plt.subplots(1, 1)
soc = (bol_ref.mAh - bol_ref.mAh.min()) / (bol_ref.mAh.max() - bol_ref.mAh.min())
bol_ref.loc[:, 'soc'] = soc
x, y = calc_hysteresis(bol_ref)
ln1 = bhax1.plot(x * 100, y, color='black', label='Voltage hystersis')
bhax2 = bhax1.twinx()
bhax2.grid(False)
ln2 = bhax2.plot(bol_ref.soc * 100, bol_ref.volt, color='red', label='Cell voltage')
bhax1.set_xlabel('SOC / %', fontdict=lbl_font)
bhax1.set_ylabel('Voltage hysteresis / V', fontdict=lbl_font)
bhax2.set_ylabel('Cell voltage / V', fontdict=lbl_font)
lns = ln1+ln2
labs = [l.get_label() for l in lns]
bhax1.legend(lns, labs, loc='center right')
bol_hyst_fig.savefig(os.path.join(output_dir, f'bol_hysteresis_plot_w_voltage_updated_label.pdf'))
bol_hyst_fig.savefig(os.path.join(output_dir, f'bol_hysteresis_plot_w_voltage_updated_label.png'), dpi=300)
bol_hyst_fig.savefig(os.path.join(output_dir, f'bol_hysteresis_plot_w_voltage_updated_label.eps'), dpi=300)

# Assemble hysteresis plots in one figure
fig6, ax6 = plt.subplots(3, 1, figsize=(x_width, 9 / 4 * x_width))
ln1 = ax6[0].plot(x * 100, y, color='black', label='Voltage hystersis')
ax6_2 = ax6[0].twinx()
ax6_2.grid(False)
ln2 = ax6_2.plot(bol_ref.soc * 100, bol_ref.volt, color='red', label='Cell voltage')
ax6[0].set_xlabel('SOC / %', fontdict=lbl_font, labelpad=2)
ax6[0].set_ylabel('Voltage hysteresis / V', fontdict=lbl_font, labelpad=2)
ax6_2.set_ylabel('Cell voltage / V', fontdict=lbl_font, labelpad=1)
lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax6[0].legend(lns, labs, loc='center right')

for n, h_case in enumerate(hyst_cases):
    settings = cell_info_dict[h_case]
    data_pts = ['rpt_1', 'rpt_4', 'rpt_8']
    for i, k in enumerate(data_pts):
        tmp_data = rpt_data.ica_dict[settings[0]][k]
        x, y = calc_hysteresis(tmp_data)
        ax6[n + 1].plot(x*100, y,
                        color=settings[1],
                        label=f'{settings[2]} FCE {look_up_fce_nrc(k)}',
                        dashes=dash_list[i - 1]
                        )
    ax6[n + 1].set_xlabel('SOC / %', fontdict=lbl_font, labelpad=2)
    ax6[n + 1].set_ylabel('Voltage hysteresis / V', fontdict=lbl_font, labelpad=2)
    #hax1.lines[0].set_label('Fresh cell')
    ax6[n + 1].lines[0].set_color('black')
    ax6[n + 1].lines[0].set_linestyle('dashed')
    ax6[n + 1].legend(loc='upper right', fontsize=9)
    ax6[n + 1].set_ylim(0, 0.3)
fig6.subplots_adjust(bottom=0.05)
fig6.subplots_adjust(top=0.96)
fig6.subplots_adjust(left=0.16)
fig6.subplots_adjust(hspace=0.25)
fig6.subplots_adjust(right=0.87)
fig6_labels = ['(a)', '(b)', '(c)']
for k, label in enumerate(fig6_labels):
    ax6[k].text(-0.22, 1.0, label, transform=ax6[k].transAxes, fontsize=9)
fig6.savefig(os.path.join(output_dir, 'figure6_hysteresis.pdf'))
fig6.savefig(os.path.join(output_dir, 'figure6_hysteresis.png'), dpi=300)
fig6.savefig(os.path.join(output_dir, 'figure6_hysteresis.eps'), dpi=300)


# Plot all DVAs from a case with gliding color scheme. Currently not used
cmap_fig, cax = plt.subplots(1, 1)
my_case = '5 to 15 SOC_2_1'
n = len(rpt_data.ica_dict[my_case])
col_map_colors = plt.cm.viridis((np.linspace(0, 1, n)))
offset_array = np.linspace(0, 0.25, n)
for i, k in enumerate(rpt_data.ica_dict[my_case]):
    tmp_df = rpt_data.ica_dict[my_case][k]
    cax.plot((tmp_df[tmp_df.curr < 0]['cap'].max() - tmp_df[tmp_df.curr < 0]['cap']) / 1000,
             -tmp_df[tmp_df.curr < 0]['dva_gauss'],
             color=col_map_colors[i])
cax.set_ylim(0, 0.75)
cax.set_xlabel('Capacity / Ah', fontdict=lbl_font)
cax.set_ylabel(r'DV, dV dQ$^{-1}$ / V Ah$^{-1}$', fontdict=lbl_font)
#cax.legend(loc='upper center')
cax.grid(color='grey', alpha=0.5)
cmap_fig.savefig(os.path.join(output_dir, '5_15_all_dva_color_map.png'), dpi=300)

# Plot all ICAs from dual cases with gliding color scheme. Currently not used
cmap_ica_fig, ciax = plt.subplots(2, 1, sharex='all', sharey='all', figsize=(x_width, 2*aspect_rat * x_width))
my_cases = [
    '5 to 15 SOC_2_2',
    '85 to 95 SOC_4_1'
]
for j, my_case in enumerate(my_cases):
    n = len(rpt_data.ica_dict[my_case])
    col_map_colors = plt.cm.jet((np.linspace(0, 1, n)))
    offset_array = np.linspace(0, 0.25, n)
    for i, k in enumerate(sorted(rpt_data.ica_dict[my_case], reverse=True)):
        tmp_df = rpt_data.ica_dict[my_case][k]
        if tmp_df[tmp_df.curr > 0]['ica_gauss'].abs().max() > 5:
            ciax[j].plot(tmp_df[tmp_df.curr > 0]['volt'],
                     tmp_df[tmp_df.curr > 0]['ica_gauss'],
                     color=col_map_colors[i])
        if tmp_df[tmp_df.curr < 0]['ica_gauss'].abs().max() > 5:
            ciax[j].plot(tmp_df[tmp_df.curr < 0]['volt'],
                      tmp_df[tmp_df.curr < 0]['ica_gauss'],
                      color=col_map_colors[i])
        ciax[j].set_xlim(2.8, 4.2)
        ciax[j].set_xlabel('Voltage / V', fontdict=lbl_font)
        ciax[0].set_ylabel(r'IC, dQ dV$^{-1}$ / Ah V$^{-1}$', fontdict=lbl_font)
        ciax[j].grid(color='grey', alpha=0.5)
        ciax[j].set_title(my_case)
cmap_fig_labels = ['(a)', '(b)']
for k, label in enumerate(cmap_fig_labels):
    ciax[k].text(-0.12, 1.0, label, transform=ciax[k].transAxes, fontsize=16)
cmap_ica_fig.savefig(os.path.join(output_dir, 'ica_all_tests_color_map.png'), dpi=300)
