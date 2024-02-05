import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import curve_fit
import os
import re
import sys
from PythonScripts.test_data_analysis.tesla_half_cell import data_reader
from PythonScripts.test_data_analysis.ica_analysis import ica_on_arb_data
from PythonScripts.test_data_analysis.tesla_data_plot import look_up_fce_nrc
import natsort
import matplotlib as mpl
import datetime as dt
from matplotlib.patches import ConnectionPatch
from scipy.signal import find_peaks
plt.rcParams['axes.grid'] = True
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['legend.fontsize'] = 20


def color_coding_kth(test_name):
    list_of_tests = ['5 to 15 SOC',
                    '15 to 25 SOC',
                    '25 to 35 SOC',
                    '35 to 45 SOC',
                    '45 to 55 SOC',
                    '55 to 65 SOC',
                    '65 to 75 SOC',
                    '75 to 85 SOC',
                    'SOC_85_95'
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
    return col_dict[test_name]


def plot_dva(fc, ne, pe, title=''):
    dva_fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title(title)
    ax[0].plot(fc['Qs'], fc['Es'], label='Full cell')
    ax[0].plot(ne['Qs'], ne['Es'], label='Negative')
    ax[0].plot(pe['Qs'], pe['Es'], label='Positive')
    ax[0].legend()
    ax[0].set_ylabel('Voltage [V]')
    ax[0].set_xlabel('Capacity [mAh]')
    ax[1].plot(fc['Qs'], fc['dvdq'], label='Full cell')
    ax[1].plot(ne['Qs'], ne['dvdq'], label='Negative')
    ax[1].plot(pe['Qs'], pe['dvdq'], label='Positive')
    ax[1].set_ylim(0, 0.7*fc['dvdq'].mean())
    ax[1].legend()
    ax[1].set_ylabel('dV/dQ [mAh/V]')
    ax[1].set_xlabel('Capacity [mAh]')
    plt.tight_layout()
    return dva_fig


def find_neighbours(value, df, colname):
    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        return exactmatch.index
    else:
        lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
        upperneighbour_ind = df[df[colname] > value][colname].idxmin()
        return [lowerneighbour_ind, upperneighbour_ind]


def vis_coeffs(slip_df, cell_id):
    # slip_df.index = [float(re.search(r'\d+', x).group()) for x in slip_df.index]
    slip_df.sort_index(inplace=True)
    # slip_fit.columns = ['A_pe', 's_pe', 'A_ne', 's_ne', 'sol_eod']
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(slip_df.FCE, slip_df.A_pe, '.')
    ax[0, 0].set_title('Mass pe')
    ax[0, 1].plot(slip_df.FCE, slip_df.s_pe, '*')
    ax[0, 1].set_title('Slip pe')
    ax[1, 0].plot(slip_df.FCE, slip_df.A_ne, '.')
    ax[1, 0].set_title('Mass ne')
    ax[1, 1].plot(slip_df.FCE, slip_df.s_ne, '*')
    ax[1, 1].set_title('Slip ne')
    fig.suptitle('Slip for cell {}'.format(cell_id))
    fig.tight_layout()
    lli_fig, lli_ax = plt.subplots(1, 1)
    lli_ax.scatter(slip_df.FCE, slip_df.LLI, color='blue', label='LLI', marker='*')
    # lli_ax.scatter(slip_df.FCE, slip_df.LAM_ne, color='red', label='LAM NE', marker='.')
    lli_ax.set_xlabel('FCE')
    lli_ax.set_ylabel('LLI [Ah]')
    lli_ax.legend()
    return fig, lli_fig


def comp_interp(x, y):
    interp_lin = interp1d(x, y, kind='linear')
    interp_pchip = PchipInterpolator(x, y)
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    ax[0, 0].plot(x, y)
    ax[0, 0].set_title('Raw data')
    ax[1, 0].plot(x, interp_lin(x))
    ax[1, 0].set_title('Linear interp')
    ax[0, 1].plot(x, interp_pchip(x))
    ax[0, 1].set_title('Pchip interp')
    ax[1, 1].plot(x, y, label='raw')
    ax[1, 1].plot(x, interp_pchip(x), label='pchip')
    ax[1, 1].plot(x, interp_lin(x), label='lin')
    # ax[1, 1].legend(True)
    ax[1, 1].set_title('All variants')
    op_dict = {
        'lin': interp_lin,
        'pchip': interp_pchip
    }
    return op_dict


def reformat_soc(k):
    lvls = re.findall(r'\d+', k)
    soc = re.search(r'SOC', k).group()
    return '{} to {} {}'.format(*lvls, soc)


def reformat_test_name(k):
    lvls = re.findall(r'\d+', k)
    soc = re.search(r'soc', k).group()
    return f'{lvls[0]} to {lvls[1]} SOC - test {lvls[2][0]}_{lvls[2][1]}'


def calculate_lam_lli(coeff_dict):
    df = pd.DataFrame.from_dict(coeff_dict,
                                orient='index',
                                columns=['A_pe', 's_pe', 'A_ne', 's_ne', 'sol_eod', 'cap'])
    df.loc[:, 'FCE'] = [look_up_fce_nrc(idx) for idx in df.index]
    df.loc[:, 'LAM_ne'] = (df.loc[df.index[0], 'A_ne'] - df.loc[:, 'A_ne']) / df.loc[df.index[0], 'A_ne']
    df.loc[:, 'LAM_pe'] = (df.loc[df.index[0], 'A_pe'] - df.loc[:, 'A_pe']) / df.loc[df.index[0], 'A_pe']
    df.loc[:, 'LLI'] = df.loc[df.index[0], 'A_pe'] * df.loc[df.index[0], 'sol_eod'] - \
                       df.loc[:, 'A_pe']*df.loc[:, 'sol_eod']
    df.loc[:, 'LLI_init100'] = df.loc[df.index[0], 'A_pe'] * 1 - df.loc[:, 'A_pe']*df.loc[:, 'sol_eod']
    df.loc[:, 'LLI_norm'] = df.loc[:, 'LLI'] / (df.loc[df.index[0], 'A_pe'] * df.loc[df.index[0], 'sol_eod'])
    df.loc[:, 'LLI_init100_norm'] = (df.loc[:, 'LLI_init100'] /
                                     (df.loc[df.index[0], 'A_pe'] * df.loc[df.index[0], 'sol_eod']))
    return df.sort_values(by='FCE')



# Handle the input data
pe_file = r"Z:\Provning\Halvcellsdata\20200910-AJS-PH0S06-Tes-C10-BB5.txt"
ne_file = r"Z:\Provning\Halvcellsdata\20200910-AJS-NH0S05-Tes-C10-BB2.txt"
rpt_dir = r"\\sol.ita.chalmers.se\groups\batt_lab_data\Ica_files"
op_dir = r"\\sol.ita.chalmers.se\groups\batt_lab_data\analysis_directory"

# Read in files for each electrode test
df_pe = data_reader(pe_file)
df_ne = data_reader(ne_file)

# Find the appropriate step limits
min_idx = df_pe[df_pe['step'] == 9]['pot'].idxmin()
max_idx = df_pe[df_pe['step'] == 9]['pot'].idxmax()
part_df_pe = df_pe.iloc[max_idx:min_idx]
part_df_pe = part_df_pe[part_df_pe['curr'] < 0]

'''
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(part_df_pe['cap'], part_df_pe['pot'])
ax[1].plot(part_df_pe['cap'], part_df_pe['curr']*1e6)
ax[2].plot(part_df_pe['cap'], part_df_pe.dva)

fig2, ax2 = plt.subplots(2, 1, sharex=True)
ax2[0].plot(part_df_pe['step_time'], part_df_pe['pot'])
ax2[1].plot(part_df_pe['step_time'], part_df_pe['cap'])
'''
part_df_ne = df_ne[(df_ne['step'] == 21)]

'''
fig_ne, ax_ne = plt.subplots(2, 1, sharex=True)
ax_ne[0].plot(part_df_ne['time'], part_df_ne['pot'])
ax_ne[1].plot(part_df_ne['time'], part_df_ne['curr'])
'''
r_hc = 1.5 / 2
A_neg_test = np.pi * (r_hc) ** 2  # cm^2
A_pos_test = np.pi * (r_hc) ** 2  # cm^2

part_df_ne.loc[:, 'cap_cell'] = cumtrapz(part_df_ne['curr'], part_df_ne['time'], initial=0) / 3.6
# Negative sign in integral to compensate discharge current
part_df_pe.loc[:, 'cap_cell'] = cumtrapz(-part_df_pe['curr'], part_df_pe['time'], initial=0) / 3.6
part_df_ne.loc[:, 'sol'] = part_df_ne['cap'] / part_df_ne.cap.max()
part_df_pe.loc[:, 'sol'] = 1 - part_df_pe['cap'] / part_df_pe.cap.max()

x_pe = part_df_pe.loc[:, 'cap_cell'] / A_pos_test  # mAh / cm^2
y_pe = part_df_pe.loc[:, 'pot']
x_ne = part_df_ne.loc[:, 'cap_cell'] / A_neg_test  # mAh / cm^2
y_ne = part_df_ne.loc[:, 'pot']
# fit_fun_pe = lambda x, A_pe, s_pe: interp1d(A_pe * x_pe + s_pe, y_pe, fill_value=(y_pe.max(), y_pe.min()),
#                                             bounds_error=False)(x)
fit_fun_pe = lambda x, A_pe, s_pe: interp1d(A_pe * x_pe + s_pe, y_pe, fill_value='extrapolate')(x)
fit_fun_pe_pchip = lambda x, A_pe, s_pe: PchipInterpolator(A_pe * x_pe + s_pe, y_pe, extrapolate=True)(x)
# fit_fun_ne = lambda x, A_ne, s_ne: interp1d(A_ne * x_ne + s_ne, y_ne, fill_value=(y_ne.min(), y_ne.max()),
#                                             bounds_error=False)(x)
fit_fun_ne = lambda x, A_ne, s_ne: interp1d(A_ne * x_ne + s_ne, y_ne, fill_value='extrapolate')(x)
fit_fun_cell = lambda x, A_pe, s_pe, A_ne, s_ne: fit_fun_pe(x, A_pe, s_pe) - fit_fun_ne(x, A_ne, s_ne)


coeffs = {}
residuals = {}
p_init = [1.1, -0.003, 1.2, -0.01]
lo_b = (0.3, -1, 0.3, -1)
hi_b = (1.3, 1, 1.3, 1)

datestamp = dt.datetime.now().__format__('%y-%m-%d')
output = os.path.join(op_dir, 'figs_{}_init_{:.2f}_{:.0e}_{:.2f}_{:.0e}'.format(datestamp, *p_init))

file_dict = {}
for root, dirs, _ in os.walk(rpt_dir):
    for d in dirs:
        file_dict[d] = os.listdir(os.path.join(root, d))
fit_figs = {}
fit_params = {}
coeff_figs = {}
dva_figs = {}
for k in file_dict:
    # In this instance we want to filter out cases with partical soc windows.
    if 'soc' in k and not 'csv' in k:
        tmp_fig, ax_fit = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        cell_coeff = {}
        for file in natsort.natsorted(file_dict[k]):
            # print(os.path.join(rpt_dir, k, file))
            temp_file = os.path.join(rpt_dir, k, file)
            cyc_num = re.search(r'rpt_\d+', file).group()
            cell_num = re.search(r'\d_\d', file).group()
            temp_df = pd.read_pickle(temp_file)
            dchg_df = temp_df[temp_df.curr < 0]
            x_data = dchg_df['cap'] / 1000
            y_data = dchg_df['volt']
            n = y_data.shape[0]
            k1 = 4
            try:
                print('Capacity at {0} is {1:.3f}'.format(cyc_num, dchg_df['cap'].max() / 1000))
                try:
                    popt, pcov = curve_fit(fit_fun_cell, x_data, y_data, p0=p_init,
                                           bounds=(lo_b, hi_b))
                except RuntimeError:
                    print('Initial fit failed.')
                    p_init = [0.11, -500, 0.10, -200]
                    popt, pcov = curve_fit(fit_fun_cell, x_data, y_data, p0=p_init,
                                           bounds=(lo_b, hi_b))
                except:
                    print('Not working for for file {}'.format(file))
                # p_init = [*popt]
                pe_fit_x = np.linspace(popt[1], x_pe.max()*popt[0] + popt[1], 4000)
                u_pe = fit_fun_pe(pe_fit_x, *popt[:2])
                pe_res = ica_on_arb_data(x_data, fit_fun_pe(x_data, *popt[:2]))
                ne_fit_x = np.linspace(popt[3], x_ne.max()*popt[2] + popt[3], 4000)
                u_ne = fit_fun_ne(ne_fit_x, *popt[2:])
                ne_res = ica_on_arb_data(x_data, fit_fun_ne(x_data, *popt[2:]))
                sol_eod_pe = (x_data.max() - popt[1]) / (x_pe.max() * popt[0])
                cell_coeff['{}'.format(cyc_num)] = [*popt, sol_eod_pe, x_data.max()]
                cell_res = ica_on_arb_data(x_data, y_data)
                res = y_data - fit_fun_cell(x_data, *popt)
                ss_res = np.sum(res**2)
                ss_tot = np.sum((dchg_df['volt'] - np.mean(dchg_df['volt']))**2)
                R_sq = 1 - ss_res / ss_tot
                R_sq_adj = 1 - (1 - R_sq) * (n - 1) / (n - k1 - 1)
                print('The fit yields parameters {:.2f}, {:.2f}, {:.2f} and {:.2f} '
                      '\nwith r^2 {:.3f} and r^2_adj {:.3f}'.format(*popt, R_sq, R_sq_adj))
                residuals['{}_{}'.format(k, cyc_num)] = [ss_res, R_sq, R_sq_adj, pcov]
                # To not crowd plot, pick only subset of RPTs
                interesting_rpts = ['rpt_1.', 'rpt_12', 'rpt_18']  # 'rpt_7'
                if any(rpt in file for rpt in interesting_rpts):
                    ax_fit[0].plot(dchg_df['cap'] / 1000, dchg_df['volt'],
                                   label='Raw data at {}FCE'.format(look_up_fce_nrc(cyc_num)),
                                   color='red')
                    ax_fit[0].plot(dchg_df['cap'] / 1000, fit_fun_cell(dchg_df['cap'] / 1000, *popt),
                                   linestyle='dashed',
                                   label='Fitted data at {}FCE'.format(look_up_fce_nrc(cyc_num)),
                                   color='black',
                                   alpha=1,
                                   linewidth=1)
                    ax_fit[0].plot(ne_fit_x, u_ne, color='green')
                    ax_fit[0].plot(pe_fit_x, u_pe, color='black')
                    ax_fit[0].axvline(x_data.min(), color='blue', linestyle='dashed')
                    ax_fit[0].axvline(x_data.max(), color='blue', linestyle='dashed')
                    ax_fit[0].grid(alpha=0.5)
                    ax_fit[1].plot(x_data, res, label=f'Residual at {look_up_fce_nrc(cyc_num)}FCE')
                    ax_fit[1].grid(alpha=0.5)
                    r = re.search('rpt_\d+', file).group()
                    c = re.search('\d_\d', file).group()
                    dva_temp_fig = plot_dva(cell_res, ne_res, pe_res, title=f'{r} channel {c}')
                    dva_figs['{}_{}'.format(c, r)] = dva_temp_fig
            except ValueError:
                print('ICA empty, check {}'.format(file))
            except IndexError:
                print('ICA empty, check {}'.format(file))
        ax_fit[0].legend()
        ax_fit[0].set_xlabel('Discharge Capacity [Ah]')
        ax_fit[0].set_ylabel('Voltage [V]')
        ax_fit[0].set_title('Fits for cell {}'.format(k))
        fit_figs[k] = tmp_fig
        fit_params[k] = calculate_lam_lli(cell_coeff)
        coeff_figs[k] = vis_coeffs(fit_params[k], cell_id=k)
        coeffs[k] = cell_coeff

for key in fit_figs:

    if not os.path.exists(output):
        os.mkdir(output)
    fit_figs[key].savefig(os.path.join(output, 'fit_fig_{}.png'.format(key)), dpi=fit_figs[key].dpi)
    coeff_figs[key][1].savefig(os.path.join(output, 'lli_fig_{}.png'.format(key)), dpi=coeff_figs[key][1].dpi)
    coeff_figs[key][0].savefig(os.path.join(output, 'coeff_fig_{}.png'.format(key)), dpi=coeff_figs[key][0].dpi)
for key in dva_figs:
    dva_figs[key].savefig(os.path.join(output, 'dva_{}.png'.format(key)), dpi=dva_figs[key].dpi)
lli_comp_fig = plt.figure()
for key in fit_params:
    plt.scatter(fit_params[key]['FCE'], fit_params[key]['LLI'], label=key, marker='*')
plt.legend()
plt.xlabel('FCE')
plt.ylabel('LLI [Ah]')
print(f'Saving figures to figs_{datestamp}')
lli_comp_fig.savefig(os.path.join(output, 'LLI_comparison.png'), dpi=lli_comp_fig.dpi)
# plt.close('all')

for param in ['LAM_pe', 'LAM_ne', 'LLI_norm', 'LLI_init100_norm']:
    fig, ax = plt.subplots(1, 1)
    if 'lli' in param or 'lam' in param:
        scale_factor = 100
    else:
        scale_factor = 1
    for k in fit_params:
        df = fit_params[k].sort_values('FCE')
        ax.plot(df['FCE'][df['FCE'] < 1900],
                 df[param][df['FCE'] < 1900] * scale_factor,
                 marker='o',
                 markersize=9,
                 fillstyle='none',
                 label=reformat_test_name(k))
    if 'LLI_norm' in param:
        ax.set_ylabel('Loss of Lithium Inventory (%)', fontsize=24)
        save_name = 'lli_normalised.png'
    elif 'ne' in param:
        ax.set_ylabel('Negative Electrode Material Loss (%)', fontsize=24)
        save_name = 'lam_ne.png'
    elif 'pe' in param:
        plt.ylabel('Positive Electrode Material Loss (%)', fontsize=24)
        save_name = 'lam_pe.png'
    else:
        ax.set_ylabel(param, fontsize=24)
    ax.set_xlabel('Number of Full Cycle Equivalents (FCE)', fontsize=24)
    plt.xticks(np.arange(0, 1900, step=200))
    ax.set_xlim(-50, 1900)
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(output, save_name), dpi=400)

# m_fit_df = pd.read_csv(matlab_comp_data_file).dropna(how='all')
# m_fit_df.loc[:, 'rpt_name'] = [re.search(r'rpt_\d+', x).group() for x in m_fit_df.name]
# m_fit_df.loc[:, 'FCE'] = [look_up_fce_nrc(x) for x in m_fit_df.rpt_name]


########################################################################################################################
######################################## Visualise ocv and soc for teaching etc ########################################
########################################################################################################################
temp_file = os.path.join(rpt_dir, '1_2', '1_2_ica_dump_rpt_1.pkl')
temp_df = pd.read_pickle(temp_file)
dchg_df = temp_df[temp_df.curr < 0]
x_data = dchg_df['cap'] / 1000
y_data = dchg_df['volt']
popt, pcov = curve_fit(fit_fun_cell, x_data, y_data, p0=p_init, bounds=(lo_b, hi_b))
pe_fit_x = np.linspace(popt[1], x_pe.max()*popt[0] + popt[1], 4000)
u_pe = fit_fun_pe(pe_fit_x, *popt[:2])
ne_fit_x = np.linspace(popt[3], x_ne.max()*popt[2] + popt[3], 4000)
u_ne = fit_fun_ne(ne_fit_x, *popt[2:])

pe_ica_dct = ica_on_arb_data(x_data, fit_fun_pe(x_data, *popt[:2]))
ne_ica_dct = ica_on_arb_data(x_data, fit_fun_ne(x_data, *popt[2:]))

x_width = 7
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
ocv_fig, ax_ocv = plt.subplots(1, 1, figsize=(x_width, 12/16*x_width))
ax_ocv.plot(100*(dchg_df['cap'].max() - dchg_df['cap']) / dchg_df['cap'].max(), dchg_df['volt'],
            label='Full cell voltage',
            color='red',
            linewidth=2)
ax_ocv.plot(100*(dchg_df['cap'].max() - 1000*ne_fit_x) / dchg_df['cap'].max(), u_ne,
            color='forestgreen',
            label='Negative electrode voltage',
            linewidth=2)
ax_ocv.plot(100*(dchg_df['cap'].max() - 1000*pe_fit_x) / dchg_df['cap'].max(), u_pe,
            color='black',
            label='Positive electrode voltage',
            linewidth=2)
# ax_ocv.plot(100*dchg_df['cap'] / (dchg_df['cap'].max()), dchg_df['volt'],
#             label='Full cell voltage',
#             color='red',
#             linewidth=2)
# ax_ocv.plot(100*1000*ne_fit_x / (dchg_df['cap'].max()), u_ne,
#             color='forestgreen',
#             label='Negative electrode voltage',
#             linewidth=2)
# ax_ocv.plot(100*1000*pe_fit_x / (dchg_df['cap'].max()), u_pe,
#             color='black',
#             label='Positive electrode voltage',
#             linewidth=2)
# ax_ocv.set_xlabel('Full cell capacity / mAh', fontsize=16)
ax_ocv.set_xlabel('Full cell SOC / %', fontsize=16)
ax_ocv.set_ylabel('Voltage / V', fontsize=16)
ax_ocv.grid(False)
plt.vlines(0, ymin=0, ymax=3.55, linestyle='dashed', linewidth=2, color='deepskyblue')
plt.vlines(100, ymin=0, ymax=4.3, linestyle='dashed', linewidth=2, color='deepskyblue')
# plt.vlines(1, ymin=0, ymax=3.55, linestyle='dashed', linewidth=2)
ax_ocv.set_ylim(-0.05, 5.1)
ax_ocv.legend(loc='upper left', fontsize=12)
# ax_ocv.grid(alpha=0.5, color='grey')
ocv_fig.subplots_adjust(bottom=0.15)
ocv_fig.subplots_adjust(left=.1)
ocv_fig.savefig(os.path.join(r'Z:\Documents\Papers\LicentiateThesis\images', 'ocv_v_soc_corr.pdf'), dpi=300)
ocv_fig.savefig(os.path.join(r'Z:\Documents\Papers\LicentiateThesis\images', 'ocv_v_soc_corr.png'), dpi=300)

## PLOT PEDAGOGIC DISPLAY OF DVDQ AND DQDV
plt.style.use('kelly_colors')
boxprops=dict(facecolor='white', edgecolor='black',
              boxstyle='round')
col_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams['axes.labelsize'] = 16
ocv_dv_fig, dax = plt.subplots(2, 2,
                               sharex='col',
                               sharey='row',
                               gridspec_kw={'height_ratios': [2, 1],
                                            'width_ratios': [2, 1]})
dax[0, 0].plot(dchg_df['cap'], dchg_df['volt'],
               label='Full cell voltage',
               color=col_list[0],
               linewidth=2)
dax[1, 0].plot(dchg_df['cap'], -dchg_df['dva_filt'],
               label='',
               color=col_list[1],
               linewidth=2)
dax[0, 1].plot(-dchg_df['ica_filt'], dchg_df['volt'],
               color=col_list[2],
               linewidth=2)
dax[1, 1].set_axis_off()
dax[0, 0].set_ylabel('Voltage / V')
dax[1, 0].set_ylabel('DV / V mAh$^{-1}$')
dax[0, 1].set_xlabel('IC / mAh V$^{-1}$')
dax[1, 0].set_xlabel('Capacity / mAh')
dax[0, 0].set_xlim(-200, 4650)
dax[0, 0].set_ylim(2.45, 4.27)
dax[0, 0].set_yticks(np.arange(2.6, 4.25, 0.2))
dax[1, 0].set_ylim(0, 0.5)
dax[0, 1].set_xlim(0, 15)
dax[0, 1].text(5, 3, 'dQ/dV', bbox=boxprops, fontsize=16)
dax[1, 0].text(2000, 0.08, 'dV/dQ', bbox=boxprops, fontsize=16)

## ADD CONNECTION POINT BETWEEN GRADIENT AND DV / IC
con_point = dchg_df['dva_filt'].iloc[100:200].idxmin()
rng = np.arange(con_point-3, con_point+4, 1)
poly = np.polyfit(dchg_df.loc[rng, 'cap'], dchg_df.loc[rng, 'volt'], deg=1)
plt_rng = np.arange(con_point - 55, con_point + 56, 1)
gradient_line = dax[0, 0].plot(dchg_df.loc[plt_rng, 'cap'], np.polyval(poly, dchg_df.loc[plt_rng, 'cap']),
                               color='red', linewidth=2, linestyle='dashed')
xyv = (dchg_df.loc[con_point, 'cap'], dchg_df.loc[con_point, 'volt'])
xydq = (dchg_df.loc[con_point, 'cap'], -dchg_df.loc[con_point, 'dva_filt'])
xydv = (-dchg_df.loc[con_point, 'ica_filt'], dchg_df.loc[con_point, 'volt'])
con_dq = ConnectionPatch(xyA=xyv, coordsA=dax[0, 0].transData,
                         xyB=xydq, coordsB=dax[1, 0].transData,
                         color='lightcoral', linewidth=2, linestyle='dashed')
con_dv = ConnectionPatch(xyA=xyv, coordsA=dax[0, 0].transData,
                         xyB=xydv, coordsB=dax[0, 1].transData,
                         color='lightcoral', linewidth=2, linestyle='dashed')
ocv_dv_fig.add_artist(con_dq)
ocv_dv_fig.add_artist(con_dv)

# SAVE FIGURE
ocv_dv_fig.savefig(os.path.join(r'Z:\Documents\Papers\LicentiateThesis\images', 'ocv_dv_dq.pdf'), dpi=300)
ocv_dv_fig.savefig(os.path.join(r'Z:\Documents\Papers\LicentiateThesis\images', 'ocv_dv_dq.eps'), dpi=300)
ocv_dv_fig.savefig(os.path.join(r'Z:\Documents\Papers\LicentiateThesis\images', 'ocv_dv_dq.png'), dpi=300)

# CREATE MULTIPLE FIGURES FOR ANIMATION OF DQDV AND DVDQ
update_animation = 1
if update_animation:
    anim_folder = r"Z:\Documents\Papers\LicentiateThesis\images\dvdq_animation_figs_second"
    if not os.path.isdir(anim_folder):
        os.mkdir(anim_folder)
    for con_point in dchg_df['dva_filt'].iloc[70:-350].index.to_list()[0::4]:
        con_dq.remove()
        con_dv.remove()
        gradient_line[0].remove()
        rng = np.arange(con_point - 3, con_point + 4, 1)
        poly = np.polyfit(dchg_df.loc[rng, 'cap'], dchg_df.loc[rng, 'volt'], deg=1)
        plt_rng = np.arange(dchg_df.loc[con_point, 'cap'] - 600, dchg_df.loc[con_point, 'cap'] + 601, 1)
        gradient_line = dax[0, 0].plot(plt_rng, np.polyval(poly, plt_rng),
                                       color='red', linewidth=2, linestyle='dashed')
        xyv = (dchg_df.loc[con_point, 'cap'], dchg_df.loc[con_point, 'volt'])
        xydq = (dchg_df.loc[con_point, 'cap'], -dchg_df.loc[con_point, 'dva_filt'])
        xydv = (-dchg_df.loc[con_point, 'ica_filt'], dchg_df.loc[con_point, 'volt'])
        con_dq = ConnectionPatch(xyA=xyv, coordsA=dax[0, 0].transData,
                                 xyB=xydq, coordsB=dax[1, 0].transData,
                                 color='lightcoral', linewidth=2, linestyle='dashed')
        con_dv = ConnectionPatch(xyA=xyv, coordsA=dax[0, 0].transData,
                                 xyB=xydv, coordsB=dax[0, 1].transData,
                                 color='lightcoral', linewidth=2, linestyle='dashed')
        ocv_dv_fig.add_artist(con_dq)
        ocv_dv_fig.add_artist(con_dv)
        dax[0, 0].set_xlim(-200, 4650)
        dax[0, 0].set_ylim(2.45, 4.27)
        dax[0, 0].set_yticks(np.arange(2.6, 4.25, 0.2))
        ocv_dv_fig.savefig(os.path.join(anim_folder, f'dvdq_point{con_point}.png'), dpi=150)


## PEAK ASSIGNMENT METHOD ICA
plt.style.use('kelly_colors')
boxprops=dict(facecolor='white', edgecolor='black',
              boxstyle='round')
col_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams['axes.labelsize'] = 16
ocv_dv_fig, dax = plt.subplots(2, 2,
                               sharex='col',
                               sharey='row',
                               gridspec_kw={'height_ratios': [2, 1],
                                            'width_ratios': [2, 1]})
dax[0, 0].plot(dchg_df['cap'], dchg_df['volt'],
               label='Full cell voltage',
               color=col_list[0],
               linewidth=2)
dax[0, 0].plot(1000*ne_fit_x, u_ne + 2,
            color='forestgreen',
            label='Negative electrode voltage',
            linewidth=2)
dax[0, 0].plot(1000*pe_fit_x , u_pe,
            color='indianred',
            label='Positive electrode voltage',
            linewidth=2)
dax[1, 0].plot(dchg_df['cap'], -dchg_df['dva_filt'],
               label='',
               color=col_list[1],
               linewidth=2)
dax[0, 1].plot(-dchg_df['ica_filt'], dchg_df['volt'],
               color=col_list[2],
               linewidth=2)
dax[1, 1].set_axis_off()
dax[0, 0].set_ylabel('Voltage / V')
dax[1, 0].set_ylabel('DV / V mAh$^{-1}$')
dax[0, 1].set_xlabel('IC / mAh V$^{-1}$')
dax[1, 0].set_xlabel('Capacity / mAh')
dax[0, 0].set_xlim(-200, 4650)
dax[0, 0].set_ylim(2.0, 4.27)
dax[0, 0].set_yticks(np.arange(2., 4.25, 0.2))
dax[1, 0].set_ylim(0, 0.5)
dax[0, 1].set_xlim(0, 15)
dax[0, 1].text(5, 3, 'dQ/dV', bbox=boxprops, fontsize=16)
dax[1, 0].text(2000, 0.08, 'dV/dQ', bbox=boxprops, fontsize=16)

## ADD CONNECTION POINT BETWEEN GRADIENT AND DV / IC
voltage_list = [4.075, 3.93, 3.83, 3.65, 3.54, 3.47, 3.401, 3.138]
peak_idx = [find_neighbours(k, dchg_df, 'volt')[0] for k in voltage_list]
peak_folder = r"Z:\Documents\Papers\LicentiateThesis\images\dqdv_peak_assignment"
if not os.path.isdir(peak_folder):
    os.mkdir(peak_folder)
# peak_idx_raw, peak_props = find_peaks(-dchg_df['ica_filt'], distance=20, threshold=0.005)
# peak_idx = dchg_df.iloc[peak_idx_raw].index
for con_point, voltage in zip(peak_idx, voltage_list):
    rng = np.arange(con_point-3, con_point+4, 1)
    poly = np.polyfit(dchg_df.loc[rng, 'cap'], dchg_df.loc[rng, 'volt'], deg=1)
    plt_rng = np.arange(con_point - 35, con_point + 36, 1)
    gradient_line = dax[0, 0].plot(dchg_df.loc[plt_rng, 'cap'], np.polyval(poly, dchg_df.loc[plt_rng, 'cap']),
                                   color='red', linewidth=2, linestyle='dashed')
    xyv = (dchg_df.loc[con_point, 'cap'], dchg_df.loc[con_point, 'volt'])
    xydq = (dchg_df.loc[con_point, 'cap'], -dchg_df.loc[con_point, 'dva_filt'])
    xydv = (-dchg_df.loc[con_point, 'ica_filt'], dchg_df.loc[con_point, 'volt'])
    con_dq = ConnectionPatch(xyA=xyv, coordsA=dax[0, 0].transData,
                             xyB=xydq, coordsB=dax[1, 0].transData,
                             color='lightcoral', linewidth=2, linestyle='dashed')
    con_dv = ConnectionPatch(xyA=xyv, coordsA=dax[0, 0].transData,
                             xyB=xydv, coordsB=dax[0, 1].transData,
                             color='lightcoral', linewidth=2, linestyle='dashed')
    ocv_dv_fig.add_artist(con_dq)
    ocv_dv_fig.add_artist(con_dv)
    ocv_dv_fig.savefig(os.path.join(peak_folder, f'peak_volt_w_gradient_{voltage:.2f}.png'), dpi=200)
    con_dq.remove()
    con_dv.remove()
    gradient_line[0].remove()


