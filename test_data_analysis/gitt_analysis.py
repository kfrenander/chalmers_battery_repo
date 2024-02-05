import pandas as pd

from test_data_analysis.read_neware_file import read_neware_xls
import numpy as np
import matplotlib.pyplot as plt
from test_data_analysis.tesla_half_cell import data_reader
from scipy.optimize import minimize, brute
from scipy.interpolate import interp1d
import scipy.constants
plt.style.use('chalmers_kf')


def find_ocv(df):
    chrg = df[df['curr'] > 1e-4]
    dchg = df[df['curr'] < -1e-4]
    dchg.loc[:, 'cap'] = dchg['cap'] - dchg['cap'].min()
    chrg.loc[:, 'cap'] = chrg['cap'] - chrg['cap'].min()
    soc_dchg = 1 - dchg['cap'] / dchg['cap'].max()
    soc_chrg = 1 - chrg['cap'] / chrg['cap'].max()
    dchg_volt = interp1d(soc_dchg, dchg['pot'])
    chrg_volt = interp1d(soc_chrg, chrg['pot'])
    soc_pts = np.linspace(0, 1, 200)
    return np.array([soc_pts, np.mean([chrg_volt(soc_pts), dchg_volt(soc_pts)], axis=0)])


def calc_dv_dsqrt(df):
    rel_df = df[(df.float_step_time > 10) & (df.float_step_time < 40)]
    ft = np.polyfit(np.sqrt(rel_df.float_step_time), rel_df.volt, deg=1)
    return ft


def plot_linear_fit(df):
    ft = calc_dv_dsqrt(df)
    plt.figure()
    sqrt_t = np.sqrt(df.float_step_time)
    plt.plot(sqrt_t, df.volt, '.-')
    x_fit = np.linspace(0, 10)
    plt.plot(x_fit, np.poly1d(ft)(x_fit), '--')
    return None


def mult_fit_by_split(split_idx, data):
    split_idx = int(split_idx)
    df1 = data.iloc[:split_idx, :]
    df2 = data.iloc[split_idx:, :]
    ft1, res1, _, _, _ = np.polyfit(np.sqrt(df1.float_step_time),
                           df1.volt, deg=1, full=True)
    ft2, res2, _, _, _ = np.polyfit(np.sqrt(df2.float_step_time),
                           df2.volt, deg=1, full=True)
    return res1 + res2


def generate_ocv(ocv_file):
    ocv_df = data_reader(ocv_file)
    ocv_df = ocv_df[ocv_df['step'].isin(range(3, 6))]
    ocv_arr = find_ocv(ocv_df)
    ocv_lookup = interp1d(ocv_arr[0, :], ocv_arr[1, :], fill_value="extrapolate")
    soc_lookup = interp1d(ocv_arr[1, :], ocv_arr[0, :], fill_value="extrapolate")
    ocv_grad = np.gradient(ocv_arr[1, :], ocv_arr[0, :])
    ocv_grad_lookup = interp1d(ocv_arr[0, :], ocv_grad, fill_value="extrapolate")
    return ocv_lookup, soc_lookup, ocv_grad_lookup


cell_1_3 = r"C:\Users\krifren\TestData\HalfCellData\GITT_CellBuildTesla_21w06\240093-1-3-2818574098.xls"
cell_1_4 = r"C:\Users\krifren\TestData\HalfCellData\GITT_CellBuildTesla_21w06\240093-1-4-2818574097.xls"
data_set = {
    'cell_1_4': read_neware_xls(cell_1_4),
    'cell_1_3': read_neware_xls(cell_1_3)
}
ocv_file = r"\\sol.ita.chalmers.se\groups\batt_lab_data\HalfCellData\HalfCellData_KTH\20200910-AJS-PH0S06-Tes-C10-BB5.txt"
ocv_lookup, soc_lookup, ocv_grad_lookup = generate_ocv(ocv_file)

df = data_set['cell_1_4']
i_chrg = df[df['mode'].str.contains('CC_Chg')]['curr'].mean()
i_dchg = df[df['mode'].str.contains('CC_DChg')]['curr'].mean()
rest_steps = df[df['mode'] == 'Rest'].arb_step2.unique()
dcg_step = df[df['mode'] == 'CC_DChg'].arb_step2.unique()

df_dict = {'step_{}'.format(k): df[df['arb_step2'] == k] for k in rest_steps}
F = scipy.constants.value('Faraday constant')
ex_df = df_dict['step_13']
diff = {}
for key in df_dict:
    ex_df = df_dict[key]
    try:
        ex_soc = soc_lookup(ex_df['volt'].iloc[-1])
        dUdd = ocv_grad_lookup(ex_soc)
        dvdsqrt = calc_dv_dsqrt(ex_df)[0]
        if dvdsqrt < 0:
            pulse_mode = 'chrg'
        else:
            pulse_mode = 'dchg'
        molar_mass = 95.88e-3  # kg/mol
        dens = 2100  #
        molar_vol = molar_mass / dens
        surf_area = 3 * 0.5 / 5e-6 * (7.5e-3**2 * np.pi * 70e-6)
        D = 4 / np.pi * (i_dchg * molar_vol / (surf_area * F))**2 * (dUdd / dvdsqrt)**2
        diff[ex_soc[()]] = [D, pulse_mode]
    except ValueError:
        print(f'Failed for {key} and {ex_df["volt"].iloc[-1]}')
    except TypeError:
        print(f'Failed for {key} with {ex_soc:.2f}')

diffusion_df = pd.DataFrame.from_dict(diff, orient='index', columns=['D', 'pulse_mode'])

print('Finished successfully')

#
# overview_fig = plt.figure()
# cc_cv_idx = {k: data_set[k][data_set[k]['mode'] == 'CCCV_Chg'].last_valid_index() for k in data_set}
# for key in cc_cv_idx:
#     corr_time = data_set[key].loc[cc_cv_idx[key], 'float_time']
#     plt.plot((data_set[key].float_time[cc_cv_idx[key]:] - corr_time) / 3600,
#              data_set[key].volt[cc_cv_idx[key]:],
#              linewidth=0.7, alpha=0.7, label=key)
# plt.legend()
# plt.xlabel('Time [h]')
# plt.ylabel('Voltage - positive half-cell')
# plt.xlim([-1, 114])
# plt.savefig(r'C:\Users\krifren\TestData\analysis_directory\half_cell_analysis\tesla_pos_electrodes.svg', dpi=600)
#
# raw_data_fig, ax = plt.subplots(2, 2, sharey=True)
# upper_stp = 55
# lower_stp = upper_stp - 10
# filt_time = 0.8
# for key in df_dict:
#     stp_df = df_dict[key]
#     if stp_df.arb_step2.iloc[0] > lower_stp and stp_df.arb_step2.iloc[0] < upper_stp:
#         stp_df.reset_index(inplace=True, drop=True)
#         start_of_diff = stp_df[stp_df.float_step_time > filt_time].first_valid_index()
#         end_of_diff = stp_df[stp_df.float_step_time < 450].last_valid_index()
#         sub_df = stp_df.loc[start_of_diff:end_of_diff, :]
#         sub_df.reset_index(inplace=True, drop=True)
#         # plt.scatter(np.sqrt(sub_df.float_step_time), sub_df.volt)
#         ax[0, 0].plot(sub_df.float_step_time, sub_df.volt, '-*', label='step time')
#         ax[0, 0].set_xlabel('Time [s]')
#         ax[0, 0].set_ylabel('Potential Li+/Li')
#         # ax[0].plot(sub_df.float_time - sub_df.float_time.iloc[0], sub_df.volt, label='total time')
#         ax[1, 0].plot(np.sqrt(sub_df.float_step_time), sub_df.volt, '+')
#         ax[0, 1].plot(np.sqrt(sub_df.float_step_time), sub_df.volt, '+')
#         ax[1, 1].plot(np.sqrt(sub_df.float_step_time), sub_df.volt, '+')
#         start_of_fit = sub_df[sub_df.float_step_time > 4].first_valid_index()
#         print(start_of_fit)
#         ft_tr = np.polyfit(np.sqrt(sub_df.float_step_time.iloc[:start_of_fit]),
#                            sub_df.volt.iloc[:start_of_fit], deg=1)
#         ft, res, _, _, _ = np.polyfit(np.sqrt(sub_df.float_step_time.iloc[start_of_fit:]),
#                                       sub_df.volt.iloc[start_of_fit:], deg=1, full=True)
#         ax[1, 0].plot(np.sqrt(sub_df.float_step_time), np.poly1d(ft)(np.sqrt(sub_df.float_step_time)),
#                       linewidth=0.8, label='Linear fit later region v sqrt(t)')
#         ax[1, 0].set_xlabel('Square root of time')
#         ax[1, 0].set_ylabel('Potential Li+/Li')
#         ax[0, 1].plot(np.sqrt(sub_df.float_step_time), np.poly1d(ft_tr)(np.sqrt(sub_df.float_step_time)),
#                       linewidth=0.8, label='Linear fit transition region v sqrt(t)')
#         ax[0, 1].set_xlabel('Square root of time')
#         ax[0, 1].set_ylim(sub_df.volt.min() - 0.04, sub_df.volt.max() + 0.1)
#         ax[0, 1].set_xlim(filt_time - 0.1, np.sqrt(sub_df.loc[start_of_fit, 'float_step_time']) + 0.5)
#         ax[1, 1].plot(np.sqrt(sub_df.float_step_time), np.poly1d(ft)(np.sqrt(sub_df.float_step_time)),
#                       linewidth=0.8, label='Linear fit later region v sqrt(t)')
#         ax[1, 1].set_xlabel('Square root of time')
#         ax[1, 1].plot(np.sqrt(sub_df.float_step_time), np.poly1d(ft_tr)(np.sqrt(sub_df.float_step_time)),
#                       linewidth=0.8, label='Linear fit transition region v sqrt(t)')
#         # raw_data_fig.text(0.04, 0.5, 'Voltage [V]', va='center', rotation='vertical')
# plt.grid(True)
# plt.legend()

