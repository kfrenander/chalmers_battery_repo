import pandas as pd
import numpy as np
import os
import re
from misc_classes.test_metadata_reader import MetadataReader
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from check_current_os import get_base_path_batt_lab_data
from test_data_analysis.rpt_analysis import characterise_steps
import natsort
from numba import njit
from scipy.signal.windows import gaussian
plt.style.use(['widthsixinches', 'kelly_colors'])
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "text.latex.preamble": r'\usepackage{siunitx}'
})



def check_ici_step(df, step):
    stp_df = df[df.step == step]
    mean_curr = stp_df.curr.mean()
    first_curr = stp_df.loc[stp_df.first_valid_index(), 'curr']
    last_curr = stp_df.loc[stp_df.last_valid_index(), 'curr']
    duration = stp_df.float_time.max() - stp_df.float_time.min()
    if abs((first_curr + last_curr)/2 - mean_curr) < 1e-2 and abs(mean_curr) > 0 and abs(duration - 300) < 10:
        return True
    else:
        return False


def calc_step_res(df):
    dchg_curr = 1
    for st in df.step.unique():
        if st > 1:
            beg_ind = df[df.step == st].first_valid_index()
            stp_ind = df[df.step == st].last_valid_index()
            stp_curr = df[df.step == st]['curr'].mean()
            if stp_curr != 0:
                dchg_curr = stp_curr
                fin_volt = df.loc[stp_ind, 'volt']
            elif stp_curr == 0 and check_ici_step(df, int(st - 1)):
                df.loc[beg_ind, 'R0'] = 1e3 * (fin_volt - df.loc[beg_ind, 'volt']) / dchg_curr
                df.loc[stp_ind, 'R10'] = 1e3 * (fin_volt - df.loc[stp_ind, 'volt']) / dchg_curr
    return df


def find_ici_parameters(df, ch_df):
    df['stp_diff'] = df['arb_step2'].diff().fillna(0)
    for k, row in ch_df.iterrows():
        if row['current_mode'] == 'interrupt':
            stp = row['step_nbr']
            idf = df[df['arb_step2'] == stp]
            curr = df[df['arb_step2'] == stp - 1]['curr'].mean()
            pol_volt = df[df['arb_step2'] == stp - 1]['volt'].iloc[-1]
            beg_volt = idf['volt'].iloc[0]
            fin_volt = idf['volt'].iloc[-1]
            R0 = (pol_volt - beg_volt) / curr
            R10 = (pol_volt - fin_volt) / curr
            ch_df.loc[k, 'R0_mohm'] = R0 * 1e3
            ch_df.loc[k, 'R10_mohm'] = R10 * 1e3
            slope, v_intcpt = fit_lin_vs_sqrt_time(idf, 'volt')
            ch_df.loc[k, 'k'] = - slope / curr
            ch_df.loc[k, 'R_reg_mohm'] = 1e3 * (pol_volt - v_intcpt) / curr
        else:
            stp = row['step_nbr']
            cdf = df[df['arb_step2'] == stp]
            fit_coeffs, residual = fit_lin_ocp_slope(cdf, 'volt')
            if residual[0] < 1e-5:
                ch_df.loc[k, 'dOCPdT'] = fit_coeffs[0]
    return ch_df


def categorize_step(ch_df):
    for k, row in ch_df.iterrows():
        if row['step_mode'] == 'Rest':
            if ch_df.loc[k - 1, 'step_mode'] == 'CC Chg':
                ch_df.loc[k, 'ici_mode'] = 'chrg'
            elif ch_df.loc[k - 1, 'step_mode'] == 'CC DChg':
                ch_df.loc[k, 'ici_mode'] = 'dchg'
            ch_df.loc[k, 'current_mode'] = 'interrupt'
        else:
            ch_df.loc[k, 'current_mode'] = 'current'
    return ch_df


def fit_lin_vs_sqrt_time(df, col):
    fit_df = df[(df['float_step_time'] > 1)] # & (df['step_time_float'] <= 5)]
    coeffs = np.polyfit(np.sqrt(fit_df['float_step_time']), fit_df[col], 1)
    return coeffs


def fit_lin_ocp_slope(df, col):
    fit_df = df[df['float_step_time'] > df['float_step_time'].max() - 120]
    fit_tuple = np.polyfit(fit_df['float_step_time'], fit_df[col], 1, full=True)
    coeffs = fit_tuple[0]
    residual = fit_tuple[1]
    return coeffs, residual


@njit
def filter_indices(voltages, indices, mode, v_flt):
    idx_to_remove = []
    volt_mark = 0 if mode == 'chrg' else 5

    for i in range(len(voltages)):
        if (mode == 'chrg' and voltages[i] > volt_mark + v_flt) or \
                (mode == 'dchg' and voltages[i] < volt_mark - v_flt):
            volt_mark = voltages[i]
        else:
            idx_to_remove.append(indices[i])  # Append the actual index value

    return np.array(idx_to_remove)


@njit
def filter_indices_positive(voltages, indices, mode, v_flt):
    idx_to_retain = []
    volt_mark = 0 if mode == 'chrg' else 5

    for i in range(len(voltages)):
        if (mode == 'chrg' and voltages[i] > volt_mark + v_flt) or \
                (mode == 'dchg' and voltages[i] < volt_mark - v_flt):
            volt_mark = voltages[i]
            idx_to_retain.append(indices[i])

    return np.array(idx_to_retain)


def extract_ica_from_ici(df, col):
    v_flt = 5e-3
    modes = {'chrg': {'first_index': df[df['mode'] == 'CC Chg'].first_valid_index(),
                      'volt_mark': 0,
                      'mAh_factor': 1},
             'dchg': {'first_index': df[df['mode'] == 'CC DChg'].first_valid_index(),
                      'volt_mark': 5,
                      'mAh_factor': -1}}

    chrg_first_idx = modes['chrg']['first_index']
    dchg_first_idx = modes['dchg']['first_index']

    if chrg_first_idx < dchg_first_idx:
        df_chrg = df.loc[:dchg_first_idx - 1]
        df_dchg = df.loc[dchg_first_idx:]
    else:
        df_chrg = df.loc[chrg_first_idx:]
        df_dchg = df.loc[:chrg_first_idx - 1]

    # Process charge and discharge dataframes
    dfs = {}
    for mode in ['chrg', 'dchg']:
        voltages = (df_chrg if mode == 'chrg' else df_dchg)[col].values
        indices = (df_chrg if mode == 'chrg' else df_dchg).index.values
        idx_to_keep = filter_indices_positive(voltages, indices, mode, v_flt)

        ica_df = df.loc[idx_to_keep, :].copy()

        # Compute the gradient for 'ica_raw' and apply Gaussian filtering
        ica_df['ica_raw'] = np.gradient(modes[mode]['mAh_factor'] * ica_df['mAh'], ica_df['volt'])
        ica_df['ica_gauss'] = gaussian_filter1d(ica_df['ica_raw'], sigma=1.3, mode='nearest')
        # ica_df['ica_gauss'] = gaussian_filtering(ica_df, 'ica_raw')

        dfs[mode] = ica_df

    # Merge the charge and discharge dataframes
    ica_df = pd.concat(dfs.values(), axis=0, ignore_index=True)
    return ica_df


def gaussian_filtering(df, col):
    N = np.floor(0.033 * len(df))
    N = N - 1 + N%2
    gauss_win = gaussian(N, 14/5)
    flt_data = np.convolve(df[col], gauss_win, mode='same') / sum(gauss_win)
    return flt_data


def perform_ici_analysis(pkl_file):
    ici_df = pd.read_pickle(pkl_file)
    ica_df = extract_ica_from_ici(ici_df, col='volt')
    proc_df = characterise_steps(ici_df)
    proc_df = categorize_step(proc_df)
    proc_df = find_ici_parameters(ici_df, proc_df)
    return proc_df, ica_df


def run_ici_analysis_on_path(base_path):
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.pkl') and 'ici_dump' in file:
                print(f'Processing {file} from {os.path.split(root)[-1]}')
                tmp_df, ica_df = perform_ici_analysis(os.path.join(root, file))
                output_file = os.path.join(root, file.replace('dump', 'processed'))
                output_ica = os.path.join(root, file.replace('ici', 'ica'))
                # tmp_df.to_pickle(output_file)
                ica_df.to_pickle(output_ica)
    return None


def sample_tests(base_folder, subfolder_filters, rpt_filters):
    """
    Samples test files from the base folder based on specified subfolder and rpt filters.

    Parameters:
        base_folder (str): The path to the base folder containing subfolders with data files.
        subfolder_filters (list of tuples): List of (y, z) tuples to filter subfolders.
        rpt_filters (list of int): List of rpt numbers to include.

    Returns:
        list of tuples: Each tuple contains (ici_id, ch_df) for each filtered test case.
    """
    # Compile regex patterns for filtering subfolders and rpt files
    subfolder_patterns = [f"pickle.*channel_.*_{y}_{z}_" for y, z in subfolder_filters]
    rpt_patterns = [f"2.*ici_processed_rpt_{rpt}.pkl" for rpt in rpt_filters]

    sampled_tests = []

    # Traverse through all directories and subdirectories
    for root, dirs, files in os.walk(base_folder):
        for d in dirs:
            # Check if the current directory matches any of the subfolder patterns
            if any(re.search(pattern, os.path.basename(d)) for pattern in subfolder_patterns):
                subfolder_dir = os.path.join(root, d)

                # Load metadata file
                meta_data = None
                for file in os.listdir(subfolder_dir):
                    if re.search('metadata.*', file):
                        meta_data = MetadataReader(file_path=os.path.join(root, d, file))

                if not meta_data:
                    continue  # Skip if metadata is not found

                # Filter and process rpt files
                for file in natsort.natsorted(os.listdir(subfolder_dir)):
                    if any(re.search(pattern, file) for pattern in rpt_patterns):
                        file_path = os.path.join(root, d, file)
                        # print(f"Processing file: {file_path}")
                        rpt_id = re.search(r'rpt_\d+', file).group().replace('_', ' ')
                        ici_id = f'{meta_data.test_condition} {rpt_id}'

                        # Load the data from the .pkl file
                        ch_df = pd.read_pickle(file_path)
                        sampled_tests.append((ici_id, ch_df))

    return sampled_tests


def plot_comparison(sampled_tests, plot_var='k', plot_mode='all', ax=None, fig=None):
    """
    Plots the comparison of the sampled test cases.

    Parameters:
        sampled_tests (list of tuples): List of (ici_id, ch_df) tuples for each test case.
        plot_var (str): The name of the variable to plot.
        plot_mode (str): The plotting mode.

    Returns:
        Figure, Axes: The figure and axes used for plotting.
    """
    # Initialize a new plot for comparison
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1)

    # Loop through the sampled tests and plot each one
    for ici_id, ch_df in sampled_tests:
        fig, ax = plot_resistance(ch_df, ax=ax, resistance=plot_var, ici_id=ici_id, plot_mode=plot_mode)

    return fig, ax


def compare_cases(base_folder, subfolder_filters, rpt_filters, plot_var='k', plot_mode='all', ax=None, fig=None):
    """
    Compares specific test cases from selected subfolders and files based on identifiers and rpt numbers.

    Parameters:
        base_folder (str): The path to the base folder containing subfolders with data files.
        subfolder_filters (list of tuples): List of (y, z) tuples to filter subfolders.
        rpt_filters (list of int): List of rpt numbers to include.
        plot_var (str): The name of the variable to plot
        plot_mode (str): The plotting mode.

    Returns:
        Figure, Axes: The figure and axes used for plotting.
    """
    # Sample tests based on filters
    sampled_tests = sample_tests(base_folder, subfolder_filters, rpt_filters)

    # Plot the comparison
    if ax:
        fig, ax = plot_comparison(sampled_tests, plot_var=plot_var, plot_mode=plot_mode, ax=ax)
    else:
        fig, ax = plot_comparison(sampled_tests, plot_var=plot_var, plot_mode=plot_mode)

    return fig, ax


def plot_resistance(ch_df, resistance='R0', fig=None, ax=None, ici_id='', plot_mode='all'):
    """
        Plots resistance values (R0 or R10) for charge and discharge data.

        Parameters:
            ch_df (DataFrame): The data to plot.
            resistance (str): The resistance type to plot ('R0' or 'R10').
            fig (Figure, optional): An existing figure to plot on.
            ax (Axes, optional): An existing axes to plot on.
            color_cycle (iterator, optional): An iterator for cycling through colors.

        Returns:
            Figure, Axes: The figure and axes used for plotting.
        """
    if resistance not in ['R0', 'R10', 'R_reg', 'k']:
        raise ValueError("Invalid resistance type. Choose 'R0' or 'R10'.")

    # Create new figure/axis if not provided
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1)

    color_cycle = ax._get_lines.prop_cycler

    # Get next colors for each mode
    color_chrg = next(color_cycle)['color']
    color_dchg = next(color_cycle)['color']
    if 'R' in resistance:
        # Determine column name based on resistance type
        col_name = f"{resistance}_mohm"
    else:
        col_name = resistance

    # Plot the data
    if plot_mode == 'all':
        ch_df[ch_df['ici_mode'] == 'chrg'].plot.scatter(x='maxV', y=col_name, color=color_chrg, ax=ax, marker='.',
                                                        label=f'{resistance} Chrg ICI {ici_id}')
        ch_df[ch_df['ici_mode'] == 'dchg'].plot.scatter(x='maxV', y=col_name, color=color_dchg, ax=ax, marker='x',
                                                        label=f'{resistance} Dchg ICI {ici_id}')
    else:
        try:
            ch_df[ch_df['ici_mode'] == plot_mode].plot.scatter(x='maxV', y=col_name, color=color_chrg, ax=ax,
                                                               marker='x', label=f'{resistance} Chrg ICI {ici_id}')
        except Exception as e:
            print(f'Plot mode unknown, must be \'chrg\' or \'dchg\'. Exception {e}.')

    # Add legend and labels
    ax.legend()
    # ax.set_title(f'{resistance} vs maxV')
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel(fr'${{{resistance}}}$ [$\unit{{\milli\ohm}}$]')
    return fig, ax


def extract_averages(subfolder_filter_list, avg_param='k'):
    full_dict = {}
    for sub_filter in subfolder_filter_list:
        smpl_wc = sample_tests(DATA_BASE_PATH, subfolder_filters=sub_filter, rpt_filters=['.*'])
        mean_dct = {}
        for smpl in smpl_wc:
            nm, df = smpl
            nm_rpt = re.search(r'rpt \d+', nm).group()
            nm_cs = nm.split(' rpt')[0]
            mean_dct[nm_rpt] = df[(df.maxV > 3.4) & (df.maxV < 4.1)][avg_param].mean()
        dfm = pd.DataFrame.from_dict(mean_dct, orient='index', columns=[nm_cs])
        full_dict[nm_cs] = dfm
    return full_dict

# my_file = r"Z:\Provning\Neware\ICI_test_127.0.0.1_240119-2-8-100.xls"
# reader = NewareDataReader(my_file)
# df = reader.read_dynamic_data()
BASE_PATH = get_base_path_batt_lab_data()
DATA_BASE_PATH = os.path.join(BASE_PATH, 'pulse_chrg_test/cycling_data_repaired')

run_ici_analysis_on_path(DATA_BASE_PATH)
subfolder_filters = [('2', '1'), ('2', '3'), ('3', '8')]  # Example tuples to filter subfolders by y and z
rpt_filters = [1, 10]  # Example rpt numbers to include
smpl = sample_tests(DATA_BASE_PATH, subfolder_filters, rpt_filters)
k_fig, kax = compare_cases(DATA_BASE_PATH, subfolder_filters, rpt_filters, plot_var='k', plot_mode='chrg')
r_fig, rax = compare_cases(DATA_BASE_PATH, subfolder_filters, rpt_filters, plot_var='R_reg', plot_mode='chrg')
fig, ax = compare_cases(DATA_BASE_PATH, [(3, 7)], [1], plot_mode='dchg')
compare_cases(DATA_BASE_PATH, [(2, 3), (2,7 ), (3, 1), (3, 7)], [10], plot_mode='dchg', ax=ax)

# subfolder_filter_list = [[(2, 3)], [(2, 7)], [(3, 1)], [(3, 8)]]
subfolder_filter_log10 = [[(2, 3)], [(2, 7)], [(3, 1)], [(3, 8)]]
subfolder_filter_nopulse = [[(2, 1)], [(2, 3)], [(2, 5)], [(2, 7)]]
log10_dict_k = extract_averages(subfolder_filter_log10, avg_param='k')
nopulse_dict_k = extract_averages(subfolder_filter_nopulse, avg_param='k')
log10_dict_rreg = extract_averages(subfolder_filter_log10, avg_param='R_reg_mohm')

fig, ax = plt.subplots(1, 1)
for k, mean_df in log10_dict_k.items():
    mean_df.plot(ax=ax)
ax.set_ylabel(r'k [$\unit{{\milli\ohm\per\sqrt{\second}}}$]', fontsize=12)

fig, ax = plt.subplots(1, 1)
for k, mean_df in nopulse_dict_k.items():
    mean_df.plot(ax=ax)
ax.set_ylabel(r'k [$\unit{{\milli\ohm\per\sqrt{\second}}}$]', fontsize=12)

fig, ax = plt.subplots(1, 1)
for k, mean_df in log10_dict_rreg.items():
    mean_df.plot(ax=ax)
ax.set_ylabel(r'R\textsubscript{{reg}} [$\unit{{\milli\ohm}}$]', fontsize=12)


# PKL_FILE_1 = r"pulse_chrg_test\cycling_data_ici\pickle_files_channel_240095_3_8\240095_3_8_ici_dump_rpt_1.pkl"
# PKL_FILE_2 = r"pulse_chrg_test\cycling_data_ici\pickle_files_channel_240095_3_8\240095_3_8_ici_dump_rpt_2.pkl"
# PKL_FILE_3 = r"pulse_chrg_test\cycling_data_ici\pickle_files_channel_240095_3_8\240095_3_8_ici_dump_rpt_3.pkl"
# PKL_FILE_8 = r"pulse_chrg_test\cycling_data_ici\pickle_files_channel_240095_3_8\240095_3_8_ici_dump_rpt_8.pkl"
#
# rreg_fig, axr = plt.subplots(1, 1)
# r10_fig, ax10 = plt.subplots(1, 1)
# k_fig, kax = plt.subplots(1, 1)
# for pkl_file in [PKL_FILE_1, PKL_FILE_2, PKL_FILE_8]:
#     pkl_ici = os.path.join(BASE_PATH, pkl_file)
#     ici_id = re.search(r'rpt_\d', pkl_file).group()
#     df_pkl = pd.read_pickle(pkl_ici)
#     ch_df = characterise_steps(df_pkl)
#     ch_df = categorize_step(ch_df)
#     ch_df = find_ici_parameters(df_pkl, ch_df)
#     rreg_fig, axr = plot_resistance(ch_df, resistance='R_reg', ax=axr, ici_id=ici_id, plot_mode='dchg')
#     r10_fig, ax10 = plot_resistance(ch_df, resistance='R10', ax=ax10, ici_id=ici_id, plot_mode='dchg')
#     k_fig, kax = plot_resistance(ch_df, resistance='k', ax=kax, ici_id=ici_id, plot_mode='dchg')
# df = calc_step_res(df)

# fig, ax1 = plt.subplots(1, 1)
# ax1.plot(df.float_time, df.volt, label='voltage')
# ax2 = ax1.twinx()
# ax2.grid(False)
# # ax2.set_ylim([0, 2e-3])
# ax2.scatter(df.float_time, df.R0, marker='x', color='r', label='R0')
# ax2.scatter(df.float_time, df.R10, marker='p', color='k', label='R10')
#
# fig2, ax3 = plt.subplots(1, 1)
# xaxis = (df.mAh - df.mAh.min()) / (df.mAh.max() - df.mAh.min())
# ax3.plot(xaxis, df.volt, label='voltage')
# ax4 = ax3.twinx()
# ax4.grid(False)
# # ax2.set_ylim([0, 2e-3])
# ax4.scatter(xaxis, df.R0, marker='x', color='r', label='R0')
# ax4.scatter(xaxis, df.R10, marker='p', color='k', label='R10')
# ax4.legend()
#
#
# test_df = df[df.curr < 0]
# rem_df = test_df[test_df.volt.diff().abs() < 1e-3]
# test_ica = np.gradient(test_df.mAh/1000, test_df.volt)
# fig3, ax = plt.subplots(2, 1, sharex=True)
# ax[0].plot(rem_df.mAh/1000, rem_df.volt)
# ax[0].plot(test_df.mAh/1000, test_df.volt)
# # ax[0].plot(test_df.volt, test_ica)
# b, a = signal.butter(3, 0.01)
# volt_filt_butt = signal.filtfilt(b, a, rem_df.volt)
# ax[0].plot(rem_df.mAh/1000, volt_filt_butt)
# # ax[1].plot(df.float_time, volt_filt_savgol)
# filt_ica_butt = np.gradient(rem_df.mAh/1000, volt_filt_butt)
# ax[1].plot(rem_df.volt, filt_ica_butt)
# ax[1].set_ylim([0, 15])
