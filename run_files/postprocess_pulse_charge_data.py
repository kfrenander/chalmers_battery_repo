import pickle
import numpy as np
from test_data_analysis.pec_smartcell_data_handler import PecSmartCellDataHandler
from check_current_os import get_base_path_batt_lab_data
import os
import re
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from itertools import product, permutations
import scienceplots
from datetime import datetime
plt.style.use(['science', 'nature_w_si_harding', 'grid'])


def load_or_generate_data(data_from_scratch, data_dir, handler_output_dir):
    if data_from_scratch:
        print("Generating data from scratch...")
        handler = PecSmartCellDataHandler(data_dir)
    else:
        latest_file = None
        latest_date = None

        # Iterate through files in the output directory
        for file in os.listdir(handler_output_dir):
            match = re.search(r'full_post_process_data_dump_(\d{4}-\d{2}-\d{2})\.pkl', file)
            if match:
                file_date = datetime.strptime(match.group(1), '%Y-%m-%d')
                if latest_date is None or file_date > latest_date:
                    latest_date = file_date
                    latest_file = os.path.join(handler_output_dir, file)

        # Load the latest file if found
        if latest_file:
            print(f"Loading data from file: {latest_file}")
            with open(latest_file, 'rb') as f:
                handler = pickle.load(f)
        else:
            print("No valid file found. Generating data from scratch...")
            handler = PecSmartCellDataHandler(data_dir)

    return handler


def extract_conditions(test_condition):
    pattern = r'(?P<Duty>\d+(\.\d+)?)\s*duty.*? (?P<C_rate>\d+(\.\d+)?)C (?P<Frequency>\d+(\.\d+)?)\s*Hz'
    match = re.search(pattern, test_condition)
    if match:
        return match.groupdict()
    else:
        return {'Duty': 100, 'C_rate': 1, 'Frequency': 0}


def reformat_label(label):
    # Replace frequency (f Hz)
    label = re.sub(r"(\d+)\s*Hz", lambda m: rf"$\SI{{{m.group(1)}}}{{\hertz}}$",
                   label, flags=re.IGNORECASE)

    # Replace duty cycle (d duty)
    label = re.sub(r"(\d+)\s*%\s*Duty", lambda m: rf"$\SI{{{m.group(1)}}}{{\percent}}$ duty",
                   label, flags=re.IGNORECASE)

    return label


def reformat_label_old(label, txt_mode='abbrv'):
    """
    Reformats a label from "f Hz d duty cycle pulse" to
    "$\SI{f}{\hertz} \SI{d}{\percent}$ duty cycle pulse".

    Args:
        label (str): Input label in the format "f Hz d duty cycle pulse".

    Returns:
        str: Reformatted label in LaTeX-compatible format.
    """
    # Check first if case is reference case
    if "reference" in label.lower():
        return label

    # Regular expression to match the format "x Hz y duty cycle pulse"
    match = re.match(r"(\d+)?\s*Hz.*?(\d+)?\s*%\s*Duty", label, re.IGNORECASE)
    f_hz = match.group(1) if match else None
    d_duty = match.group(2) if match else None
    if f_hz and d_duty:
        return rf"$\SI{{{f_hz}}}{{\hertz}}$ $\SI{{{d_duty}}}{{\percent}}$ duty cycle pulse"
    elif f_hz:
        return rf"$\SI{{{f_hz}}}{{\hertz}}$ pulse"
    elif d_duty:
        return rf"$\SI{{{d_duty}}}{{\percent}}$ duty cycle"
    else:
        warnings.warn(f"Label '{label}' is not in the expected format. Returning unaltered.")
        return label  # Return unchanged if neither is found


def scatter_plot_with_color_code(df, x_col, y_col, color_col, colormap='tab20'):
    """
    Creates a scatter plot where points are color-coded based on a categorical or numerical column.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to plot.
    x_col : str
        The name of the column to use for the x-axis.
    y_col : str
        The name of the column to use for the y-axis.
    color_col : str
        The name of the column whose unique values determine the color coding.
    colormap : str or matplotlib.colors.Colormap, optional
        The name of the colormap to use for assigning colors to the unique values in `color_col`.
        Defaults to 'tab20'.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object for the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib axes object for the plot.

    Example:
    --------
    fig, ax = scatter_plot_with_color_code(dataframe, 'x', 'y', 'category', colormap='rainbow')
    plt.show()
    """
    unique_data = df[color_col].unique()
    fig, ax = plt.subplots()
    # Generate a color map and assign colors to each unique value
    cmap = plt.cm.get_cmap(colormap)  # Get the colormap
    colors = cmap([i / len(unique_data) for i in range(len(unique_data))])
    for unique_data, color in zip(unique_data, colors):
        subset = df[df[color_col] == unique_data]
        ax.scatter(subset[x_col], subset[y_col], label=f"{unique_data}", color=color, s=30)
    return fig, ax


def update_style_dict(style_dict):
    tmp_dict = style_dict.copy()
    tmp_dict['label'] = reformat_label(tmp_dict['label'])
    return tmp_dict


def get_nearest_k_mohm(df, V_nom, col='k_mohm'):
    # Filter out rows where 'k_mohm' is NaN
    df_filtered = df.dropna(subset=[col])
    # Find the index where 'maxV' is closest to 'V_nom'
    idx = (df_filtered['maxV'] - V_nom).abs().idxmin()
    # Return the corresponding 'k_mohm' value
    return df_filtered.loc[idx, col]


def identify_dva_peaks_ici(hndlr):
    dva_dict = {}
    for cell, pscd in hndlr.merged_pscd.items():
        for rpt, ici in pscd.ici_dict.items():
            cycle = pscd.rpt_obj.rpt_summary.loc[rpt, 'fce']
            soh = pscd.rpt_obj.rpt_summary.loc[rpt, 'cap_normalised']
            if not ici.ica_df.empty and ici.ica_df.volt.max() > 4.1:
                dvadf = ici.ica_df.copy()
                dvadf = dvadf[dvadf.curr > 0]
                dvadf['cap'] = dvadf.mAh - dvadf.mAh.iloc[0]
                dvadf['dva_flt'] = gaussian_filter1d(np.gradient(dvadf.volt, dvadf.cap),
                                                     sigma=4, mode='nearest')
                dvadf = dvadf.reset_index()
                peak_idx = argrelextrema(dvadf.dva_flt.to_numpy(), np.greater)
                identifier = f'Cell{cell:.0f}_cycle{cycle}_soh{soh * 100:.2f}'
                peak_idx = np.append(peak_idx[0], dvadf.index[-1])
            dva_dict[identifier] = (dvadf, peak_idx)
    return dva_dict


data_from_scratch = 1
BASE_PATH = get_base_path_batt_lab_data()
data_dir = os.path.join(BASE_PATH, 'pulse_chrg_test/high_frequency_testing/PEC_export')
# data_dir = r"D:\PEC_logs\pulse_charge_logs"
op_dir = r"\\file00.chalmers.se\home\krifren\Provning\Analysis\pulse_charge\ThirdBatch_PEC"
handler_output_dir = os.path.join(BASE_PATH, r"pulse_chrg_test/high_frequency_testing/PEC_export/post_process_data")
# handler_output_dir = os.path.join(data_dir, 'post_processed_data')
os.makedirs(handler_output_dir, exist_ok=True)
handler = load_or_generate_data(data_from_scratch, data_dir, handler_output_dir)

handler.merge_test_condition_data()
mean_temps = handler.calculate_mean_temperature()
curr_date = datetime.today().strftime('%Y-%m-%d')
handler_op_file = f'full_post_process_data_dump_{curr_date}.pkl'
handler_output = os.path.join(handler_output_dir, handler_op_file)
# with open(handler_output, 'wb') as f_h:
#     pickle.dump(handler, f_h)

cap_90_dict = {}
for cell, pscd in handler.merged_pscd.items():
    pscd.fit_degradation_function()
    condition_dict = extract_conditions(pscd.formatted_metadata['TEST_CONDITION'])
    cap_90_dict[cell] = [pscd.find_fce_at_given_q(0.9), mean_temps.loc[cell, 'Mean Temperature'], *condition_dict.values()]
cap_90_df = pd.DataFrame.from_dict(cap_90_dict, orient='index')
cap_90_df.columns = ['fce_at_q', 'mean_temp', *condition_dict.keys()]
cap_90_df['Duty'] = cap_90_df['Duty'].astype(float)
cap_90_df['C_rate'] = cap_90_df['C_rate'].astype(float)
cap_90_df['Frequency'] = cap_90_df['Frequency'].astype(float)
# ax_f = cap_90_df.plot.scatter(x='Frequency', y='fce_at_q', c='Duty', cmap='rainbow')
figf, ax_f = scatter_plot_with_color_code(cap_90_df, x_col='Frequency', y_col='fce_at_q', color_col='Duty', colormap='viridis')
ax_f.legend(title='Duty')
ax_f.set_xlabel(r'Frequency [\unit{\hertz}]')
ax_f.set_ylabel('Cycles at SoH 90 \% [-]')
figf_name = os.path.join(op_dir, 'freq_v_eol_scatter_c_dutyscience_style.pdf')
figf.savefig(figf_name)
figf.savefig(figf_name.replace('.pdf', '.png'), dpi=300)

# ax_T = cap_90_df.plot.scatter(x='mean_temp', y='fce_at_q', c='Duty', cmap='rainbow')
figt, ax_T = scatter_plot_with_color_code(cap_90_df, x_col='mean_temp', y_col='fce_at_q', color_col='Duty')
ax_T.legend(title='Duty', loc='lower right')
ax_T.set_xlabel(r'Mean temperature [\unit{\degreeCelsius}]')
ax_T.set_ylabel('Cycles at SoH 90 \% [-]')
figt_name = os.path.join(op_dir, 'temperature_v_eol_scatter_c_dutyscience_style.pdf')
figt.savefig(figt_name)
figt.savefig(figt_name.replace('.pdf', '.png'), dpi=300)

# ax_d = cap_90_df.plot.scatter(x='Duty', y='fce_at_q', c='Frequency', cmap='rainbow')
figd, ax_d = scatter_plot_with_color_code(cap_90_df, x_col='Duty', y_col='fce_at_q', color_col='Frequency', colormap='viridis')
ax_d.legend(title=r'Frequency [\unit{\hertz}]')
ax_d.set_xlabel(r'Duty cycle ratio [\%]')
ax_d.set_ylabel('Cycles at SoH 90 \% [-]')
figd_name = os.path.join(op_dir, 'duty_v_eol_scatter_c_freqscience_style.pdf')
figd.savefig(figd_name)
figd.savefig(figd_name.replace('.pdf', '.png'), dpi=300)

cap_90_corr = cap_90_df.drop('C_rate', axis=1).corr()

ax_corr = sns.heatmap(
    cap_90_corr,
    cmap="rainbow",
    square=True,
    cbar_kws={'label': 'Feature Importance'}
)

cap_90_ici = {cell: pscd.filter_ici_on_cap([0.9]) for cell, pscd in handler.merged_pscd.items()}
i_fig, iax = plt.subplots(1, 1)
for cell, ici_dict in cap_90_ici.items():
    for ici_idx, ici_obj in ici_dict.items():
        soh = handler.merged_pscd[cell].rpt_obj.rpt_summary.loc[ici_idx, 'cap_normalised']
        ici_obj = handler.merged_pscd[cell].ici_dict[ici_idx]
        idf = ici_obj.ici_result_df
        iax.scatter(idf[idf.ici_mode == 'dchg'].minV, idf[idf.ici_mode == 'dchg'].k_mohm,
                    label=f'Cell {cell}, SOH {soh*100:.2f}%')
iax.set_xlabel(r'Voltage [$\unit{\volt}$]')
iax.set_ylabel(r'k [$\unit{{\milli\ohm\per\sqrt{\second}}}$]')
iax.legend()

ica_op = r"\\file00.chalmers.se\home\krifren\Provning\Analysis\pulse_charge\ThirdBatch_PEC\ica_plots"
summary_df = pd.DataFrame()
for test_case, dct in handler.merged_condition_data.items():
    tmp_df = dct['merged_df'].filter(like='cap_norm')
    summary_df = pd.concat([summary_df, tmp_df], axis=1)
ica_summary_df = summary_df.iloc[::2]
bol_ref = handler.merged_pscd[231].ici_dict[0].ica_df
bol_ref = bol_ref[bol_ref.ica_mode == 'chrg']

### PERFORM REPLICATE-COMPARISONS, PUT CELL NUMBER IN LEGEND
rpt_cases = {
    232: [16],  # [6, 8],
    233: [12], # [10, 14]
}
soh_string = 'soh'
fig, ax = plt.subplots(1, 1)
ax.plot(bol_ref.volt, bol_ref.ica_gauss, label='BoL Reference')
for cell, rptlist in rpt_cases.items():
    for rpt in rptlist:
        ici_analysis = handler.merged_pscd[cell].ici_dict[rpt]
        tmpdf = ici_analysis.ica_df[ici_analysis.ica_df.ica_mode == 'chrg']
        soh = handler.merged_pscd[cell].rpt_obj.rpt_summary.loc[rpt, 'cap_normalised']
        soh_string = soh_string + f'_{soh*100:.0f}'
        case_name = handler.merged_pscd[cell].style['label']
        ax.plot(tmpdf.volt, tmpdf.ica_gauss, label=f'Cell {cell} at SoH {soh*100:.1f}')
ax.legend()
ax.set_xlabel(r'Voltage [$\unit{\volt}$]')
ax.set_ylabel(r'Incremental Capacity $\frac{\mathrm{d}Q}{\mathrm{d}V}$ [$\SI{}{\milli\ampere\hour\per\volt}$]')
cell_nbrs_as_strings = '_'.join([f'{k}' for k in rpt_cases.keys()])
ica_name = os.path.join(ica_op, f'ICA_cells_{cell_nbrs_as_strings}_{soh_string}_w_bol_ref.png')
# fig.savefig(ica_name, dpi=400)
# fig.savefig(ica_name.replace('.png', '.pdf'))


########################################################################################################################
################################# ICA CROSS-COMPARISONS, PUT CASE ABBRV NAME IN LEGEND  ################################
########################################################################################################################
plot_style = 'same_cycle'
if plot_style == 'same_soh':
    rpt_cases = {
        220: [0, 16, 24],
        232: [0, 8, 12],
        222: [0, 16, 24],
        238: [0, 4, 6]
    }
    # rpt_cases = {
    #     231: [0, 4, 8, 10],  # [6, 8],
    #     236: [0, 6, 10, 12],  # [10, 14]
    #     235: [0, 6, 12, 16],
    #     233: [0, 6, 12, 16],
    # }
else:
    rpt_cases = {
        231: np.arange(0, 17, 2),  # [6, 8],
        237: np.arange(0, 17, 2),  # [10, 14]
        238: np.arange(0, 17, 2),
        232: np.arange(0, 17, 2),
    }
max_rpt_length = max(len(rptlist) for rptlist in rpt_cases.values())
bbox = {'facecolor': 'white', 'boxstyle': 'round'}
for rpt_index in range(max_rpt_length):
    fig, ax = plt.subplots(1, 1)
    ax.plot(bol_ref.volt, bol_ref.ica_gauss, label='BoL Reference', color='forestgreen')
    soh_string = 'soh'
    soh_list = []
    for cell, rptlist in rpt_cases.items():
        temp_style = handler.merged_pscd[cell].style.copy()
        temp_style.pop('marker')
        temp_style.pop('linestyle')
        if rpt_index < len(rptlist):
            rpt = rptlist[rpt_index]
            ici_analysis = handler.merged_pscd[cell].ici_dict[rpt]
            tmpdf = ici_analysis.ica_df[ici_analysis.ica_df.ica_mode == 'chrg']
            soh = handler.merged_pscd[cell].rpt_obj.rpt_summary.loc[rpt, 'cap_normalised']
            cycle = handler.merged_pscd[cell].rpt_obj.rpt_summary.loc[rpt, 'fce']
            soh_list.append(soh)
            soh_string = soh_string + f'_{soh*100:.0f}'
            case_name = handler.merged_pscd[cell].style['label']
            # ax.plot(tmpdf.volt, tmpdf.ica_gauss, label=f'{case_name} at SoH {soh*100:.1f}')
            ax.plot(tmpdf.volt, tmpdf.ica_gauss, **temp_style)
    soh_avg = np.array(soh_list).mean()
    if plot_style == 'same_soh':
        ax.text(3.1, 3.5, f'SoH {soh_avg*100:.0f}\%', bbox=bbox)
    else:
        ax.text(3.1, 3.5, f'Cycle {cycle:.0f}', bbox=bbox)
    ax.set_xlabel(r'Voltage [$\unit{\volt}$]')
    ax.set_ylabel(r'Incremental Capacity $\frac{\mathrm{d}Q}{\mathrm{d}V}$ [$\unit{{\ampere\hour}\per\volt}$]')
    cell_nbrs_as_strings = '_'.join([f'{k}' for k in rpt_cases.keys()])
    ica_name = os.path.join(ica_op, f'ICA_{plot_style}_cells_{cell_nbrs_as_strings}_{soh_string}_solid_line.png')
    ax.set_ylim((-0.2, 9.5))
    ax.legend()
    fig.savefig(ica_name, dpi=400)
    fig.savefig(ica_name.replace('.png', '.pdf'))



########################################################################################################################
################################# DVA CROSS-COMPARISONS, PUT CASE ABBRV NAME IN LEGEND  ################################
########################################################################################################################
dva_op = r"\\file00.chalmers.se\home\krifren\Provning\Analysis\pulse_charge\ThirdBatch_PEC\dva_plots"
os.makedirs(dva_op, exist_ok=True)
plot_style = 'same_soh'
if plot_style == 'same_soh':
    # rpt_cases = {
    #     220: [0, 16, 24],
    #     232: [0, 8, 12],
    #     222: [0, 16, 24],
    #     238: [0, 4, 6]
    # }
    rpt_cases = {
        231: [0, 4, 8, 10],  # [6, 8],
        236: [0, 6, 10, 12],  # [10, 14]
        235: [0, 6, 12, 16],
        233: [0, 6, 12, 16],
    }
else:
    rpt_cases = {
        231: np.arange(0, 17, 2),  # [6, 8],
        237: np.arange(0, 17, 2),  # [10, 14]
        238: np.arange(0, 17, 2),
        232: np.arange(0, 17, 2),
    }
max_rpt_length = max(len(rptlist) for rptlist in rpt_cases.values())
bbox = {'facecolor': 'white', 'boxstyle': 'round'}
for rpt_index in range(max_rpt_length):
    fig, ax = plt.subplots(1, 1)
    dva_gauss = gaussian_filter1d(np.gradient(bol_ref.volt, bol_ref.mAh), sigma=2, mode='nearest')
    ax.plot(bol_ref.mAh - bol_ref.mAh.min(), dva_gauss, label='BoL Reference', color='forestgreen')
    soh_string = 'soh'
    soh_list = []
    for cell, rptlist in rpt_cases.items():
        temp_style = handler.merged_pscd[cell].style.copy()
        temp_style.pop('marker')
        temp_style.pop('linestyle')
        if rpt_index < len(rptlist):
            rpt = rptlist[rpt_index]
            ici_analysis = handler.merged_pscd[cell].ici_dict[rpt]
            tmpdf = ici_analysis.ica_df[ici_analysis.ica_df.ica_mode == 'chrg']
            soh = handler.merged_pscd[cell].rpt_obj.rpt_summary.loc[rpt, 'cap_normalised']
            cycle = handler.merged_pscd[cell].rpt_obj.rpt_summary.loc[rpt, 'fce']
            soh_list.append(soh)
            soh_string = soh_string + f'_{soh*100:.0f}'
            case_name = handler.merged_pscd[cell].style['label']
            dva_gauss = gaussian_filter1d(np.gradient(tmpdf.volt, tmpdf.mAh), sigma=2, mode='nearest')
            ax.plot(tmpdf.mAh - tmpdf.mAh.min(), dva_gauss, **temp_style)
    soh_avg = np.array(soh_list).mean()
    ax.set_ylim((-0.1, 1.45))
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.legend()
    if plot_style == 'same_soh':
        ax.text(xmin + 0.45, ymax - 0.25, f'SoH {soh_avg*100:.0f}\%', bbox=bbox)
    else:
        ax.text(xmin + 0.45, ymax - 0.25, f'Cycle {cycle:.0f}', bbox=bbox)
    ax.set_xlabel(r'Capacity [$\unit{{\ampere\hour}}$]')
    ax.set_ylabel(r'Differential Voltage $\frac{\mathrm{d}V}{\mathrm{d}Q}$ [$\unit{\volt\per{\ampere\hour}}$]')
    cell_nbrs_as_strings = '_'.join([f'{k}' for k in rpt_cases.keys()])
    dva_name = os.path.join(dva_op, f'DVA_{plot_style}_cells_{cell_nbrs_as_strings}_{soh_string}_solid_line.png')
    fig.savefig(dva_name, dpi=400)
    fig.savefig(dva_name.replace('.png', '.pdf'))
# plt.close('all')


########################################################################################################################
################################ ICI k CROSS-COMPARISONS, PUT CASE ABBRV NAME IN LEGEND  ###############################
########################################################################################################################
ici_op = r"\\file00.chalmers.se\home\krifren\Provning\Analysis\pulse_charge\ThirdBatch_PEC\ici_plots"
plot_style = 'same_soh'
if plot_style == 'same_soh':
    # rpt_cases = {
    #     220: [0, 16, 24],
    #     232: [0, 8, 12],
    #     222: [0, 16, 24],
    #     238: [0, 4, 6]
    # }
    rpt_cases = {
        231: [0, 4, 8, 10],  # [6, 8],
        236: [0, 6, 10, 12],  # [10, 14]
        235: [0, 6, 12, 16],
        233: [0, 6, 12, 16],
    }
else:
    rpt_cases = {
        230: np.arange(0, 17, 2),  # [6, 8],
        236: np.arange(0, 17, 2),  # [10, 14]
        235: np.arange(0, 17, 2),
        233: np.arange(0, 17, 2),
    }

max_rpt_length = max(len(rptlist) for rptlist in rpt_cases.values())

for rpt_index in range(max_rpt_length):
    fig, ax = plt.subplots(1, 1)
    soh_string = 'soh'
    soh_list = []

    for cell, rptlist in rpt_cases.items():
        # Check if the current cell has enough rpt entries to plot
        if rpt_index < len(rptlist):
            rpt = rptlist[rpt_index]
            ici_analysis = handler.merged_pscd[cell].ici_dict[rpt]
            tmpdf = ici_analysis.ici_result_df
            tmpdf = tmpdf[tmpdf.ici_mode == 'chrg']
            soh = handler.merged_pscd[cell].rpt_obj.rpt_summary.loc[rpt, 'cap_normalised']
            cycle = handler.merged_pscd[cell].rpt_obj.rpt_summary.loc[rpt, 'fce']
            temp_style = handler.merged_pscd[cell].style.copy()
            temp_style.pop('linestyle')
            soh_string = soh_string + f'_{soh * 100:.0f}'
            soh_list.append(soh)
            ax.scatter(tmpdf.maxV, tmpdf.k_mohm, **temp_style, s=6)
    soh_avg = np.array(soh_list).mean()
    if plot_style == 'same_soh':
        ax.text(3.4, 7.4, f'SoH {soh_avg*100:.0f}\%', bbox={'facecolor': 'white', 'boxstyle': 'round'})
    else:
        ax.text(3.4, 7.4, f'Cycle {cycle:.0f}', bbox={'facecolor': 'white', 'boxstyle': 'round'})
    ax.set_xlabel(r'Voltage [$\unit{\volt}$]')
    ax.set_ylabel(r'Diffusion resistance $k$ [$\unit{\milli\ohm\per\sqrt\second}$]')
    # ax.set_ylabel(r'R\textsubscript{10} [$\unit{\milli\ohm}$]')
    cell_nbrs_as_strings = '_'.join([f'{k}' for k in rpt_cases.keys()])
    ici_name = os.path.join(ici_op, f'ICI_k_mohm_{plot_style}_cells_{cell_nbrs_as_strings}_{soh_string}.png')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim((ymin, 8.5))
    ax.legend(loc='upper right')
    fig.savefig(ici_name, dpi=400)
    fig.savefig(ici_name.replace('.png', '.pdf'))



########################################################################################################################
############################### ICI R10 CROSS-COMPARISONS, PUT CASE ABBRV NAME IN LEGEND  ##############################
########################################################################################################################
ici_op = r"\\file00.chalmers.se\home\krifren\Provning\Analysis\pulse_charge\ThirdBatch_PEC\ici_plots"
plot_style = 'same_soh'
if plot_style == 'same_soh':
    rpt_cases = {
        231: [0, 4, 8, 10],  # [6, 8],
        236: [0, 6, 10, 12],  # [10, 14]
        235: [0, 6, 12, 16],
        233: [0, 6, 12, 16],
    }
else:
    rpt_cases = {
        230: np.arange(0, 17, 2),  # [6, 8],
        236: np.arange(0, 17, 2),  # [10, 14]
        235: np.arange(0, 17, 2),
        233: np.arange(0, 17, 2),
    }

max_rpt_length = max(len(rptlist) for rptlist in rpt_cases.values())

for rpt_index in range(max_rpt_length):
    fig, ax = plt.subplots(1, 1)
    soh_string = 'soh'
    soh_list = []

    for cell, rptlist in rpt_cases.items():
        # Check if the current cell has enough rpt entries to plot
        if rpt_index < len(rptlist):
            rpt = rptlist[rpt_index]
            ici_analysis = handler.merged_pscd[cell].ici_dict[rpt]
            tmpdf = ici_analysis.ici_result_df
            tmpdf = tmpdf[tmpdf.ici_mode == 'chrg']
            soh = handler.merged_pscd[cell].rpt_obj.rpt_summary.loc[rpt, 'cap_normalised']
            cycle = handler.merged_pscd[cell].rpt_obj.rpt_summary.loc[rpt, 'fce']
            temp_style = handler.merged_pscd[cell].style.copy()
            temp_style.pop('linestyle')
            soh_string = soh_string + f'_{soh * 100:.0f}'
            soh_list.append(soh)
            ax.scatter(tmpdf.maxV, tmpdf.R10_mohm, **temp_style, s=6)
    soh_avg = np.array(soh_list).mean()
    if plot_style == 'same_soh':
        ax.text(3.13, 50, f'SoH {soh_avg*100:.0f}\%', bbox={'facecolor': 'white', 'boxstyle': 'round'})
    else:
        ax.text(3.13, 50, f'Cycle {cycle:.0f}', bbox={'facecolor': 'white', 'boxstyle': 'round'})
    ax.set_xlabel(r'Voltage [$\unit{\volt}$]')
    ax.set_ylabel(r'\SI{10}{\second} resistance $R_{\text{10}}$ [$\unit{\milli\ohm}$]')
    cell_nbrs_as_strings = '_'.join([f'{k}' for k in rpt_cases.keys()])
    ici_name_r10 = os.path.join(ici_op, f'ICI_R10_mohm_cells_{plot_style}_{cell_nbrs_as_strings}_{soh_string}.png')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim((ymin, 78))
    ax.set_xlim((3.1, 4.25))
    ax.legend(loc='upper right')
    fig.savefig(ici_name_r10, dpi=400)
    fig.savefig(ici_name_r10.replace('.png', '.pdf'))



########################################################################################################################
############################### ICI R0 CROSS-COMPARISONS, PUT CASE ABBRV NAME IN LEGEND  ###############################
########################################################################################################################
ici_op = r"\\file00.chalmers.se\home\krifren\Provning\Analysis\pulse_charge\ThirdBatch_PEC\ici_plots"
plot_style = 'same_soh'
if plot_style == 'same_soh':
    rpt_cases = {
        231: [0, 4, 8, 10],  # [6, 8],
        236: [0, 6, 10, 12],  # [10, 14]
        235: [0, 6, 12, 16],
        233: [0, 6, 12, 16],
    }
else:
    rpt_cases = {
        230: np.arange(0, 17, 2),  # [6, 8],
        236: np.arange(0, 17, 2),  # [10, 14]
        235: np.arange(0, 17, 2),
        233: np.arange(0, 17, 2),
    }

max_rpt_length = max(len(rptlist) for rptlist in rpt_cases.values())

for rpt_index in range(max_rpt_length):
    fig, ax = plt.subplots(1, 1)
    soh_string = 'soh'
    soh_list = []

    for cell, rptlist in rpt_cases.items():
        # Check if the current cell has enough rpt entries to plot
        if rpt_index < len(rptlist):
            rpt = rptlist[rpt_index]
            ici_analysis = handler.merged_pscd[cell].ici_dict[rpt]
            tmpdf = ici_analysis.ici_result_df
            tmpdf = tmpdf[tmpdf.ici_mode == 'chrg']
            soh = handler.merged_pscd[cell].rpt_obj.rpt_summary.loc[rpt, 'cap_normalised']
            cycle = handler.merged_pscd[cell].rpt_obj.rpt_summary.loc[rpt, 'fce']
            temp_style = handler.merged_pscd[cell].style.copy()
            temp_style.pop('linestyle')
            soh_string = soh_string + f'_{soh * 100:.0f}'
            soh_list.append(soh)
            ax.scatter(tmpdf.maxV, tmpdf.R0_mohm, **temp_style, s=20)
    soh_avg = np.array(soh_list).mean()
    ymin, ymax = ax.get_ylim()
    ax.set_ylim((ymin, 43))
    if plot_style == 'same_soh':
        ax.text(3.2, 40, f'SoH {soh_avg*100:.0f}\%', bbox={'facecolor': 'white', 'boxstyle': 'round'})
    else:
        ax.text(3.2, 40, f'Cycle {cycle:.0f}', bbox={'facecolor': 'white', 'boxstyle': 'round'})
    ax.set_xlabel(r'Voltage [$\unit{\volt}$]')
    ax.set_ylabel(r'Ohmic resistance $R_{\text{0}}$ [$\unit{\milli\ohm}$]')
    cell_nbrs_as_strings = '_'.join([f'{k}' for k in rpt_cases.keys()])
    ici_name_r10 = os.path.join(ici_op, f'ICI_R0_mohm_{plot_style}_cells_{cell_nbrs_as_strings}_{soh_string}.png')

    ax.set_xlim((3.1, 4.25))
    ax.legend(loc='upper right')
    fig.savefig(ici_name_r10, dpi=400)
    fig.savefig(ici_name_r10.replace('.png', '.pdf'))


ica_fig, icax = plt.subplots(1, 1)
for cell, ici_dict in cap_90_ici.items():
    test_metadata = handler.merged_pscd[cell].formatted_metadata
    if '1 Hz' in test_metadata['TEST_CONDITION']:
        for ici_idx, ici_obj in ici_dict.items():
            soh = handler.merged_pscd[cell].rpt_obj.rpt_summary.loc[ici_idx, 'cap_normalised']
            ici_obj = handler.merged_pscd[cell].ici_dict[ici_idx]
            for cmode in ['chrg']:
                idf = ici_obj.ica_df
                icax.plot(idf[idf.ica_mode == cmode].volt, idf[idf.ica_mode == cmode].ica_gauss,
                          label=f'{test_metadata["OUTPUT_NAME"].replace("duty cycle pulse", "perc")}'
                                f', SOH {soh*100:.2f}%')
icax.set_xlabel(r'Voltage [$\unit{\volt}$]')
icax.set_ylabel(r'Incremental Capacity $\frac{\mathrm{d}Q}{\mathrm{d}V}$ [$\SI{}{\milli\ampere\hour\per\volt}$]')
# icax.set_ylabel(r'\frac{dQ}{dV} [$\unit{\milli{\ampere\hour}\per\volt}$]')
icax.legend()


fig, ax = plt.subplots(1, 1)
all_lines = []
for cell, pscd in handler.merged_pscd.items():
    repl_style = update_style_dict(pscd.line_styler.get_abbrv_style(pscd.formatted_metadata['OUTPUT_NAME']))
    red_df = pscd.rpt_obj.rpt_summary[pscd.rpt_obj.rpt_summary.fce < 330]
    line, = ax.plot(red_df.fce, red_df.cap_normalised, **repl_style)
    all_lines.append((line, repl_style['label']))
unique_lines = {label: line for line, label in all_lines}
ax.legend(unique_lines.values(), unique_lines.keys())
ax.set_xlabel('Cycles')
ax.set_ylabel('Capacity Retention')
_, ymax = ax.get_ylim()
_, xmax = ax.get_xlim()
ax.set_ylim(0.5, ymax)
ax.set_xlim(-25, xmax)
fig.savefig(os.path.join(op_dir, 'cap_retention_all_cells_high_f_science_style.pdf'))

for spec_group, dct in handler.merged_condition_data.items():
    tmpfig, ax = plt.subplots(1, 1)
    df = dct['merged_df'].copy()
    df = df[df.fce < 330]
    repl_style = dct['style'].copy()
    repl_style.pop('label')
    mean_style = {
        'label': 'Mean capacity',
        'linestyle': 'solid',
        'color': 'black',
        'marker': repl_style['marker'],
    }
    plot_cols = [col for col in df.columns if 'cap_normalised' in col and 'sigma' not in col]
    for c in plot_cols:
        line, = ax.plot(df.fce, df[c],
                        label='Replicate data',
                        **repl_style)
    line, = ax.plot(df.fce, df.mean_capacity, **mean_style)
    ax.set_xlabel('Cycles [-]')
    ax.set_ylabel('Capacity Retention [-]')
    lines, labels = ax.get_legend_handles_labels()
    unique_lines = {label: line for line, label in zip(lines, labels)}
    ax.set_ylim((0.55, 1.06))
    ax.legend(unique_lines.values(), unique_lines.keys(), loc='lower left')
    ax.text(0, 0.71, dct['style']['label'], bbox={'facecolor': 'white', 'boxstyle': 'round'})
    tmpfig_name = os.path.join(op_dir, f'mean_and_replicates_{spec_group.replace(" ", "_")}_fix_yaxis.pdf')
    tmpfig.savefig(tmpfig_name)
    tmpfig.savefig(tmpfig_name.replace('.pdf', '.png'), dpi=400)


exclude_key = 'colorsds'
fig2, ax_err = plt.subplots(1, 1)
fig3, ax_avg = plt.subplots(1, 1)
for cond, dct in handler.merged_condition_data.items():
    dct['style']['label'] = reformat_label(dct['style']['label'])
    tmp_style_dict = {k: val for k, val in dct['style'].items() if k != exclude_key}
    df_320 = dct['merged_df'][dct['merged_df'].fce < 330]
    ax_err.errorbar(df_320.fce, df_320.mean_capacity, yerr=df_320.sigma_capacity,
                    capsize=2, **dct['style'])
    ax_avg.plot(df_320.fce, df_320.mean_capacity, **tmp_style_dict)
ax_avg.legend()
ax_err.legend()
ax_avg.set_xlabel('Cycles [-]')
ax_avg.set_ylabel('Capacity retention [-]')
ax_avg.set_ylim((0.49, 1.02))
all_cell_avg_op = os.path.join(op_dir, 'mean_cap_retention_all_cells_high_fscience_style.pdf')
fig3.savefig(all_cell_avg_op)
fig3.savefig(all_cell_avg_op.replace('.pdf', '.png'), dpi=400)


cell_r_vs_soh = {}
col = 'R10_mohm'
if 'R10' in col:
    y_lab_pt = r'Normalised $R_\textrm{10,pt}$ [-]'
    y_lab_avg = r'Normalised $R_\textrm{10,avg}$ [-]'
    plot_name_pt = 'q_loss_v_R10_point.png'
    plot_name_avg = 'q_loss_v_R10_avg.png'
else:
    y_lab_pt = r'Normalised $k_\textrm{pt}$ [-]'
    y_lab_avg = r'Normalised $k_\textrm{avg}$ [-]'
    plot_name_pt = 'q_loss_v_k_point.png'
    plot_name_avg = 'q_loss_v_k_avg.png'

fig_pt, axpt = plt.subplots(1, 1)
fig_avg, axavg = plt.subplots(1, 1)
for cell, pscd in handler.merged_pscd.items():
    cap_list = []
    col_list_pt = []
    col_list_avg = []
    for rpt, ici in pscd.ici_dict.items():
        cap_list.append(pscd.rpt_obj.rpt_summary.loc[rpt, 'cap_normalised'])
        col_list_pt.append(get_nearest_k_mohm(ici.ici_result_df, 3.7, col=col))
        col_list_avg.append(ici.ici_result_df[col].mean())
    tmpdf = pd.DataFrame({'cap_norm': cap_list, f'{col}_pt': col_list_pt, f'{col}_avg': col_list_avg})
    tmpdf[f'{col}_pt_norm'] = tmpdf[f'{col}_pt'] / tmpdf[f'{col}_pt'].iloc[0]
    tmpdf[f'{col}_avg_norm'] = tmpdf[f'{col}_avg'] / tmpdf[f'{col}_avg'].iloc[0]
    cell_r_vs_soh[cell] = tmpdf
    style = pscd.line_styler.get_abbrv_style(pscd.formatted_metadata['OUTPUT_NAME']).copy()
    style.pop('linestyle')
    axavg.scatter(1 - tmpdf.cap_norm, tmpdf[f'{col}_avg_norm'], **style)
    axpt.scatter(1 - tmpdf.cap_norm, tmpdf[f'{col}_pt_norm'], **style)
x_lab = r'Normalised $Q_\textrm{loss}$ [-]'
axavg.set_xlabel(x_lab)
axpt.set_xlabel(x_lab)
axavg.set_ylabel(y_lab_avg)
axpt.set_ylabel(y_lab_pt)
lines, labels = axpt.get_legend_handles_labels()
unique_lines = {label: line for line, label in zip(lines, labels)}
axpt.legend(unique_lines.values(), unique_lines.keys())
axavg.legend(unique_lines.values(), unique_lines.keys())
pt_name = os.path.join(op_dir, plot_name_pt)
avg_name = os.path.join(op_dir,plot_name_avg)
fig_pt.savefig(pt_name, dpi=400)
fig_pt.savefig(pt_name.replace('.png', '.pdf'))
fig_avg.savefig(avg_name, dpi=400)
fig_avg.savefig(avg_name.replace('.png', '.pdf'))


dva_summary_dict = identify_dva_peaks_ici(handler)
dva_peaks = {k: df.loc[pks] for k, (df, pks) in dva_summary_dict.items()}

gr_peak1_rng = [3.47, 3.53]
gr_peak2_rng = [3.77, 3.85]
nmc_peak_rng = [4.0, 4.05]
fc_peak_rng = [4.19, 4.22]

for k, df in dva_peaks.items():
    df['peak1_mask'] = df.volt.between(*gr_peak1_rng)
    df['peak2_mask'] = df.volt.between(*gr_peak2_rng)
    df['nmc_peak_mask'] = df.volt.between(*nmc_peak_rng)
    df['fc_peak_mask'] = df.volt.between(*fc_peak_rng)

peak_cap = {}
for k, df in dva_peaks.items():
    if not df.empty:
        try:
            peak_cap[k] = [df[df.peak1_mask].cap.iloc[0],
                           df[df.peak2_mask].cap.iloc[0],
                           df[df.nmc_peak_mask].cap.iloc[0],
                           df[df.fc_peak_mask].cap.iloc[0]]
        except IndexError as e:
            print(f'Index error for {k}')
rows = []
for key, values in peak_cap.items():
    match = re.match(r'(Cell\d+)_cycle(\d+)_soh(\d+)', key)
    if match:
        cell, cycle, soh = match.groups()
        rows.append({
            'cell': cell,
            'cycle': int(cycle),
            'soh': float(soh) / 100,
            'cap_ne1': values[0],
            'cap_ne2': values[1],
            'cap_pe': values[2],
            'cap_fc': values[3],
        })

# Generate peak track and delta-cap df
dcap_df = pd.DataFrame(rows)  # .set_index('Cell')
dcap_df.sort_values(by=['cell', 'cycle'], inplace=True)
dcap_df['dcap_ne'] = dcap_df['cap_ne2'] - dcap_df['cap_ne1']
dcap_df['dcap_pe_ne'] = dcap_df['cap_pe'] - dcap_df['cap_ne2']
dcap_df['dcap_fc_pe'] = dcap_df['cap_fc'] - dcap_df['cap_pe']
for col in ['cap_ne1', 'cap_ne2', 'dcap_ne', 'cap_pe', 'dcap_pe_ne', 'dcap_fc_pe']:
    dcap_df[f'{col}_norm'] = dcap_df.groupby('cell')[col].transform(lambda x: x / x.iloc[0])


norm_cols = [c for c in dcap_df.columns if '_norm' in c]
names_for_plots = [
    r'$Q_\textrm{NE,peak1}$',
    r'$Q_\textrm{NE,peak2}$',
    r'$\Delta Q_\textrm{NE,peaks}$',
    r'$Q_\textrm{PE,peak}$',
    r'$\Delta Q_\textrm{PE-NE}$',
    r'$\Delta Q_\textrm{FC-PE}$',
]
plot_name_dict = dict(zip(norm_cols, names_for_plots))
dcap_op = r"\\file00.chalmers.se\home\krifren\Provning\Analysis\pulse_charge\FirstBatch\dm_analysis"
dcap_fit_dict = {}
x_, y_ = plt.rcParams['figure.figsize']
for col in norm_cols:
    # Create a dictionary to store results
    fit_results = []
    fig = plt.figure(figsize=(x_, x_))
    # plt.title(col)
    # Perform linear regression for each cell and store results
    for i, (cell, group) in enumerate(dcap_df.groupby('cell')):
        # RETRIEVE STYLE FOR THIS CELL
        cell_nbr = int(re.search(r'\d+', cell).group())
        style = handler.merged_pscd[cell_nbr].style.copy()
        scat_style = style.copy()
        scat_style.pop('linestyle')

        group = group.dropna(subset=['soh', col])
        slope, intercept, r_value, p_value, std_err = stats.linregress(group['soh'], group[col])
        fit_results.append({'cell': cell, 'slope': slope, 'PCC': r_value, 'R^2': r_value ** 2})
        # Scatter plot of actual data points
        plt.scatter(group['soh'], group[col], **scat_style)

        # Generate fit line (over the same range as actual data)
        # soh_range = np.linspace(0.5, 1.1, 10)
        # soh_range = np.linspace(group['soh'].min() / 1.15, group['soh'].max() * 1.15, 5)
        # fit_line = intercept + slope * soh_range
        # plt.plot(soh_range, fit_line, **style)
    soh_range = np.linspace(0.5, 1.15, 10)
    ax = plt.gca()
    lines, labels = ax.get_legend_handles_labels()
    unique_lines = {label: line for line, label in zip(lines, labels)}
    ax.legend(unique_lines.values(), unique_lines.keys())
    plt.plot(soh_range, soh_range, color='black', linestyle='--')
    plt.xlabel('SoH [-]')
    plt.ylabel(f'{plot_name_dict[col]} [-]')
    cap_col_name = os.path.join(dcap_op, f'cat2_{col}_with_trendline_and_fits_square.png')
    fig.savefig(cap_col_name, dpi=400)
    fig.savefig(cap_col_name.replace('.png', '.pdf'))
    # Convert results to a DataFrame
    fit_df = pd.DataFrame(fit_results)
    dcap_fit_dict[col] = fit_df



# MAKE STOCK IMAGES
rpt_op = "Z:/Documents/Papers/PulseChargingPaper"
bbox = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
rpt_fig, rax = plt.subplots(2, 1, sharex=True, figsize=(x_, 1.5*y_))
raw_df = handler.pscd_dict['2805_cell1'].rpt_dict['comp'][2]
raw_df = raw_df[raw_df.step != 115]
rax[0].plot((raw_df.float_time - raw_df.float_time.iloc[0]) / 3600, raw_df.volt, color='black')
rax[1].plot((raw_df.float_time - raw_df.float_time.iloc[0]) / 3600, raw_df.curr, color='black')
rax[0].set_ylabel(r'Voltage [\unit{\volt}]')
rax[1].set_ylabel(r'Current [\unit{\ampere}]')
rax[1].set_xlabel(r'Time [\unit{\hour}]')

# Color code the different parts of the rpt depending on action
cap_meas_step_final = 82
cap_end_time = raw_df.loc[raw_df[raw_df.step == cap_meas_step_final].last_valid_index(), 'float_time'] - raw_df.float_time.iloc[0]
ici_end_time = raw_df.loc[raw_df[raw_df.ici_bool == 1].last_valid_index(), 'float_time'] - raw_df.float_time.iloc[0]
fin_time = raw_df.float_time.max() - raw_df.float_time.iloc[0]
rax[0].axvspan(-1, cap_end_time / 3600, facecolor='green', alpha=0.3)
rax[0].axvspan(cap_end_time / 3600, ici_end_time / 3600, facecolor='red', alpha=0.3)
rax[0].axvspan(ici_end_time / 3600, fin_time / 3600 + 2, facecolor='blue', alpha=0.3)
y_text_pos = 4.38
rax[0].text(0, y_text_pos, 'Capacity Test', bbox=bbox)
rax[0].text((cap_end_time + ici_end_time) / (2.5*3600), y_text_pos, 'ICI Test', bbox=bbox)
rax[0].text(ici_end_time / 3600 + 0.5, y_text_pos, 'Impedance \nTest', bbox=bbox)
rax[0].set_ylim(2.5, 4.8)
rpt_fig_labels = ['(a)', '(b)']
for k, label in enumerate(rpt_fig_labels):
    rax[k].text(-0.12, 1.0, label, transform=rax[k].transAxes, fontsize=7)
rpt_fig.savefig(os.path.join(rpt_op, 'rpt_visual_.png'), dpi=400)
rpt_fig.savefig(os.path.join(rpt_op, 'rpt_visual__science_style.pdf'))

rax[1].axvspan(-1, cap_end_time / 3600, facecolor='green', alpha=0.3)
rax[1].axvspan(cap_end_time / 3600, ici_end_time / 3600, facecolor='red', alpha=0.3)
rax[1].axvspan(ici_end_time / 3600, fin_time / 3600 + 1, facecolor='blue', alpha=0.3)
# rpt_fig.savefig(os.path.join(rpt_op, 'rpt_visual_full_color_code_larger.png'), dpi=400)
# rpt_fig.savefig(os.path.join(rpt_op, 'rpt_visual_full_color_codescience_style.pdf'))


# Perform t-test to see if difference between results are significant
data_sets = {}
for tname, merged_data in handler.merged_condition_data.items():
    data_sets[tname.replace('Pulse Charge ', '')] = merged_data['merged_df'].filter(like='cap_normalised').iloc[-1, :]
all_combs = permutations(data_sets.keys(), 2)

ttest_results = {}
for comb in all_combs:
    c1, c2 = comb
    if not c1 == c2:
        ttest_results[f'{c1}__{c2}'] = stats.ttest_rel(data_sets[c1], data_sets[c2])
prob_df = pd.DataFrame([{'SpecGroup1': k.split('__')[0], 'SpecGroup2': k.split('__')[1], 'pval': v.pvalue}
                            for k, v in ttest_results.items()])
pmatrix = prob_df.pivot(index='SpecGroup1', columns='SpecGroup2', values='pval')
pmatrix = pmatrix.fillna(1)
pmat_triu = np.triu(pmatrix)
pmat_triu = pd.DataFrame(pmat_triu, index=pmatrix.index, columns=pmatrix.columns)
