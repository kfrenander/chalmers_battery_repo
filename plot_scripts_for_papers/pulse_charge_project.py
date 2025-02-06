import pandas as pd
from rpt_data_analysis.post_process_cycling_data import CycleAgeingDataIndexer
from misc_classes.pulse_charge_style_class import TestCaseStyler
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import numpy as np
from natsort import natsorted
from scipy import stats
from itertools import permutations
import re
import os


plt.style.use(['science', 'nature_w_si_harding', 'grid'])
mark_size = 5
cap_size = 6
plt.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r'\usepackage{siunitx}'
        })
output_dir = r'\\file00.chalmers.se\home\krifren\Provning\Analysis\pulse_charge\FirstBatch_natureformat'


def calculate_fce_from_rpt(rpt_str):
    rpt_num = int(re.findall(r'\d+', rpt_str)[0])
    fce_num = (rpt_num - 1) * 40
    return fce_num


def merge_dicts_with_suffix(dict1, dict2, suffix="_2"):
    merged_dict = dict1.copy()  # Start with a copy of the first dictionary
    for key, value in dict2.items():
        if key in merged_dict:
            new_key = f"{key}{suffix}"
            # Ensure the new key doesn't also conflict
            counter = 2
            while new_key in merged_dict:
                new_key = f"{key}_{counter}"
                counter += 1
            merged_dict[new_key] = value
        else:
            merged_dict[key] = value
    return merged_dict


def extract_info(input_string):
    # Check if it is reference test
    if "Reference test" in input_string:
        return 0.0, False
    # Define the regex pattern
    pattern = r"(\d+)(?:mHz) Pulse Charge(?: no pulse discharge)?"

    # Match the pattern in the input string
    match = re.match(pattern, input_string)

    if match:
        # Extract frequency as a float
        frequency = float(match.group(1))

        # Determine if "no pulse discharge" is present
        pulse_discharge = not input_string.endswith("no pulse discharge")

        return frequency, pulse_discharge
    else:
        raise ValueError("Input string does not match the expected format.")


def make_fce_at_q_df(cycle_data, q_lvl=0.92):
    cap_dict = {ch: [ag_data.find_fce_at_given_q(q_lvl), ag_data.meta_data.cell_id, ag_data.TEST_NAME]
                for ch, ag_data in cycle_data.ageing_data.items()}
    cap_df = pd.DataFrame.from_dict(cap_dict, orient='index', columns=['fce_at_q', 'cell_id', 'test_condition'])
    cap_df[['freq_mhz', 'pulse_discharge']] = cap_df['test_condition'].apply(lambda x: pd.Series(extract_info(x)))
    cap_df['freq_hz'] = cap_df['freq_mhz'] / 1000
    return cap_df.set_index('cell_id')


if __name__ == '__main__':
    from check_current_os import get_base_path_batt_lab_data
    BASE_DATA_PATH = get_base_path_batt_lab_data()
    data_loc_batch1 = os.path.join(BASE_DATA_PATH, 'pulse_chrg_test/cycling_data')
    data_loc_batch2 = os.path.join(BASE_DATA_PATH, 'pulse_chrg_test/cycling_data_repaired')
    cycle_data_batch1 = CycleAgeingDataIndexer()
    cycle_data_batch1.run(data_loc_batch1)
    cycle_data_batch2 = CycleAgeingDataIndexer()
    cycle_data_batch2.run(data_loc_batch2)
    merged_data = CycleAgeingDataIndexer()
    merged_data.ageing_data = merge_dicts_with_suffix(cycle_data_batch1.ageing_data, cycle_data_batch2.ageing_data)
    merged_data.generate_arbitrary_replicates_combined_data()
    style_class = TestCaseStyler()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    capdf1 = make_fce_at_q_df(cycle_data=cycle_data_batch1)
    capdf2 = make_fce_at_q_df(cycle_data=cycle_data_batch2)
    capdf = pd.concat([capdf1, capdf2])
    ax = capdf.plot.scatter(x='freq_mhz', y='fce_at_q')
    ax.semilogx()
    ax.set_xlabel(r'Frequency [\unit{\milli\hertz}]')
    ax.set_ylabel(r'Cycles at 90\% SOH')
    ax_no_pulse = capdf[~capdf.pulse_discharge].plot.scatter(x='freq_mhz', y='fce_at_q')
    ax_no_pulse.semilogx()
    ax_no_pulse.set_xlabel(r'Frequency [\unit{\milli\hertz}]')
    ax_no_pulse.set_ylabel(r'Cycles at 90\% SOH')
    ax_pulse = capdf[capdf.pulse_discharge].plot.scatter(x='freq_mhz', y='fce_at_q')
    ax_pulse.semilogx()
    ax_pulse.set_xlabel(r'Frequency [\unit{\milli\hertz}]')
    ax_pulse.set_ylabel(r'Cycles at 90\% SOH')

    all_lines = []
    fig, ax = plt.subplots(1, 1)
    for k, ag_data in cycle_data_batch1.ageing_data.items():
        tmp_df = ag_data.rpt_data
        line, = ax.plot(tmp_df.fce, tmp_df.cap_relative,
                        **style_class.get_abbrv_style(ag_data.TEST_NAME))
        all_lines.append((line, style_class.get_abbrv_style(ag_data.TEST_NAME)['label']))
    unique_lines = {label: line for line, label in all_lines}
    ax.legend(unique_lines.values(), unique_lines.keys())
    plt.legend(ncols=2)
    ax.set_xlabel('FCE [-]')
    ax.set_ylabel('Capacity retention [-]')
    fig.savefig(os.path.join(output_dir, 'all_cells_relative_capacity.png'), dpi=400)

    x_size, y_size = plt.rcParams['figure.figsize']
    bbox = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
    fig, ax = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    fig3, ax3 = plt.subplots(2, 4, figsize=(3*x_size, 1.5*y_size), sharey='all', sharex='all')
    all_lines = []
    i = 0
    ax3 = ax3.ravel()
    for test_case, df in merged_data.combined_data_arbitrary_replicates.items():
        tmp_fig, tmp_ax = plt.subplots(1, 1)
        style_dct = style_class.get_abbrv_style(test_case)
        line, = ax.plot(df.fce, df.avg_rel_cap, **style_dct)
        ax2.errorbar(x=df.fce, y=df.avg_rel_cap, yerr=df.std_rel_cap, capsize=4, **style_dct)
        # ax2.fill_between(df.fce, df.avg_rel_cap - df.avg_rel_cap, df.avg_rel_cap + df.std_rel_cap, alpha=0.3, color=style_dct['color'])
        plot_cols = [col for col in df.columns if 'cap_relative' in col and 'sigma' not in col]
        df.plot(x='fce', y=plot_cols, title=test_case, ax=ax3[i])
        repl_style = style_dct.copy()
        repl_style.pop('label')
        for c in plot_cols:
            tmp_ax.plot(df.fce, df[c], label='Replicate data', **repl_style)
        # df.plot(x='fce', y=plot_cols, ax=tmp_ax)
        # for idx, line in enumerate(tmp_ax.get_lines()):
        #     line.set_color(style_dct['color'])
        #     line.set_marker(style_dct['marker'])
        #     line.set_linestyle(style_dct['linestyle'])
        #     line.set_label(f'Replicate {idx + 1}')
        tmp_ax.plot(df.fce, df[plot_cols].mean(axis=1), color='black', marker=repl_style['marker'], label='Mean capacity')
        tmp_ax.set_xlabel('Cycles [-]')
        tmp_ax.set_ylabel('Capacity retention [-]')
        tmp_ax.set_ylim(((0.6053, 1.0187)))
        tmp_ax.set_xlim((-20.0, 420.0))
        tmp_ax.text(20, 0.8, f'{style_dct["label"]}', bbox=bbox)
        lines, labels = tmp_ax.get_legend_handles_labels()
        unique_lines = {label: line for line, label in zip(lines, labels)}
        tmp_ax.legend(unique_lines.values(), unique_lines.keys(), loc='lower left')
        tmp_fig_name = os.path.join(output_dir, f'{test_case.replace(" ", "_")}_all_replicates_science_style_abbrv_w_mean.pdf')
        tmp_fig.savefig(tmp_fig_name)
        tmp_fig.savefig(tmp_fig_name.replace('.pdf', '.png'), dpi=400)
        # plt.close(tmp_fig)
        for idx, line in enumerate(ax3[i].get_lines()):
            line.set_color(style_dct['color'])
            line.set_marker(style_dct['marker'])
            line.set_linestyle(style_dct['linestyle'])
            line.set_label(f'_Replicate data')
        ax3[i].get_lines()[0].set_label('Replicate data')
        ax3[i].plot(df.fce, df.avg_rel_cap,
                    color=style_dct['color'],
                    marker=style_dct['marker'],
                    linestyle='solid',
                    label='Average capacity')
        ax3[i].legend()
        ax3[i].set_xlabel('')
        i += 1
        all_lines.append((line, style_dct['label']))
    unique_lines = {label: line for line, label in all_lines}
    ax.legend(unique_lines.values(), unique_lines.keys())
    ax.set_xlabel('Cycles [-]')
    ax.set_ylabel('Capacity retention [-]')
    ax.set_ylim((0.57, 1.02))
    ax2.legend(unique_lines.values(), unique_lines.keys())
    ax2.set_xlabel('Cycles [-]')
    ax2.set_ylabel('Capacity retention [-]')
    fig3.supxlabel('Cycles [-]')
    fig3.supylabel('Normalised capacity retention [-]')
    mean_cap_op = os.path.join(output_dir, 'mean_cap_retention_all_cells_lowf_science_style.pdf')
    fig.savefig(mean_cap_op)
    fig.savefig(mean_cap_op.replace('.pdf', '.png'), dpi=300)
    fig2.savefig(os.path.join(output_dir, 'avg_cap_retention_all_cells_plus_errorbars_science_style.pdf'))
    all_subplots_fname = os.path.join(output_dir, 'subplots_all_replicates_one_figure_science_style.pdf')
    fig3.savefig(all_subplots_fname)
    fig3.savefig(all_subplots_fname.replace('.pdf', 'png'), dpi=300)

    fig, ax = plt.subplots(1, 1)
    all_lines = []
    for k, ag_data in cycle_data_batch1.ageing_data.items():
        tmp_df = ag_data.rpt_data
        style_dct = style_class.get_abbrv_style(ag_data.TEST_NAME)
        line, = ax.plot(tmp_df.fce, tmp_df.cap_relative, **style_dct)
        all_lines.append((line, style_dct['label']))
    unique_lines = {label: line for line, label in all_lines}
    ax.legend(unique_lines.values(), unique_lines.keys())
    ax.set_xlabel('Cycles [-]')
    ax.set_ylabel('Capacity retention [-]')
    # fig.savefig(os.path.join(output_dir, 'all_cells_relative_capacity_new_style.pdf'), dpi=400)


    fig, ax = plt.subplots(1, 1)
    for t_name, cmb_dat in merged_data.combined_data_arbitrary_replicates.items():
        style_dct = style_class.get_abbrv_style(t_name)
        ax.plot(cmb_dat.fce, cmb_dat.avg_rel_cap,
                **style_dct)
    # plt.legend(ncols=1)
    ax.set_xlabel('Cycles [-]')
    ax.set_ylabel('Capacity retention [-]')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + 0.03,
                     box.width, box.height])
    # plt.legend(loc='center right', bbox_to_anchor=(1.8, 0.5), fancybox=True, ncol=1)
    plt.legend()
    ax.set_yticks(np.arange(0.6, 1.05, 0.1))
    pdf_output = os.path.join(output_dir, 'mean_cap_retention_all_cases_low_f.png')
    fig.savefig(pdf_output.replace('.pdf', '.png'), dpi=400)
    fig.savefig(pdf_output)

    fig, ax = plt.subplots(1, 1)
    for t_name, cmb_dat in cycle_data_batch1.combined_data.items():
        ax.errorbar(cmb_dat.fce, cmb_dat.avg_rel_cap,
                    yerr=cmb_dat.std_rel_cap,
                    elinewidth=1.5,
                    color=cycle_data_batch1.visual_profile.COLORS[t_name],
                    marker=cycle_data_batch1.visual_profile.MARKERS[t_name],
                    label=t_name,
                    markersize=mark_size,
                    capsize=cap_size)
    plt.legend(ncols=2)
    ax.set_xlabel('FCE [-]')
    ax.set_ylabel('Capacity retention [-]')
    fig.savefig(os.path.join(output_dir, 'error_bar_all_cells.png'), dpi=400)

    ref_cases_dchg_pulsing = ['1000mHz Pulse Charge',
                              '1000mHz Pulse Charge no pulse discharge',
                              '100mHz Pulse Charge',
                              '100mHz Pulse Charge no pulse discharge']
    fig, ax = plt.subplots(1, 1)
    for t_name, cmb_dat in cycle_data_batch1.combined_data.items():
        if t_name in ref_cases_dchg_pulsing:
            ax.errorbar(cmb_dat.fce, cmb_dat.avg_rel_cap,
                        yerr=cmb_dat.std_rel_cap,
                        elinewidth=1.5,
                        color=cycle_data_batch1.visual_profile.COLORS[t_name],
                        marker=cycle_data_batch1.visual_profile.MARKERS[t_name],
                        label=t_name,
                        markersize=mark_size,
                        capsize=cap_size)
    plt.legend(ncols=1)
    ax.set_xlabel('FCE [-]')
    ax.set_ylabel('Capacity retention [-]')
    ax.grid(alpha=0.4, color='grey')
    fig.savefig(os.path.join(output_dir, 'error_bar_pulse_ref_cells.png'), dpi=400)

    fig, ax = plt.subplots(1, 1)
    for t_name, cmb_dat in cycle_data_batch1.combined_data.items():
        if '_no_pulse' not in t_name:
            ax.errorbar(cmb_dat.fce, cmb_dat.avg_rel_cap,
                        yerr=cmb_dat.std_rel_cap,
                        elinewidth=1.5,
                        color=cycle_data_batch1.visual_profile.COLORS[t_name],
                        marker=cycle_data_batch1.visual_profile.MARKERS[t_name],
                        label=t_name,
                        markersize=mark_size,
                        capsize=cap_size)
    plt.legend(ncols=1)
    ax.set_xlabel('FCE [-]')
    ax.set_ylabel('Capacity retention [-]')
    ax.grid(alpha=0.4, color='grey')
    fig.savefig(os.path.join(output_dir, 'error_bar_dchg_pulse_cells.png'), dpi=400)

    # plt.style.use('kelly_colors')
    plt.figure()
    for rpt, df in natsorted(cycle_data_batch1.ageing_data['240095_3_1'].ica_data.items()):
        plt.plot(df[df.curr > 0].cap, df[df.curr > 0].dva_gauss, label=f'FCE {calculate_fce_from_rpt(rpt)}')
    plt.ylim((0, 2))
    plt.legend()
    plt.xlabel(r'Charge capacity $\left[\SI{}{\milli\ampere\hour}\right]$')
    plt.ylabel(r'Differential voltage $\frac{\mathrm{d}V}{\mathrm{d}Q}$ '
               r'$\left[\unit[per-mode=fraction]{\volt\per\milli\ampere\per\hour}\right]$')

    plt.figure()
    for rpt, df in natsorted(cycle_data_batch1.ageing_data['240095_3_1'].ica_data.items()):
        plt.plot(df[df.curr > 0].volt, df[df.curr > 0].ica_gauss, label=f'FCE {calculate_fce_from_rpt(rpt)}')
    plt.xlabel(r'Voltage [$\unit{\volt}$]')
    plt.ylabel(r'Incremental capacity $\frac{\mathrm{d}Q}{\mathrm{d}V}$ [$\unit{\milli{\ampere\hour}\per\volt}$]')
    plt.legend()

    plt.figure()
    for cell, age_data in cycle_data_batch1.ageing_data.items():
        df = age_data.ica_data['rpt_1']
        plt.plot(df[df.curr > 0].volt, df[df.curr > 0].ica_gauss, color=age_data.visual_profile.COLOR)
    plt.xlabel(r'Voltage [$\SI{}{\volt}$]')
    plt.ylabel(r'Incremental capacity $\frac{\mathrm{d}Q}{\mathrm{d}V}$ [$\SI{}{\milli\ampere\hour\per\volt}$]')

    cases_to_plot = ['Reference test constant current', '1000mHz Pulse Charge']
    x_fig, y_fig = plt.rcParams['figure.figsize']
    plt.figure(figsize=(x_fig, y_fig))
    bol_rpt = cycle_data_batch1.ageing_data['240095_2_1'].ica_data['rpt_1']
    bol_rpt_pos = bol_rpt[bol_rpt.curr > 0]
    plt.plot(bol_rpt_pos.volt, bol_rpt_pos.ica_gauss, color='forestgreen', label='BoL reference')

    # Set to keep track of already plotted test names
    plotted_cases = set()
    rpt_cases = {
        140: 'rpt_11',
        156: 'rpt_5'
    }
    plot_type = 'same_cycle'

    for cell, age_data in cycle_data_batch1.ageing_data.items():
        if age_data.TEST_NAME in cases_to_plot and age_data.TEST_NAME not in plotted_cases:
            style_dct = style_class.get_abbrv_style(age_data.TEST_NAME)
            plotted_cases.add(age_data.TEST_NAME)
            cell_id = age_data.meta_data.cell_id
            print(f'Plotting for cell {cell_id}')
            if plot_type == 'same_soh':
                df = age_data.ica_data[rpt_cases[cell_id]]
                dtext = 'SoH = 92\%'
            else:
                df = age_data.ica_data['rpt_11']
                dtext = 'Cycle 400'
            plt.plot(df[df.curr > 0].volt, df[df.curr > 0].ica_gauss,
                     color=style_dct['color'],
                     label=style_dct['label'])
    plt.xlabel(r'Voltage $\left[\unit{\volt}\right]$')
    plt.ylabel(r'Incremental capacity $\frac{\mathrm{d}Q}{\mathrm{d}V}$ $[\unit{{\ampere\hour}\per\volt}]$')
    plt.text(3, 3.5, dtext,
             fontdict={'fontsize': 7, 'family': 'serif'},
             bbox={'facecolor': 'white', 'boxstyle': 'round'})
    plt.legend(ncol=1)
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + 0.03,
                     box.width, box.height])
    plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), fancybox=True, ncol=2)
    ica_fig_name = os.path.join(output_dir, f'EoL_ICA_1Hz_and_ref_w_bol_ref_single_replicate_{plot_type}.png')
    plt.savefig(ica_fig_name, dpi=400)
    plt.savefig(ica_fig_name.replace('.png', '.pdf'))


    ########################################################################################################################
    ################################# ICA CROSS-COMPARISONS, PUT CASE ABBRV NAME IN LEGEND  ################################
    ########################################################################################################################
    ica_op = r"\\file00.chalmers.se\home\krifren\Provning\Analysis\pulse_charge\low_f\ica_plots"
    os.makedirs(ica_op, exist_ok=True)
    cell_to_channel_mapper = {age_data.meta_data.cell_id: ch for ch, age_data in merged_data.ageing_data.items()}
    plot_style = 'same_cycle'
    if plot_style == 'same_soh':
        rpt_cases = {
            140: [1, 4, 11],  # [6, 8],
            # 142: [1, 4, 8],
            150: [1, 4, 7],  # [10, 14]
            155: [1, 4, 5],
            156: [1, 4, 5]
        }
    else:
        rpt_cases = {
            140: np.arange(1, 12, 1),  # [6, 8],
            142: np.arange(1, 12, 1),  # [10, 14]
            150: np.arange(1, 12, 1),
            156: np.arange(1, 12, 1),
        }

    max_rpt_length = max(len(rptlist) for rptlist in rpt_cases.values())

    for rpt_index in range(max_rpt_length):
        fig, ax = plt.subplots(1, 1)
        ax.plot(bol_rpt_pos.volt, bol_rpt_pos.ica_gauss, color='forestgreen', label='BoL ref')
        # ax.plot(bol_rpt_pos.cap, bol_rpt_pos.dva_gauss, color='forestgreen', label='BoL Ref')
        soh_string = 'soh'
        soh_list = []
        for cell, rptlist in rpt_cases.items():
            ch_id = cell_to_channel_mapper[cell]
            # Check if the current cell has enough rpt entries to plot
            if rpt_index < len(rptlist):
                rpt = f'rpt_{rptlist[rpt_index]}'
                tmpdf = merged_data.ageing_data[ch_id].ica_data[rpt]
                tmpdf = tmpdf[tmpdf.curr > 0]
                soh = merged_data.ageing_data[ch_id].rpt_data.loc[rpt, 'cap_relative']
                cycle = merged_data.ageing_data[ch_id].rpt_data.loc[rpt, 'fce']
                temp_style = style_class.get_abbrv_style(merged_data.ageing_data[ch_id].TEST_NAME).copy()
                temp_style.pop('linestyle')
                temp_style.pop('marker')
                soh_string = soh_string + f'_{soh * 100:.0f}'
                soh_list.append(soh)
                ax.plot(tmpdf.volt, tmpdf.ica_gauss, **temp_style)
        soh_avg = np.array(soh_list).mean()
        if plot_style == 'same_soh':
            ax.text(3.0, 3.5, f'SoH {soh_avg * 100:.0f}\%', bbox={'facecolor': 'white', 'boxstyle': 'round'})
        else:
            ax.text(3.0, 3.5, f'Cycle {cycle:.0f}', bbox={'facecolor': 'white', 'boxstyle': 'round'})
        ax.set_xlabel(r'Voltage [$\unit{\volt}$]')
        ax.set_ylabel(r'Incremental Capacity $\frac{\mathrm{d}Q}{\mathrm{d}V}$ [$\unit{{\ampere\hour}\per\volt}$]')
        # ax.set_ylabel(r'R\textsubscript{10} [$\unit{\milli\ohm}$]')
        cell_nbrs_as_strings = '_'.join([f'{k}' for k in rpt_cases.keys()])
        ica_name = os.path.join(ica_op, f'ICA_{plot_style}_cells_{cell_nbrs_as_strings}_{soh_string}.png')
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        ax.set_ylim((ymin, 9.5))
        ax.set_xlim((2.8, xmax))
        ax.legend()
        fig.savefig(ica_name, dpi=400)
        fig.savefig(ica_name.replace('.png', '.pdf'))
    plt.close('all')


    ########################################################################################################################
    ################################# ICI CROSS-COMPARISONS, PUT CASE ABBRV NAME IN LEGEND  ################################
    ########################################################################################################################
    ici_op = r"\\file00.chalmers.se\home\krifren\Provning\Analysis\pulse_charge\low_f\ici_plots"
    os.makedirs(ici_op, exist_ok=True)
    plot_style = 'same_soh'
    if plot_style == 'same_soh':
        rpt_cases = {
            181: [9, 10, 11],
            187: [9, 10, 11],
        }
        #rpt_cases = {
        #    190: [1, 4, 11],  # [6, 8],
        #    192: [1, 4, 11],
        #    180: [1, 3, 7],  # [10, 14]
        #    186: [1, 3, 4],
        #}
    else:
        rpt_cases = {
            190: np.arange(1, 12, 1),  # [6, 8],
            192: np.arange(1, 12, 1),  # [10, 14]
            180: np.arange(1, 12, 1),
            186: np.arange(1, 12, 1),
        }

    max_rpt_length = max(len(rptlist) for rptlist in rpt_cases.values())

    for rpt_index in range(max_rpt_length):
        fig, ax = plt.subplots(1, 1)
        soh_string = 'soh'
        soh_list = []
        for cell, rptlist in rpt_cases.items():
            ch_id = cell_to_channel_mapper[cell]
            # Check if the current cell has enough rpt entries to plot
            if rpt_index < len(rptlist):
                rpt = f'rpt_{rptlist[rpt_index]}'
                tmpdf = merged_data.ageing_data[ch_id].ici_data[rpt].ici_result_df
                tmpdf = tmpdf[tmpdf.ici_mode == 'chrg']
                soh = merged_data.ageing_data[ch_id].rpt_data.loc[rpt, 'cap_relative']
                cycle = merged_data.ageing_data[ch_id].rpt_data.loc[rpt, 'fce']
                temp_style = style_class.get_abbrv_style(merged_data.ageing_data[ch_id].TEST_NAME).copy()
                temp_style.pop('linestyle')
                soh_string = soh_string + f'_{soh * 100:.0f}'
                soh_list.append(soh)
                ax.scatter(tmpdf.maxV, tmpdf.k_mohm, **temp_style)
        soh_avg = np.array(soh_list).mean()
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        ax.set_ylim((ymin, 8.5))
        y_txt = ymin + 0.3
        x_txt = xmin + 0.03
        if plot_style == 'same_soh':
            ax.text(x_txt, y_txt, f'SoH {soh_avg * 100:.0f}\%', bbox={'facecolor': 'white', 'boxstyle': 'round'})
        else:
            ax.text(x_txt, y_txt, f'Cycle {cycle:.0f}', bbox={'facecolor': 'white', 'boxstyle': 'round'})
        ax.set_xlabel(r'Voltage [$\unit{\volt}$]')
        ax.set_ylabel(r'Diffusion resistance $k$ [$\unit{\milli\ohm\per\sqrt\second}$]')
        cell_nbrs_as_strings = '_'.join([f'{k}' for k in rpt_cases.keys()])
        ici_name = os.path.join(ici_op, f'ICI_k_mohm_{plot_style}_cells_{cell_nbrs_as_strings}_{soh_string}.png')
        ax.legend(loc='upper right')
        fig.savefig(ici_name, dpi=400)
        fig.savefig(ici_name.replace('.png', '.pdf'))
    plt.close('all')

    ########################################################################################################################
    ################################# DVA CROSS-COMPARISONS, PUT CASE ABBRV NAME IN LEGEND  ################################
    ########################################################################################################################
    dva_op = r"\\file00.chalmers.se\home\krifren\Provning\Analysis\pulse_charge\low_f\dva_plots"
    dva_case = 'chrg'
    os.makedirs(dva_op, exist_ok=True)
    cell_to_channel_mapper = {age_data.meta_data.cell_id: ch for ch, age_data in merged_data.ageing_data.items()}
    plot_style = 'same_soh'
    if plot_style == 'same_soh':
        rpt_cases = {
            140: [1, 4, 11],  # [6, 8],
            # 142: [1, 4, 8],
            150: [1, 4, 7],  # [10, 14]
            155: [1, 4, 5],
            156: [1, 4, 5]
        }
        # rpt_cases = {
        #     156: [1, 4, 5],  # [6, 8],
        #     142: [1, 4, 8],
        #     150: [1, 4, 7],  # [10, 14]
        #     155: [1, 4, 5],
        # }
    else:
        rpt_cases = {
            140: np.arange(1, 12, 1),
            # 142: [1, 4, 8],
            150: np.arange(1, 12, 1),
            155: np.arange(1, 12, 1),
            156: np.arange(1, 12, 1),
        }
        # rpt_cases = {
        #     156: np.arange(1, 12, 1),  # [6, 8],
        #     142: np.arange(1, 12, 1),  # [10, 14]
        #     150: np.arange(1, 12, 1),
        #     155: np.arange(1, 12, 1),
        # }

    max_rpt_length = max(len(rptlist) for rptlist in rpt_cases.values())

    for rpt_index in range(max_rpt_length):
        fig, ax = plt.subplots(1, 1)
        bol_tmp = bol_rpt.copy()
        if dva_case == 'chrg':
            bol_tmp = bol_tmp[bol_tmp.curr > 0].iloc[::4]
            bol_tmp['dva_flt'] = gaussian_filter1d(np.gradient(bol_tmp.volt, bol_tmp.cap / 1000), sigma=4,
                                                   mode='nearest')
        else:
            bol_tmp = bol_tmp[bol_tmp.curr < 0].iloc[::4]
            bol_tmp['dva_flt'] = gaussian_filter1d(np.gradient(bol_tmp.volt, bol_tmp.cap / 1000), sigma=4,
                                                   mode='nearest')
        ax.plot(bol_tmp.cap / 1000, bol_tmp.dva_flt, color='forestgreen', label='BoL ref')
        # ax.plot(bol_rpt_pos.cap, bol_rpt_pos.dva_gauss, color='forestgreen', label='BoL Ref')
        soh_string = 'soh'
        soh_list = []
        for cell, rptlist in rpt_cases.items():
            ch_id = cell_to_channel_mapper[cell]
            # Check if the current cell has enough rpt entries to plot
            if rpt_index < len(rptlist):
                rpt = f'rpt_{rptlist[rpt_index]}'
                tmpdf = merged_data.ageing_data[ch_id].ica_data[rpt].copy()
                if dva_case == 'chrg':
                    tmpdf = tmpdf[tmpdf.curr > 0]
                else:
                    tmpdf = tmpdf[tmpdf.curr < 0]
                tmpdf = tmpdf.iloc[::4]
                tmpdf['dva_flt'] = gaussian_filter1d(np.gradient(tmpdf.volt, tmpdf.cap / 1000),
                                                     sigma=4, mode='nearest')
                soh = merged_data.ageing_data[ch_id].rpt_data.loc[rpt, 'cap_relative']
                cycle = merged_data.ageing_data[ch_id].rpt_data.loc[rpt, 'fce']
                temp_style = style_class.get_abbrv_style(merged_data.ageing_data[ch_id].TEST_NAME).copy()
                temp_style.pop('linestyle')
                temp_style.pop('marker')
                soh_string = soh_string + f'_{soh * 100:.0f}'
                soh_list.append(soh)
                # tmpdf = tmpdf.reset_index()
                # peak_idx = argrelextrema(tmpdf.dva_flt.to_numpy(), np.greater, order=15)
                # ax.scatter(tmpdf.loc[peak_idx, 'cap'] / 1000, tmpdf.loc[peak_idx, 'dva_flt'], color='black')
                # print(f'Cap diff between peaks 1 and 2 is {tmpdf.loc[peak_idx, "cap"].diff().iloc[1] / 1000 :.2f} Ah '
                #       f'for cell {cell} at SoH {soh * 100:.0f}')
                ax.plot(tmpdf.cap / 1000, tmpdf.dva_flt, **temp_style)
        soh_avg = np.array(soh_list).mean()
        if dva_case == 'chrg':
            x_txt, y_txt = 0.5, 1.2
            ax.set_ylim((0, 1.8))
        else:
            x_txt, y_txt = 0.5, -0.8
            ax.set_ylim((-2, 0))
        if plot_style == 'same_soh':
            ax.text(x_txt, y_txt, f'SoH {soh_avg * 100:.0f}\%', bbox={'facecolor': 'white', 'boxstyle': 'round'})
        else:
            ax.text(x_txt, y_txt, f'Cycle {cycle:.0f}', bbox={'facecolor': 'white', 'boxstyle': 'round'})
        ax.set_xlabel(r'Capacity [$\unit{{\ampere\hour}}$]')
        ax.set_ylabel(r'Differential Voltage $\frac{\mathrm{d}V}{\mathrm{d}Q}$ [$\unit{\volt\per{\ampere\hour}}$]')
        # ax.set_ylabel(r'R\textsubscript{10} [$\unit{\milli\ohm}$]')
        cell_nbrs_as_strings = '_'.join([f'{k}' for k in rpt_cases.keys()])
        dva_name = os.path.join(dva_op, f'DVA_{dva_case}_{plot_style}_cells_{cell_nbrs_as_strings}_{soh_string}.png')
        ax.legend()
        fig.savefig(dva_name, dpi=400)
        fig.savefig(dva_name.replace('.png', '.pdf'))
    plt.close('all')

    dva_peaks = {}
    for ch, age_data in merged_data.ageing_data.items():
        # tmp_fig = plt.figure()
        cell = age_data.meta_data.cell_id
        for rpt, df in age_data.ica_data.items():
            cycle = age_data.rpt_data.loc[rpt, 'fce']
            abs_cap = age_data.rpt_data.loc[rpt, 'cap']
            soh = age_data.rpt_data.loc[rpt, 'cap_relative']
            if not df.empty:
                # tmp_fig = plt.figure()
                dvadf = df.copy()
                dvadf = dvadf.iloc[::4]
                dvadf = dvadf[dvadf.curr > 0]
                dvadf['dva_flt'] = gaussian_filter1d(np.gradient(dvadf.volt, dvadf.cap / 1000),
                                                     sigma=4, mode='nearest')
                dvadf = dvadf.reset_index()
                peak_idx = argrelextrema(dvadf.dva_flt.to_numpy(), np.greater, order=15)
                plt.plot(dvadf.cap, dvadf.dva_flt)
                plt.scatter(dvadf.loc[peak_idx, 'cap'], dvadf.loc[peak_idx, 'dva_flt'], color='red')
                identifier = f'Cell{cell:.0f}_cycle{cycle}_soh{soh * 100:.0f}'
                peak_idx = np.append(peak_idx[0], dvadf.index[-1])
                dva_peaks[identifier] = dvadf.loc[peak_idx]
                op_name = os.path.join(r"\\file00.chalmers.se\home\krifren\Provning\Analysis\pulse_charge\FirstBatch_natureformat\bulk_dva", f'{identifier}_filter4.png')
                # op_name = os.path.join(r"\\file00.chalmers.se\home\krifren\Provning\Analysis\pulse_charge\FirstBatch_natureformat\bulk_dva", f'All_dva_{cell}_filter3.png')
                # plt.ylim((0, 2))
                # tmp_fig.savefig(op_name, dpi=200)
                # plt.close(tmp_fig)

    gr_peak1_rng = [3.47, 3.53]
    gr_peak2_rng = [3.77, 3.85]
    nmc_peak_rng = [4.0, 4.05]
    fc_peak_rng = [4.19, 4.22]
    for k, df in dva_peaks.items():
        df['peak1_mask'] = df.volt.between(*gr_peak1_rng)
        df['peak2_mask'] = df.volt.between(*gr_peak2_rng)
        df['nmc_peak_mask'] = df.volt.between(*nmc_peak_rng)
        df['fc_peak_mask'] = df.volt.between(*fc_peak_rng)
    # delta_cap = {k: df[df.peak2_mask].cap.iloc[0] - df[df.peak1_mask].cap.iloc[0]
    #              for k, df in dva_peaks.items() if not df.empty}
    delta_cap = {}
    for k, df in dva_peaks.items():
        if not df.empty:
            try:
                delta_cap[k] = [df[df.peak1_mask].cap.iloc[0],
                                df[df.peak2_mask].cap.iloc[0],
                                df[df.nmc_peak_mask].cap.iloc[0],
                                df[df.fc_peak_mask].cap.iloc[0]]
            except IndexError as e:
                print(f'Index error for {k}')
    rows = []
    for key, values in delta_cap.items():
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
    dcap_df = pd.DataFrame(rows)  #.set_index('Cell')
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
    palette = sns.color_palette("tab10", n_colors=dcap_df['cell'].nunique())
    for col in norm_cols:
        # Create a dictionary to store results
        fit_results = []
        fig = plt.figure(figsize=(x_fig, x_fig))
        # plt.title(col)
        # Perform linear regression for each cell and store results
        for i, (cell, group) in enumerate(dcap_df.groupby('cell')):
            # RETRIEVE STYLE FOR THIS CELL
            cell_nbr = int(re.search(r'\d+', cell).group())
            ch_id = cell_to_channel_mapper[cell_nbr]
            style = style_class.get_abbrv_style(merged_data.ageing_data[ch_id].TEST_NAME)
            scat_style = style.copy()
            scat_style.pop('linestyle')

            group = group.dropna(subset=['soh', col])
            slope, intercept, r_value, p_value, std_err = stats.linregress(group['soh'], group[col])
            fit_results.append({'cell': cell, 'slope': slope, 'PCC': r_value, 'R^2': r_value ** 2})
            # Scatter plot of actual data points
            plt.scatter(group['soh'], group[col], **scat_style)

            # Generate fit line (over the same range as actual data)
            # soh_range = np.linspace(0.5, 1.1, 10)
            soh_range = np.linspace(group['soh'].min() / 1.15, group['soh'].max() * 1.15, 5)
            fit_line = intercept + slope * soh_range
            plt.plot(soh_range, fit_line, **style)
        soh_range = np.linspace(0.4, 1.25, 10)
        ax = plt.gca()
        lines, labels = ax.get_legend_handles_labels()
        unique_lines = {label: line for line, label in zip(lines, labels)}
        ax.legend(unique_lines.values(), unique_lines.keys())
        plt.plot(soh_range, soh_range, color='black', linestyle='--')
        plt.xlabel('SoH [-]')
        plt.ylabel(f'{plot_name_dict[col]} [-]')
        cap_col_name = os.path.join(dcap_op, f'{col}_with_trendline_and_fits_square.png')
        fig.savefig(cap_col_name, dpi=400)
        fig.savefig(cap_col_name.replace('.png', '.pdf'))
        # Convert results to a DataFrame
        fit_df = pd.DataFrame(fit_results)
        dcap_fit_dict[col] = fit_df



    # Set to keep track of already plotted test names
    plotted_cases = set()
    plt.figure()
    plt.plot(bol_rpt_pos.cap, bol_rpt_pos.dva_gauss, color='forestgreen', label='BoL Ref')
    for cell, age_data in cycle_data_batch1.ageing_data.items():
        if age_data.TEST_NAME in cases_to_plot and age_data.TEST_NAME not in plotted_cases:
            style_dict = style_class.get_abbrv_style(age_data.TEST_NAME)
            plotted_cases.add(age_data.TEST_NAME)
            df = age_data.ica_data['rpt_11']
            plt.plot(df[df.curr > 0].cap, df[df.curr > 0].dva_gauss,
                    color=style_dict['color'], label=style_dict['label'])
    plt.ylim((0, 1.5))
    plt.xlabel(r'Charge capacity $\left[\unit{{\milli\ampere\hour}}\right]$')
    plt.ylabel(r'Differential voltage $\frac{\mathrm{d}V}{\mathrm{d}Q}$ '
               r'$\left[\unit[per-mode=fraction]{\volt\per{\milli\ampere\hour}}\right]$')
    plt.text(2500, 1.0, 'EoL DVA ',
             fontdict={'fontsize': 7, 'family': 'serif'},
             bbox={'facecolor': 'white', 'boxstyle': 'round'})
    plt.legend(ncol=2)
    plt.savefig(os.path.join(output_dir, 'EoL_DVA_1Hz_and_ref.png'), dpi=400)

    temperature_analysis = 1
    if temperature_analysis:
        avg_temperatures = {}
        col_list = []
        for age_data in cycle_data_batch1.ageing_data.values():
            age_data.make_temperature_summary()
            avg_temperatures[age_data.TEST_NAME] = age_data.average_temperature
        plt.figure()
        col_list = [style_class.get_abbrv_style(k)['color'] for k in avg_temperatures.keys()]
        plt.scatter(avg_temperatures.keys(), avg_temperatures.values(), color=col_list)

    # Perform t-test to see if difference between results are significant
    data_sets = {}
    for tname, df in merged_data.combined_data_arbitrary_replicates.items():
        cols = [col for col in df.columns if 'cap_relative' in col and 'sigma' not in col]
        abbrv_name = style_class.get_abbrv_style(tname)['label']
        data_sets[abbrv_name] = df[cols].iloc[-1]
    all_combs = permutations(data_sets.keys(), 2)

    ttest_results = {}
    for comb in all_combs:
        c1, c2 = comb
        if not c1 == c2:
            ttest_results[f'{c1}__{c2}'] = stats.ttest_rel(data_sets[c1], data_sets[c2]).pvalue

    prob_df = pd.DataFrame([{'SpecGroup1': k.split('__')[0], 'SpecGroup2': k.split('__')[1], 'pval': v}
                            for k, v in ttest_results.items()])
    pmatrix = prob_df.pivot(index='SpecGroup1', columns='SpecGroup2', values='pval')
    pmatrix = pmatrix.fillna(1)
    pmat_triu = np.triu(pmatrix)
    pmat_triu = pd.DataFrame(pmat_triu, index=pmatrix.index, columns=pmatrix.columns)


