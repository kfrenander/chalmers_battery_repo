from rpt_data_analysis.post_process_cycling_data import CycleAgeingDataIndexer
import matplotlib.pyplot as plt
from natsort import natsorted
import re
import os


plot_style = 'large_scale'
if plot_style == 'large_scale':
    x_width = 8
    aspect_rat = 12 / 16
    plt.rcParams['figure.figsize'] = x_width, aspect_rat * x_width
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['lines.linewidth'] = 1.7
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    lbl_font = {'weight': 'normal',
                'size': 18}
    plt.rc('legend', fontsize=14)
    peak_mark_size = 15
    mark_size = 5
    cap_size = 6
elif plot_style == 'double_col':
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
plt.rcParams['axes.grid'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'
output_dir = r'Z:\Provning\Analysis\pulse_charge'


def calculate_fce_from_rpt(rpt_str):
    rpt_num = int(re.findall(r'\d+', rpt_str)[0])
    fce_num = (rpt_num - 1) * 40
    return fce_num


if __name__ == '__main__':
    data_loc = r"\\sol.ita.chalmers.se\groups\batt_lab_data\pulse_chrg_test\cycling_data"
    cycle_data = CycleAgeingDataIndexer()
    cycle_data.run(data_loc)

    fig, ax = plt.subplots(1, 1)
    for k, ag_data in cycle_data.ageing_data.items():
        tmp_df = ag_data.rpt_data
        ax.plot(tmp_df.fce, tmp_df.cap_relative,
                color=ag_data.visual_profile.color,
                marker=ag_data.visual_profile.marker,
                label=ag_data.test_name)
    plt.legend(ncols=2)
    ax.set_xlabel('FCE [-]')
    ax.set_ylabel('Capacity retention [-]')

    fig, ax = plt.subplots(1, 1)
    for t_name, cmb_dat in cycle_data.combined_data.items():
        ax.plot(cmb_dat.fce, cmb_dat.avg_rel_cap,
                color=cycle_data.visual_profile.colors[t_name],
                marker=cycle_data.visual_profile.markers[t_name],
                label=t_name)
    plt.legend(ncols=2)
    ax.set_xlabel('FCE [-]')
    ax.set_ylabel('Capacity retention [-]')

    fig, ax = plt.subplots(1, 1)
    for t_name, cmb_dat in cycle_data.combined_data.items():
        ax.errorbar(cmb_dat.fce, cmb_dat.avg_rel_cap,
                    yerr=cmb_dat.std_rel_cap,
                    elinewidth=1.5,
                    color=cycle_data.visual_profile.colors[t_name],
                    marker=cycle_data.visual_profile.markers[t_name],
                    label=t_name,
                    markersize=mark_size,
                    capsize=cap_size)
    plt.legend(ncols=2)
    ax.set_xlabel('FCE [-]')
    ax.set_ylabel('Capacity retention [-]')
    fig.savefig(os.path.join(output_dir, 'error_bar_all_cells.png'), dpi=400)

    ref_cases_dchg_pulsing = ['1000mHz', '1000mHz_no_pulse',
                              '100mHz', '100mHz_no_pulse']
    fig, ax = plt.subplots(1, 1)
    for t_name, cmb_dat in cycle_data.combined_data.items():
        if t_name in ref_cases_dchg_pulsing:
            ax.errorbar(cmb_dat.fce, cmb_dat.avg_rel_cap,
                        yerr=cmb_dat.std_rel_cap,
                        elinewidth=1.5,
                        color=cycle_data.visual_profile.colors[t_name],
                        marker=cycle_data.visual_profile.markers[t_name],
                        label=t_name,
                        markersize=mark_size,
                        capsize=cap_size)
    plt.legend(ncols=2)
    ax.set_xlabel('FCE [-]')
    ax.set_ylabel('Capacity retention [-]')
    fig.savefig(os.path.join(output_dir, 'error_bar_pulse_ref_cells.png'), dpi=400)

    fig, ax = plt.subplots(1, 1)
    for t_name, cmb_dat in cycle_data.combined_data.items():
        if '_no_pulse' not in t_name:
            ax.errorbar(cmb_dat.fce, cmb_dat.avg_rel_cap,
                        yerr=cmb_dat.std_rel_cap,
                        elinewidth=1.5,
                        color=cycle_data.visual_profile.colors[t_name],
                        marker=cycle_data.visual_profile.markers[t_name],
                        label=t_name,
                        markersize=mark_size,
                        capsize=cap_size)
    plt.legend(ncols=2)
    ax.set_xlabel('FCE [-]')
    ax.set_ylabel('Capacity retention [-]')
    fig.savefig(os.path.join(output_dir, 'error_bar_dchg_pulse_cells.png'), dpi=400)

    plt.style.use('kelly_colors')
    plt.figure()
    for rpt, df in natsorted(cycle_data.ageing_data['240095_3_1'].ica_data.items()):
        plt.plot(df[df.curr > 0].cap, df[df.curr > 0].dva_gauss, label=f'FCE {calculate_fce_from_rpt(rpt)}')
    plt.ylim((0, 5))
    plt.legend()
    plt.xlabel(r'Charge capacity $\left[\SI{}{\milli\ampere\hour}\right]$')
    plt.ylabel(r'Differential Voltage $\frac{dV}{dQ}$ '
               r'$\left[\unit[per-mode=fraction]{\volt\per\milli\ampere\per\hour}\right]$')

    plt.figure()
    for rpt, df in cycle_data.ageing_data['240095_3_1'].ica_data.items():
        plt.plot(df[df.curr > 0].volt, df[df.curr > 0].ica_gauss, label=f'FCE {calculate_fce_from_rpt(rpt)}')
    plt.xlabel(r'Voltage [$\SI{}{\volt}$]')
    plt.ylabel(r'Incremental Capacity $\frac{dQ}{dV}$ [$\SI{}{\milli\ampere\hour\per\volt}$]')
    plt.legend()

    plt.figure()
    for cell, age_data in cycle_data.ageing_data.items():
        df = age_data.ica_data['rpt_1']
        plt.plot(df[df.curr > 0].volt, df[df.curr > 0].ica_gauss, color=age_data.visual_profile.color)
    plt.xlabel(r'Voltage [$\SI{}{\volt}$]')
    plt.ylabel(r'Incremental Capacity $\frac{dQ}{dV}$ [$\SI{}{\milli\ampere\hour\per\volt}$]')

    cases_to_plot = ['CC reference', '1000mHz']
    plt.figure()
    for cell, age_data in cycle_data.ageing_data.items():
        if age_data.test_name in cases_to_plot:
            df = age_data.ica_data['rpt_11']
            plt.plot(df[df.curr > 0].volt, df[df.curr > 0].ica_gauss,
                     color=age_data.visual_profile.color,
                     label=age_data.test_name)
    plt.xlabel(r'Voltage $\left[\unit{\volt}\right]$')
    plt.ylabel(r'Incremental Capacity $\frac{dQ}{dV}$ $[\unit{\milli\ampere\hour\per\volt}]$')
    plt.text(3, 4.2, 'EoL ICA ',
             fontdict={'fontsize': 16, 'family': 'serif'},
             bbox={'facecolor': 'white', 'boxstyle': 'round'})
    plt.legend(ncol=2)
    plt.savefig(os.path.join(output_dir, 'EoL_ICA_1Hz_and_ref.png'), dpi=400)

    plt.figure()
    for cell, age_data in cycle_data.ageing_data.items():
        if age_data.test_name in cases_to_plot:
            df = age_data.ica_data['rpt_11']
            plt.plot(df[df.curr > 0].cap, df[df.curr > 0].dva_gauss,
                     color=age_data.visual_profile.color,
                     label=age_data.test_name)
    plt.ylim((0, 3))
    plt.xlabel(r'Charge capacity $\left[\unit{\milli\ampere\hour}\right]$')
    plt.ylabel(r'Differential Voltage $\frac{dV}{dQ}$ '
               r'$\left[\unit[per-mode=fraction]{\volt\per\milli\ampere\per\hour}\right]$')
    plt.text(2500, 2.0, 'EoL DVA ',
             fontdict={'fontsize': 16, 'family': 'serif'},
             bbox={'facecolor': 'white', 'boxstyle': 'round'})
    plt.legend(ncol=2)
    plt.savefig(os.path.join(output_dir, 'EoL_DVA_1Hz_and_ref.png'), dpi=400)

    temperature_analysis = 0
    if temperature_analysis:
        avg_temperatures = {}
        col_list = []
        for age_data in cycle_data.ageing_data.values():
            age_data.make_temperature_summary()
            avg_temperatures[age_data.test_name] = age_data.average_temperature
        plt.figure()
        col_list = [cycle_data.visual_profile.colors[k] for k in avg_temperatures.keys()]
        plt.scatter(avg_temperatures.keys(), avg_temperatures.values(), color=col_list)
