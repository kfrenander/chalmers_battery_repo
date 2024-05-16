from rpt_data_analysis.post_process_cycling_data import CycleAgeingDataIndexer
import matplotlib.pyplot as plt


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
    plt.rcParams['axes.grid'] = True
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

    plt.style.use('kelly_colors')
    plt.figure()
    for rpt, df in cycle_data.ageing_data['240095_3_1'].ica_data.items():
        plt.plot(df[df.curr > 0].cap, df[df.curr > 0].dva_gauss)
    plt.ylim((0, 5))

    plt.figure()
    for rpt, df in cycle_data.ageing_data['240095_3_1'].ica_data.items():
        plt.plot(df[df.curr > 0].volt, df[df.curr > 0].ica_gauss)

    plt.figure()
    for cell, age_data in cycle_data.ageing_data.items():
        df = age_data.ica_data['rpt_1']
        plt.plot(df[df.curr > 0].volt, df[df.curr > 0].ica_gauss, color=age_data.visual_profile.color)

    plt.figure()
    for cell, age_data in cycle_data.ageing_data.items():
        df = age_data.ica_data['rpt_11']
        plt.plot(df[df.curr > 0].volt, df[df.curr > 0].ica_gauss, color=age_data.visual_profile.color)

    plt.figure()
    for cell, age_data in cycle_data.ageing_data.items():
        df = age_data.ica_data['rpt_11']
        plt.plot(df[df.curr > 0].cap, df[df.curr > 0].dva_gauss, color=age_data.visual_profile.color)
    plt.ylim((0, 5))
