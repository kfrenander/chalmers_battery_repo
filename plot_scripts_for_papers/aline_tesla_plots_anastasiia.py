import matplotlib.pyplot as plt
import pandas as pd
from rpt_data_analysis.ReadRptClass import OrganiseRpts
from test_data_analysis.ica_analysis import gaussianfilterint, remove_faulty_points
import datetime as dt
import os


def find_rpt_from_fce(dataset, cellname, fce):
    rpt_data = dataset.summary_dict[cellname].data
    rpt_idx = rpt_data.iloc[(rpt_data['fce'] - fce).abs().argsort()[:1]].index[0]
    return rpt_idx


def find_rpt_from_soh(dataset, cellname, soh):
    rpt_data = dataset.summary_dict[cellname].data
    rpt_idx = rpt_data.iloc[(rpt_data['cap_relative'] - soh).abs().argsort()[:1]].index[0]
    return rpt_idx


def filter_dva_data(df):
    dva_df = df[df.curr > 0]
    dva_mah = dva_df.mAh - dva_df.mAh.min()
    dva_soc = (dva_df.mAh - dva_df.mAh.min()) / (dva_df.mAh.max() - dva_df.mAh.min())
    return dva_df, dva_mah, dva_soc


def build_df_from_tmp_dict(tmp_dict, ageing_case='cycling'):
    for df in tmp_dict.values():
        upper_case = [k for k in df.columns if 'FCE' in k]
        lower_case = [k for k in df.columns if 'fce' in k]
        if lower_case:
            fce_col = 'fce'
        elif upper_case:
            fce_col = 'FCE'
        else:
            print('No FCE col found, breaking')
            return None
    if ageing_case == 'cycling':
        cols_to_include = ['cap', 'cap_relative', 'date', 'res_dchg_50_relative', 'egy_thrg', fce_col]
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


def name_for_peak_bool(peak_bol_int):
    if peak_bol_int:
        return 'w_peak_marks'
    else:
        return 'no_peak_marks'


if __name__ == '__main__':
    data_set_location = r"\\sol.ita.chalmers.se\groups\batt_lab_data\ALINE_data\50_dod_data"
    op_location = r"Z:\Provning\Analysis\ALINE_plots\large_soc\update_230413"
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

    color_upd_bool = 1
    if color_upd_bool:
        bol_color = '#799150'
    else:
        bol_color = '#99b070'


    plot_settings_dict = {
        'rpt_5': ['MoL', 'solid', 'limegreen'],
    #    'rpt_6': ['300 FCE', 'dashed', 'gold'],
    #    'rpt_9': ['600 FCE', 'solid', 'darkred'],
        'rpt_12': ['EoL', 'dashed', 'midnightblue']
    }

    cell_dict_init = {
        '644': ['0 to 50 SOC high temp_5_1', '0-50% SOC', '45$^\circ$C', 'H50', '#e87052', 0.2],
        '644_bk' : ['0 to 50 SOC high temp_5_3', '0-50% SOC', '45$^\circ$C', 'H50', '#e87052', 0.2],
        '648': ['0 to 100 SOC high temp_5_6', '0-100% SOC', '45$^\circ$C', 'H0100', '#e87052', 1],
        '648_bk': ['0 to 100 SOC high temp_5_2', '0-100% SOC', '45$^\circ$C', 'H0100', '#e87052', 1],
        '673': ['50 to 100 SOC room temp_4_6', '50-100% SOC', 'room temp', 'R100', '#264554', 0.6],
        '673_bk': ['50 to 100 SOC room temp_4_5', '50-100% SOC', 'room temp', 'R100', '#264554', 0.6],
        '643': ['50 to 100 SOC high temp_5_4', '50-100% SOC', '45$^\circ$C', 'H100', '#e87052', 0.6],
        '643_bk': ['50 to 100 SOC high temp_5_5', '50-100% SOC', '45$^\circ$C', 'H100', '#e87052', 0.6],
        '682': ['0 to 50 SOC room temp_4_7', '0-50% SOC', 'room temp', 'R50', '#264554', 0.2],
        '682_bk': ['0 to 50 SOC room temp_4_7', '0-50% SOC', 'room temp', 'R50', '#264554', 0.2],
        '685': ['0 to 100 SOC room temp_4_3', '0-100% SOC', 'room temp', 'R0100', '#264554', 1],
        '685_bk': ['0 to 100 SOC room temp_4_4', '0-100% SOC', 'room temp', 'R0100', '#264554', 1]
    }

    cell_dict_upd = {
        '644': ['0 to 50 SOC high temp_5_1', '0-50% SOC', '45$^\circ$C', 'H0-50', '#966B9D', 1],
        '644_bk' : ['0 to 50 SOC high temp_5_3', '0-50% SOC', '45$^\circ$C', 'H0-50', '#966B9D', 1],
        '648': ['0 to 100 SOC high temp_5_6', '0-100% SOC', '45$^\circ$C', 'H0-100', '#26021B', 1],
        '648_bk': ['0 to 100 SOC high temp_5_2', '0-100% SOC', '45$^\circ$C', 'H0-100', '#26021B', 1],
        '673': ['50 to 100 SOC room temp_4_6', '50-100% SOC', 'room temp', 'R50-100', '#1E7099', 1],
        '673_bk': ['50 to 100 SOC room temp_4_5', '50-100% SOC', 'room temp', 'R50-100', '#1E7099', 1],
        '643': ['50 to 100 SOC high temp_5_4', '50-100% SOC', '45$^\circ$C', 'H50-100', '#5C2751', 1],
        '643_bk': ['50 to 100 SOC high temp_5_5', '50-100% SOC', '45$^\circ$C', 'H50-100', '#5C2751', 1],
        '682': ['RPT_240119_4_7', '0-50% SOC', 'room temp', 'R0-50', '#6EA4BF', 1],
        '682_bk': ['0 to 50 SOC room temp_4_8', '0-50% SOC', 'room temp', 'R0-50', '#6EA4BF', 1],
        '685': ['0 to 100 SOC room temp_4_3', '0-100% SOC', 'room temp', 'R0-100', '#274654', 1],
        '685_bk': ['0 to 100 SOC room temp_4_4', '0-100% SOC', 'room temp', 'R0-100', '#274654', 1]
    }

    cell_list = ['644', '682', '648', '685', '643', '673']
    cell_subset = ['644', '682', '648', '685']

    if not os.path.isdir(op_location):
        os.mkdir(op_location)

    # Specify peaks
    yloc_neg = [15]*7
    yloc_pos = [18]*7
    x_coords = [3.32, 3.47, 3.587, 3.67, 3.865, 3.95, 4.115]
    peaks_pos = [str(x) for x in [4, 4, 3, 3, 2, 2, 1]]
    peaks_neg = [str(x) for x in [4, 3, 3, 2, 2, 1, 1]]
    peak_in_fig = 1


    # Reference figure for ICA with peak marking
    ref_case = '50 to 100 SOC high temp_5_4'
    bol_ref = data.ica_dict[ref_case]['rpt_1']
    bol_ref_fig, bol_ref_ax = plt.subplots(1, 1)
    bol_ref_ax.plot(bol_ref[bol_ref.curr > 0]['volt'],
                    bol_ref[bol_ref.curr > 0]['ica_gauss'],
                    linestyle='solid',
                    color=bol_color,
                    label='Fresh cell')
    bol_ref_ax.plot(bol_ref[bol_ref.curr < 0]['volt'],
                    bol_ref[bol_ref.curr < 0]['ica_gauss'],
                    linestyle='solid',
                    color=bol_color)
    box_props_neg = dict(boxstyle='circle', facecolor='black', alpha=0.9, edgecolor='black')
    box_props_pos = dict(boxstyle='circle', facecolor='white', alpha=1, edgecolor='black')
    plt.text(2.9, yloc_pos[0], 'NCA reaction', color='red', fontsize=16, weight='bold')
    plt.text(2.9, yloc_neg[0], 'Si-Gr reaction', color='blue', fontsize=16, weight='bold')
    plt.vlines(x_coords, 0, yloc_neg[0] - 1.5, color='black', linewidth=0.8)
    for p, x, y in zip(peaks_pos, x_coords, yloc_pos):
        plt.text(x, y, p, fontdict=mrk_font, bbox=box_props_pos, horizontalalignment='center')
    for p, x, y in zip(peaks_neg, x_coords, yloc_neg):
        plt.text(x, y, p, fontdict=mrk_font_neg, bbox=box_props_neg, horizontalalignment='center')
    bol_ref_ax.set_xlim(2.85, 4.2)
    bol_ref_ax.set_ylim(-15, 20)
    bol_ref_ax.grid(False)
    bol_ref_ax.set_xlabel('Cell Voltage, V', fontdict=lbl_font)
    bol_ref_ax.set_ylabel(r'IC dQ dV$^{-1}$, Ah V$^{-1}$', fontdict=lbl_font)
    bol_ref_fig.savefig(os.path.join(op_location, 'bol_reference_marked_reaction_updated_label.pdf'))
    bol_ref_fig.savefig(os.path.join(op_location, 'bol_reference_marked_reaction_updated_label.png'), dpi=400)


    for cell_to_analyse in cell_list:
        dva_ref = bol_ref[bol_ref.curr > 0]
        # cell_to_analyse = '644'
        # for cell_to_analyse in cell_dict_upd.keys():
        ica_fig, ica_ax = plt.subplots(1, 1)
        # Draw reference data
        ica_ax.plot(bol_ref.volt[bol_ref.curr > 0],
                    bol_ref.ica_gauss[bol_ref.curr > 0],
                    label='BoL', color=bol_color, alpha=1)
        ica_ax.plot(bol_ref.volt[bol_ref.curr < 0],
                    bol_ref.ica_gauss[bol_ref.curr < 0],
                    color=bol_color, alpha=1)
        # Mark peaks if wanted
        if peak_in_fig:
            for p, x, y in zip(peaks_pos, x_coords, yloc_pos):
                plt.text(x, y, p, fontdict=mrk_font, bbox=box_props_pos, horizontalalignment='center')
            for p, x, y in zip(peaks_neg, x_coords, yloc_neg):
                plt.text(x, y, p, fontdict=mrk_font_neg, bbox=box_props_neg, horizontalalignment='center')

        for rp in plot_settings_dict:
            ex = data.ica_dict[cell_dict_upd[cell_to_analyse][0]][rp]
            tmp_fce = data.summary_dict[cell_dict_upd[cell_to_analyse][0]].data.loc[rp, 'fce']
            ica_ax.plot(ex.volt[ex.curr > 0], ex.ica_gauss[ex.curr > 0],
                        label=f'{plot_settings_dict[rp][0]} {cell_dict_upd[cell_to_analyse][3]}',
                        linestyle=plot_settings_dict[rp][1],
                        color=cell_dict_upd[cell_to_analyse][4],
                        alpha=cell_dict_upd[cell_to_analyse][5])
            ica_ax.plot(ex.volt[ex.curr < 0], ex.ica_gauss[ex.curr < 0],
                        linestyle=plot_settings_dict[rp][1],
                        color=cell_dict_upd[cell_to_analyse][4],
                        alpha=cell_dict_upd[cell_to_analyse][5])
        ica_ax.legend(loc='lower left')
        ica_ax.set_xlim(2.9, 4.2)
        ica_ax.set_xlabel('Voltage, V', fontdict=lbl_font)
        ica_ax.set_ylabel(r'IC dQ dV$^{-1}$, Ah V$^{-1}$', fontdict=lbl_font)
        if peak_in_fig:
            ica_ax.set_ylim(-15, 20)
        else:
            ica_ax.set_ylim(-20, 15)
        ica_ax.grid(False)
        ica_fig.savefig(os.path.join(op_location, f'ICA_cell{cell_to_analyse}_{name_for_peak_bool(peak_in_fig)}_labelud_no_grid.png'),
                        dpi=400)
        ica_fig.savefig(os.path.join(op_location, f'ICA_cell{cell_to_analyse}_{name_for_peak_bool(peak_in_fig)}_labelud_no_grid.pdf'),
                        dpi=400)
        plt.close(ica_fig)

    dva_cf, dva_ca = plt.subplots(1, 1)
    dva_sf, dva_sa = plt.subplots(1, 1)
    ref_mah = bol_ref[bol_ref.curr > 0].mAh - bol_ref[bol_ref.curr > 0].mAh.min()
    ref_soc = (bol_ref[bol_ref.curr > 0].mAh - bol_ref[bol_ref.curr > 0].mAh.min()) / \
                (bol_ref[bol_ref.curr > 0].mAh.max() - bol_ref[bol_ref.curr > 0].mAh.min())
    dva_ca.plot(ref_mah, bol_ref[bol_ref.curr > 0].dva_gauss,
                label='BoL', color=bol_color, linestyle='dashed')
    dva_sa.plot(ref_soc, bol_ref[bol_ref.curr > 0].dva_gauss,
                label='BoL', color=bol_color, linestyle='dashed')

    for rp in plot_settings_dict:
        # ex = data.ica_dict['50 to 100 SOC room temp_4_6'][rp]
        ex = data.ica_dict[cell_dict_upd[cell_to_analyse][0]][rp]
        ex = ex[ex.curr > 0]
        tmp_fce = data.summary_dict[cell_dict_upd[cell_to_analyse][0]].data.loc[rp, 'fce']
        x_mah = ex.mAh - ex.mAh.min()
        x_soc = (ex.mAh - ex.mAh.min()) / (ex.mAh.max() - ex.mAh.min())
        dva_ca.plot(x_mah, ex.dva_gauss,
                    label=f'{cell_dict_upd[cell_to_analyse][1]} {tmp_fce} FCE {cell_dict_upd[cell_to_analyse][2]}',
                    linestyle=plot_settings_dict[rp][1],
                    color=plot_settings_dict[rp][2])
        dva_sa.plot(x_soc, ex.dva_gauss,
                    label=f'{cell_dict_upd[cell_to_analyse][1]} {tmp_fce} FCE {cell_dict_upd[cell_to_analyse][2]}',
                    linestyle=plot_settings_dict[rp][1],
                    color=plot_settings_dict[rp][2])
    for tmp_ax in [dva_sa, dva_ca]:
        tmp_ax.legend(loc='lower left')
        tmp_ax.set_ylabel('Differential Voltage, V Ah$^{-1}$')
        tmp_ax.set_ylim(0, 0.5)
    dva_sa.set_xlabel('SOC, -')
    dva_ca.set_xlabel('Q, mAh')
    dva_cf.savefig(os.path.join(op_location, f'DVA_cap_cell{cell_to_analyse}.png'), dpi=400)
    dva_sf.savefig(os.path.join(op_location, f'DVA_soc_cell{cell_to_analyse}.png'), dpi=400)
    dva_cf.savefig(os.path.join(op_location, f'DVA_cap_cell{cell_to_analyse}.pdf'), dpi=400)
    dva_sf.savefig(os.path.join(op_location, f'DVA_soc_cell{cell_to_analyse}.pdf'), dpi=400)

    q_fig, qax = plt.subplots(1, 1)
    for c in cell_subset:
        if 'bk' not in c:
            qdf = data.summary_dict[cell_dict_upd[c][0]].data
            qax.plot(qdf.fce, qdf.cap_relative*100,
                     label=f'{cell_dict_upd[c][3]}',
                     marker='s',
                     color=cell_dict_upd[c][4],
                     alpha=cell_dict_upd[c][5])
    qax.legend(loc='upper right')
    qax.set_xlabel('Number of Full Cycle Equivalents, FCE', fontdict=lbl_font)
    qax.set_ylabel('Percentage of Capacity Retained, %', fontdict=lbl_font)
    qax.set_ylim(70, 102)
    q_fig.savefig(os.path.join(op_location, f'CapacityRetention_no_errbar_subset_labelud_{dt.datetime.now():%Y-%m_%d}.png'), dpi=400)
    q_fig.savefig(os.path.join(op_location, f'CapacityRetention_no_errbar_subset_labelud_{dt.datetime.now():%Y-%m_%d}.pdf'), dpi=400)

    qe_fig, qe_ax = plt.subplots(1, 1)
    mol_mark = 1
    all_cell_dict = {}
    for c in cell_list:
        tmp_q_dict = {}
        for c_name in cell_dict_upd:
            if c in c_name:
                tmp_q_dict[c_name] = data.summary_dict[cell_dict_upd[c_name][0]].data
        qdf = build_df_from_tmp_dict(tmp_q_dict)
        all_cell_dict[c] = qdf
        qe_ax.errorbar(qdf.loc[:, f'fce_{c}'], qdf['y_mean']*100, qdf['y_err'].abs()*100,
                    color=cell_dict_upd[c][4],
                    alpha=cell_dict_upd[c][5],
                    elinewidth=1.5,
                    marker='s',
                    capsize=6,
                    label=cell_dict_upd[c][3])
    #qe_ax.grid(False)
    qe_ax.legend(loc='upper right')
    qe_ax.set_xlabel('Number of Full Cycle Equivalents, FCE', fontdict=lbl_font)
    qe_ax.set_ylabel('Percentage of Capacity Retained, %', fontdict=lbl_font)
    qe_ax.set_ylim(70, 102)
    if mol_mark:
        plt.vlines(300, 74, 96.5, linewidth=1.2, color='red')
        plt.text(250, 98, 'MoL',
                 color='red',
                 fontdict=mrk_font,
                 bbox=dict(boxstyle='square', facecolor='white', alpha=0.9, edgecolor='black'))
    qe_fig.savefig(os.path.join(op_location, f'CapacityRetention_w_errorbar_and_mol_full_labelud_{dt.datetime.now():%Y-%m_%d}.png'), dpi=400)
    qe_fig.savefig(os.path.join(op_location, f'CapacityRetention_w_errorbar_and_mol_full_labelud_{dt.datetime.now():%Y-%m_%d}.pdf'), dpi=400)

    cross_fig, cr_ax = plt.subplots(1, 1)
    cross_fig_dva, cr_dva = plt.subplots(1, 1)
    comb_fig_soc, cr_soc = plt.subplots(1, 1)
    dva_ref, ref_mah, ref_soc = filter_dva_data(bol_ref)
    cell1 = '673'
    cell2 = '685'
    d1 = data.ica_dict[cell_dict_upd[cell1][0]]
    d2 = data.ica_dict[cell_dict_upd[cell2][0]]
    fce_list = [150, 900]
    rpt_list1 = [find_rpt_from_fce(data, cell_dict_upd[cell1][0], f) for f in fce_list]
    rpt_list2 = [find_rpt_from_fce(data, cell_dict_upd[cell2][0], f) for f in fce_list]
    style_list = ['solid', (0, (1, 1))]
    c_list = ['lime', 'crimson']
    st_dict1 = dict(zip(rpt_list1, style_list))
    st_dict2 = dict(zip(rpt_list2, style_list))
    c_dict = dict(zip([cell1, cell2], c_list))
    cr_ax.plot(bol_ref.volt, bol_ref.ica_gauss,
               label='BoL', color=bol_color, linestyle='dashed')
    cr_dva.plot(ref_mah, dva_ref.dva_gauss,
                label='BoL', color=bol_color, linestyle='dashed')
    cr_soc.plot(ref_soc, dva_ref.dva_gauss,
                label='BoL', color=bol_color, linestyle='dashed')
    for i, rp in enumerate(rpt_list1):
        rp2 = rpt_list2[i]
        dva1, xmah1, xsoc1 = filter_dva_data(d1[rp])
        dva2, xmah2, xsoc2 = filter_dva_data(d2[rp2])
        cr_ax.plot(d1[rp].volt, d1[rp].ica_gauss,
                   label=f'{cell_dict_upd[cell1][1]} {data.summary_dict[cell_dict_upd[cell1][0]].data.loc[rp, "fce"]}'
                         f' FCE {cell_dict_upd[cell1][2]}',
                   linestyle=st_dict1[rp],
                   color=c_dict[cell1])
        cr_dva.plot(xmah1, dva1.dva_gauss,
                    label=f'{cell_dict_upd[cell1][1]} {data.summary_dict[cell_dict_upd[cell1][0]].data.loc[rp, "fce"]}'
                          f' FCE {cell_dict_upd[cell1][2]}',
                    linestyle=st_dict1[rp],
                    color=c_dict[cell1])
        cr_soc.plot(xsoc1, dva1.dva_gauss,
                    label=f'{cell_dict_upd[cell1][1]} {data.summary_dict[cell_dict_upd[cell1][0]].data.loc[rp, "fce"]}'
                          f' FCE {cell_dict_upd[cell1][2]}',
                    linestyle=st_dict1[rp],
                    color=c_dict[cell1])
        cr_ax.plot(d2[rp2].volt, d2[rp2].ica_gauss,
                   label=f'{cell_dict_upd[cell2][1]} {data.summary_dict[cell_dict_upd[cell2][0]].data.loc[rp2, "fce"]} '
                         f'FCE {cell_dict_upd[cell2][2]}',
                   linestyle=st_dict2[rp2],
                   color=c_dict[cell2])
        cr_dva.plot(xmah2, dva2.dva_gauss,
                    label=f'{cell_dict_upd[cell2][1]} {data.summary_dict[cell_dict_upd[cell1][0]].data.loc[rp, "fce"]}'
                          f' FCE {cell_dict_upd[cell2][2]}',
                    linestyle=st_dict2[rp2],
                    color=c_dict[cell2])
        cr_soc.plot(xsoc2, dva2.dva_gauss,
                    label=f'{cell_dict_upd[cell2][1]} {data.summary_dict[cell_dict_upd[cell1][0]].data.loc[rp, "fce"]}'
                          f' FCE {cell_dict_upd[cell2][2]}',
                    linestyle=st_dict2[rp2],
                    color=c_dict[cell2])
    cr_ax.legend(loc='lower left')
    cr_ax.set_xlim(2.9, 4.25)
    cr_ax.set_xlabel('Voltage, V')
    cr_ax.set_ylabel('Differential Capacity, Ah V$^{-1}$')
    cr_ax.set_ylim(-20, 15)
    cross_fig.savefig(os.path.join(op_location, f'ICA_cross_correlation_cell{cell1}_and_{cell2}.png'), dpi=400)
    cross_fig.savefig(os.path.join(op_location, f'ICA_cross_correlation_cell{cell1}_and_{cell2}.pdf'), dpi=400)

    cr_dva.set_xlabel('Q, mAh')
    cr_dva.legend(loc='lower left')
    cr_dva.set_ylabel('Differential Voltage, V Ah$^{-1}$')
    cr_dva.set_ylim(0, 0.5)
    cross_fig_dva.savefig(os.path.join(op_location, f'DVA_cap_cross_correlation_cell{cell1}_and_{cell2}.pdf'), dpi=400)
    cross_fig_dva.savefig(os.path.join(op_location, f'DVA_cap_cross_correlation_cell{cell1}_and_{cell2}.png'), dpi=400)

    cr_soc.set_xlabel('SOC, -')
    cr_soc.legend(loc='lower left')
    cr_soc.set_ylabel('Differential Voltage, V Ah$^{-1}$')
    cr_soc.set_ylim(0, 0.5)
    cross_fig_dva.savefig(os.path.join(op_location, f'DVA_soc_cross_correlation_cell{cell1}_and_{cell2}.pdf'), dpi=400)
    cross_fig_dva.savefig(os.path.join(op_location, f'DVA_soc_cross_correlation_cell{cell1}_and_{cell2}.png'), dpi=400)


    cross_ica_fig, cr_ica_ax = plt.subplots(1, 1)
    cr_ica_ax.plot(bol_ref[bol_ref.curr > 0].volt, bol_ref[bol_ref.curr > 0].ica_gauss,
                   label='BoL', color=bol_color, linestyle='dashed')
    cell1 = '673'
    cell2 = '685'
    cell3 = '682'

    d1 = data.ica_dict[cell_dict_upd[cell1][0]]
    d2 = data.ica_dict[cell_dict_upd[cell2][0]]
    d3 = data.ica_dict[cell_dict_upd[cell3][0]]

    soh = 0.9
    rpt673 = find_rpt_from_soh(data, cell_dict_upd[cell1][0], soh)
    rpt685 = find_rpt_from_soh(data, cell_dict_upd[cell2][0], soh)
    rpt3 = find_rpt_from_soh(data, cell_dict_upd[cell3][0], soh)
    cr_ica_ax.plot(d1[rpt673][d1[rpt673].curr > 0].volt, d1[rpt673][d1[rpt673].curr > 0].ica_gauss,
                   label=f'{cell_dict_upd[cell1][1]}, SOH {soh*100}%')
    cr_ica_ax.plot(d2[rpt685][d2[rpt685].curr > 0].volt, d2[rpt685][d2[rpt685].curr > 0].ica_gauss,
                   label=f'{cell_dict_upd[cell2][1]}, SOH {soh*100}%')
    cr_ica_ax.plot(d3[rpt3][d3[rpt3].curr > 0].volt, d3[rpt3][d3[rpt3].curr > 0].ica_gauss,
                   label=f'{cell_dict_upd[cell3][1]}, SOH {soh*100}%')
    cr_ica_ax.legend(loc='upper left')

