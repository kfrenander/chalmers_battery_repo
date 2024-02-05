from PythonScripts.test_data_analysis.read_neware_file import read_neware_xls
import matplotlib.pyplot as plt
from PythonScripts.test_data_analysis.rpt_analysis import characterise_steps, find_step_characteristics
from PythonScripts.test_data_analysis.three_electrode_ici import read_neware_v80
from PythonScripts.test_data_analysis.ica_analysis import perform_ica
import numpy as np
import pandas as pd
import os
from PythonScripts.backend_fix import fix_mpl_backend
fix_mpl_backend()
# plt.style.use('chalmers_kf')


def find_neighbours(value, df, colname):
    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        return exactmatch.index
    else:
        lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
        upperneighbour_ind = df[df[colname] > value][colname].idxmin()
        return [lowerneighbour_ind, upperneighbour_ind]


fig_save_dir = r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\init_test\analysis"
raw_files_1 = {
    "094": r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\init_test\240119-2-1-365.xls",
    "095": r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\init_test\240119-2-2-365.xls",
    "096": r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\init_test\240119-2-3-365.xls"
}
raw_files_2 = {
    "094": r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\init_test\240119-2-1-368.xls",
    "095": r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\init_test\240119-2-2-368.xls"
}
cell_screening_files = {
    "3-1": r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\ScreeningTests\240119-3-2-368.xlsx",
    "3-2": r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\ScreeningTests\240119-3-5-368.xlsx",
    "3-3": r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\ScreeningTests\240119-3-1-368.xlsx"
}

init_rpt = [r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\cyc_test_init\240095-1-5-2818575192_1..xlsx",
            r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\cyc_test_init\240095-1-5-2818575192_2..xlsx",
            r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\cyc_test_init\240095-1-5-2818575192.xlsx"]

style_dict = {
    "094": ["*", "dashed", "orange"],
    "095": [".", "solid", "forestgreen"],
    "096": ["^", "dotted", "grey"]
}

raw_data = {k: read_neware_xls(raw_files_2[k], calc_c_rate=True) for k in raw_files_2}
stp_data = {k: characterise_steps(raw_data[k]) for k in raw_data}
stp_data = {k: find_step_characteristics(raw_data[k]) for k in raw_data}
# screening_data = {k: read_neware_v80(cell_screening_files[k]) for k in cell_screening_files}
rpt_data = [read_neware_xls(k) for k in init_rpt]
rpt_df = pd.concat(rpt_data, ignore_index=False)
rpt_df.sort_values('Measurement', inplace=True)

fig, ax = plt.subplots(1, 1)
for c in stp_data:
    tmp_df = stp_data[c]
    dchg_stp = tmp_df[tmp_df.step_mode == 'CC_DChg']
    ax.scatter(dchg_stp.curr.abs() / 3.4, dchg_stp.cap / 3410, label=f'Cell {c}',
               marker=style_dict[c][0], color=style_dict[c][2], s=200)
ax.legend(loc='upper left')
ax.set_xlabel('Crate [-]')
ax.set_ylabel('Nomalised discharge capacity [-]')
my_x_ticks = ax.set_xticks([0.1, 0.33, 0.5, 1, 1.5, 1.75])
my_y_ticks = ax.set_yticks(np.linspace(0.95, 1.05, 11))
ax.grid(False)
# fig.savefig(os.path.join(fig_save_dir, 'c_rate_scatter_norm.png'), dpi=400)

f_all, ax2 = plt.subplots(1, 1)
f_1c, ax1c = plt.subplots(1, 1)
for c in raw_data:
    gb_mode = raw_data[c].groupby(by=['mode'])
    dchg_ = gb_mode.get_group('CC_DChg')
    gb_crate = dchg_.groupby(by='c_rate')
    dct = {k: gb_crate.get_group(k) for k in gb_crate.groups}
    for k in dct:
        ax2.plot(dct[k].cap, dct[k].volt, label=f'Cell {c}, C rate {k:.1f}', linestyle=style_dict[c][1])
        if 0.9 < k < 1.1:
            soc50 = dct[k].cap.max() / 2
            lnb, unb = find_neighbours(soc50, dct[k], 'cap')
            ax1c.plot(dct[k].cap, dct[k].volt, label=f'Cell {c}, C rate {k:.1f}', linestyle=style_dict[c][1])
            ax1c.axvline(dct[k].loc[lnb, 'cap'], color=style_dict[c][2], linestyle=style_dict[c][1],
                         label='50% cap voltage')
            ax1c.axhline(dct[k].loc[lnb, 'volt'], color=style_dict[c][2], linestyle=style_dict[c][1])
            print(f'50% cap voltage for 1c discharge for cell {c} is {dct[k].loc[lnb, "volt"]:.2f}')
ax2.legend(loc='lower left')
ax2.set_xlabel('Discharged capacity [mAh]')
ax2.set_ylabel('Cell Voltage [V]')
# f_all.savefig(os.path.join(fig_save_dir, 'all_crates_dchg.png'), dpi=400)

ax1c.legend(loc='lower left')
ax1c.set_xlabel('Discharged capacity [mAh]')
ax1c.set_ylabel('Cell Voltage [V]')
# f_1c.savefig(os.path.join(fig_save_dir, '1C_dchg_w_volt_markers.png'), dpi=400)


f_ch, axch = plt.subplots(1, 1)
f_ch50, axch50 = plt.subplots(1, 1)
for c in raw_data:
    gb_mode = raw_data[c].groupby(by='mode')
    chrg_ = gb_mode.get_group('CCCV_Chg')
    gb_stps = chrg_.groupby(by='arb_step2')
    stp_dct = {k: gb_stps.get_group(k) for k in gb_stps.groups}
    for k in stp_dct:
        if stp_dct[k].volt.min() < 3.3:
            axch.plot(stp_dct[k].cap, stp_dct[k].volt, label=f'Cell {c}, step {k}',
                      linestyle=style_dict[c][1])
        if k==10:
            soc50 = stp_dct[k].cap.max() / 2
            lch, uch = find_neighbours(soc50, stp_dct[k], 'cap')
            soc100_v_lim_idx = stp_dct[k][abs(stp_dct[k].volt - 4.195) < 2e-3].first_valid_index()
            soc50_v_lim = stp_dct[k].loc[soc100_v_lim_idx, 'cap'] / 2
            v_lim_l, v_lim_u = find_neighbours(soc50_v_lim, stp_dct[k], 'cap')
            axch50.plot(stp_dct[k].cap, stp_dct[k].volt, label=f'Cell {c}',
                        linestyle=style_dict[c][1], color=style_dict[c][2])
            axch50.axvline(stp_dct[k].loc[lch, 'cap'], color=style_dict[c][2], linestyle=style_dict[c][1])
            axch50.axhline(stp_dct[k].loc[lch, 'volt'], color=style_dict[c][2], linestyle=style_dict[c][1],
                           label=f'50% cap voltage, cell {c}')
            axch50.axvline(stp_dct[k].loc[v_lim_l, 'cap'], linestyle=style_dict[c][1])
            axch50.axhline(stp_dct[k].loc[v_lim_l, 'volt'], linestyle=style_dict[c][1],
                           label=f'50% vlim cap voltage, cell {c}')
            print(f'50% cap voltage for .5C charge and cell {c} is {stp_dct[k].loc[lch, "volt"]:.2f}')
            print(f'50% vlim cap voltage for .5C charge and cell {c} is {stp_dct[k].loc[v_lim_l, "volt"]:.2f}')
axch.legend(loc='lower right')
axch.set_xlabel('Charged capacity [mAh]')
axch.set_ylabel('Cell Voltage [V]')
# f_ch.savefig(os.path.join(fig_save_dir, 'all_cccv_charges.png'), dpi=400)

axch50.legend(loc='lower right')
axch50.set_xlabel('Charged capacity [mAh]')
axch50.set_ylabel('Cell Voltage [V]')
# f_ch50.savefig(os.path.join(fig_save_dir, 'chrg_05c_w_dual_volt_marker.png'), dpi=400)

## Find voltages for pulses at 70, 50 and 30% SOC during C/3 discharge.
soc_lvls = [0.3, 0.5, 0.7]
vlt_dct = {}
for c_nbr, df in raw_data.items():
    c3 = df[abs(df['c_rate'] - 0.33) < 0.04]
    caps = [soc*c3.cap.abs().max() for soc in soc_lvls]
    idx_lst = [find_neighbours(cap, c3, 'cap') for cap in caps]
    vlt_lst = [c3.loc[idx[0]:idx[1], 'volt'].mean() for idx in idx_lst]
    vlt_dct[c_nbr] = vlt_lst
