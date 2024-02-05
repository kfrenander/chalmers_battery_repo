import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
from PythonScripts.rpt_data_analysis.ReadRptClass import OrganiseRpts, look_up_fce
#from PythonScripts.test_data_analysis.rpt_analysis import characterise_steps
from PythonScripts.test_data_analysis.ica_analysis import perform_ica
import os
from scipy.signal import find_peaks
from PythonScripts.backend_fix import fix_mpl_backend

#plt.style.use('seaborn-poster')

my_lrg_font = 16
my_med_font = 14
my_sml_font = 12

x_width = 8
aspect_rat = 12 / 16
plt.rcParams['figure.figsize'] = x_width, aspect_rat * x_width
plt.rcParams['legend.fontsize'] = my_med_font
plt.rcParams['axes.labelsize'] = my_lrg_font
plt.rcParams['axes.titlesize'] = my_lrg_font
plt.rcParams['axes.grid'] = False
plt.rcParams['lines.linewidth'] = 1.7
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
# plt.rcParams["text.usetex"] = True
lbl_font = {'weight': 'normal',
            'size': my_lrg_font}
plt.rc('legend', fontsize=my_sml_font)
mark_size = 5
cap_size = 6
plt.rc('font', **{"family": 'sans-serif', 'sans-serif': 'Helvetica'})

mrk_font = {'weight': 'normal',
            'size': my_sml_font}
new_fig_folder = r"Z:\Documents\Papers\TeslaLowFrequencyPulse\updated_images"


def ica_from_raw_rpt(df):
    import PythonScripts.test_data_analysis.rpt_analysis as rpt_analysis
    step_df = rpt_analysis.characterise_steps(df)
    ica = df[df.arb_step2.isin(step_df[step_df.step_duration > 10 * 3600]['step_nbr'])]
    ica = perform_ica(ica)
    return ica


def kelly_contrast_colors(i):
    kelly_colors_hex = [
        '#E68FAC',
        '#222222',
        '#F3C300',
        '#875692',
        '#F38400',
        '#A1CAF1',
        '#BE0032',
        '#C2B280',
        '#848482',
        '#008856'
    ]
    return kelly_colors_hex[i]


def replace_zero(inp_str):
    try:
        fce_num = re.search(r'\d+ ', inp_str).group()
        if int(fce_num) == 0:
            return f'Fresh cell'
        else:
            return inp_str
    except AttributeError:
        return inp_str


def replace_label(ax_inp):
    for ln in ax_inp.lines:
        ln.set_label(replace_zero(ln.get_label()))
    ax_inp.legend()
    return ax_inp


def look_up_color(test_dur):
    color_dict = {
        1: 'maroon',
        2: 'forestgreen',
        4: 'darkorange',
        8: 'mediumblue',
        16: 'crimson',
        32: 'chartreuse',
        64: 'darkviolet',
        128: 'black',
        256: 'indianred'
    }
    return color_dict[test_dur]


def calc_soc(df):
    df.loc[:, 'soc'] = (df.mAh - df.mAh.min()) / (df.mAh.max() - df.mAh.min())
    return df


def yield_dataset(fin, stp, dataset, operation=False, cell='128s_2_8', start=1):
    data_pts = [f'rpt_{i}' for i in range(start, fin, stp)]
    if operation == 'chrg':
        subset_of_data = {k: calc_soc(dataset.ica_dict[cell][k][dataset.ica_dict[cell][k]['curr'] > 0])
                          for k in data_pts}
    elif operation == 'dchg':
        subset_of_data = {k: calc_soc(dataset.ica_dict[cell][k][dataset.ica_dict[cell][k]['curr'] < 0])
                          for k in data_pts}
    else:
        subset_of_data = {k: dataset.ica_dict[cell][k] for k in data_pts}
    return subset_of_data


def find_peak_coords(df, md='chrg'):
    if md == 'chrg':
        peak_idx, peak_prop = find_peaks(df.dva_gauss, height=0.100, distance=30)
    elif md == 'dchg':
        peak_idx, peak_prop = find_peaks(df.dva_gauss.abs(), height=0.100, distance=30)
    df_idx = df.reset_index().loc[peak_idx, 'index']
    peak_data = df.loc[df_idx, :]
    if md == 'chrg':
        peak_data.loc[:, 'peak_id'] = [identify_charge_peak_from_voltage(peak_data.loc[k, 'volt'])
                                       for k in peak_data.index]
    elif md == 'dchg':
        peak_data.loc[:, 'peak_id'] = [identify_discharge_peak_from_voltage(peak_data.loc[k, 'volt'])
                                       for k in peak_data.index]
    return peak_data


def identify_charge_peak_from_voltage(peak_volt):
    if 3.3 <= peak_volt <= 3.6:
        return 'Si-Gr_mix1'
    elif 3.7 <= peak_volt <= 3.8:
        return 'NCA_H1M'
    elif 3.82 <= peak_volt <= 3.98:
        return 'Gr_central'
    elif 4 <= peak_volt <= 4.1:
        return 'NCA_MH2'
    else:
        return 'unknown peak'


def identify_discharge_peak_from_voltage(peak_volt):
    if 3.1 <= peak_volt <= 3.5:
        return 'Si-Gr_mix1'
    elif 3.6 <= peak_volt <= 3.8:
        return 'NCA_H1M'
    elif 3.82 <= peak_volt <= 3.94:
        return 'Gr_central'
    elif 3.96 <= peak_volt <= 4.1:
        return 'NCA_MH2'
    else:
        return 'unknown peak'


def calculate_electrode_ageing_proxies(pk_dict):
    nca_h1m_mh2 = []
    si_gr_central = []
    si_gr_mix = []
    for k in pk_dict:
        tmp = pk_dict[k]
        try:
            cap_h1m = tmp[tmp['peak_id'] == 'NCA_H1M'].reset_index().loc[0, 'cap']
            cap_mh2 = tmp[tmp['peak_id'] == 'NCA_MH2'].reset_index().loc[0, 'cap']
            nca_h1m_mh2.append(abs(cap_mh2 - cap_h1m))
        except KeyError:
            print(f'NCA peaks not identified for {k}. Will leave empty')
            nca_h1m_mh2.append(np.nan)
        try:
            cap_gr_si_cntrl = tmp[tmp['peak_id'] == 'Gr_central'].reset_index().loc[0, 'cap']
            cap_gr_si_mix = tmp[tmp['peak_id'] == 'Si-Gr_mix1'].reset_index().loc[0, 'cap']
            si_gr_central.append(cap_gr_si_cntrl)
            si_gr_mix.append(cap_gr_si_mix)
        except KeyError:
            print(f'Central graphite peak not identified for {k}. Will leave empty')
            si_gr_central.append(np.nan)
    fce = [look_up_fce(k) for k in pk_dict]
    return pd.DataFrame([[k for k in pk_dict], fce, si_gr_central, si_gr_mix, nca_h1m_mh2],
                        index=['rpt', 'fce', 'si_gr_proxy', 'si_gr_mix_proxy', 'nca_proxy']).T


def plot_dva_for_data_set(sub_data):
    fig, ax = plt.subplots(2, 1)
    for k in sub_data:
        l = ax[0].plot(sub_data[k].cap, sub_data[k].dva_gauss, label=k, linewidth=0.75)
        act_col = l[0].get_color()
        # ax[0].scatter(pks[k].cap, pks[k].dva_gauss, color=act_col, marker='.')
        ax[1].plot(sub_data[k].soc, sub_data[k].dva_gauss, label=k, color=act_col, linewidth=0.75)
        # ax[1].scatter(pks[k].soc, pks[k].dva_gauss, color=act_col, marker='*')
    ax[0].set_ylim(0, 0.5)
    ax[0].set_xlabel('Capacity [mAh]')
    ax[1].set_ylim(0, 0.5)
    ax[1].set_xlabel('SOC [-]')
    ax[0].legend()
    ax[1].legend()
    return fig

def plot_fun_bda(data_dict, fig_name, save_bool=1):
    fig_dir = r"\\sol.ita.chalmers.se\groups\batt_lab_data\analysis_directory\TeslaPulseAgeingPaper"
    hyst_fig, h_ax = plt.subplots(1, 1)
    ica_fig, i_ax = plt.subplots(1, 1)
    dva_fig, d_ax = plt.subplots(1, 1)
    vfig, v_ax = plt.subplots(1, 1)
    h_ax.grid(True)
    i_ax.grid(True)
    d_ax.grid(True)
    for i, key in enumerate(data_dict):
        tmp_ica = data_dict[key]
        soc = (tmp_ica.mAh - tmp_ica.mAh.min()) / (tmp_ica.mAh.max() - tmp_ica.mAh.min())
        tmp_ica.loc[:, 'soc'] = soc
        u_int_chrg = interp1d(tmp_ica[tmp_ica.curr > 0].soc, tmp_ica[tmp_ica.curr > 0].volt)
        u_int_dchg = interp1d(tmp_ica[tmp_ica.curr < 0].soc, tmp_ica[tmp_ica.curr < 0].volt)
        x_low = max(tmp_ica[tmp_ica.curr > 0].soc.min(), tmp_ica[tmp_ica.curr < 0].soc.min())
        x_hi = min(tmp_ica[tmp_ica.curr > 0].soc.max(), tmp_ica[tmp_ica.curr < 0].soc.max())
        x_int = np.linspace(x_low, x_hi, 400)
        # ica_ch = tmp_ica[tmp_ica.curr > 0]
        # ica_dc = tmp_ica[tmp_ica.curr < 0]
        i_ax.plot(tmp_ica[tmp_ica.curr > 0].volt, tmp_ica[tmp_ica.curr > 0].ica_gauss,
                  label=f'{fig_name} {look_up_fce(key)} FCE',
                  linewidth=1.5,
                  color=kelly_contrast_colors(i))
        i_ax.plot(tmp_ica[tmp_ica.curr < 0].volt, tmp_ica[tmp_ica.curr < 0].ica_gauss,
                  linewidth=1.5,
                  color=kelly_contrast_colors(i))
        h_ax.plot(x_int * 100, u_int_chrg(x_int) - u_int_dchg(x_int),
                  label=f'{fig_name} {look_up_fce(key)} FCE',
                  linewidth=1.5,
                  color=kelly_contrast_colors(i))
        # d_ax.plot(soc, tmp_ica.dva_gauss,
        #           label=f'{fig_name} {look_up_fce(key)} FCE',
        #           linewidth=1.5,
        #           color=kelly_contrast_colors(i))
        d_ax.plot(tmp_ica[tmp_ica.curr > 0].cap, tmp_ica[tmp_ica.curr > 0].dva_gauss,
                  label=f'{fig_name} {look_up_fce(key)} FCE',
                  linewidth=1.5,
                  color=kelly_contrast_colors(i))
        v_ax.plot(soc, tmp_ica.volt, linewidth=0.85)
    h_ax.set_xlabel('SOC / %')
    h_ax.set_ylabel('$\Delta U$ / V')
    h_ax.lines[0].set_label('Fresh cell')
    h_ax.set_ylim(0, 0.28)
    h_ax.legend()
    h_ax.grid(color='grey', alpha=0.5)
    i_ax.set_xlabel('Voltage / V')
    i_ax.set_ylabel('IC / Ah V$^{-1}$')
    i_ax.lines[0].set_label('Fresh cell')
    i_ax.set_ylim(-15, 15)
    i_ax.legend(loc='upper left')
    i_ax.grid(color='grey', alpha=0.5)
    d_ax.set_xlabel('Capacity / mAh')
    d_ax.set_ylabel('DV / V Ah$^{-1}$')
    d_ax.set_ylim(0, 0.5)
    d_ax.legend()
    d_ax.grid(color='grey', alpha=0.5)
    v_ax.set_xlabel('SOC / %')
    v_ax.set_ylabel('Voltage / V')
    v_ax.lines[0].set_label('Fresh cell')
    v_ax.legend()
    op_dict = {
        'hyst_fig': hyst_fig,
        'ica_fig': ica_fig,
        'dva_fig': dva_fig,
        'volt_fig': vfig
    }
    if save_bool:
        dva_fig.savefig(os.path.join(fig_dir, f'{fig_name}_dva_chrg.png'), dpi=dva_fig.dpi)
        dva_fig.savefig(os.path.join(fig_dir, f'{fig_name}_dva_chrg.pdf'), dpi=dva_fig.dpi)
        ica_fig.savefig(os.path.join(fig_dir, '{}_ica_bol_mol_eol.pdf'.format(fig_name)),
                        dpi=hyst_fig.dpi)
        ica_fig.savefig(os.path.join(fig_dir, '{}_ica_bol_mol_eol.png'.format(fig_name)),
                        dpi=1200)
        hyst_fig.savefig(os.path.join(fig_dir, '{}_hysteresis_bol_mol_eol_dU.pdf'.format(fig_name)),
                         dpi=hyst_fig.dpi)
        hyst_fig.savefig(os.path.join(fig_dir, '{}_hysteresis_bol_mol_eol_dU.png'.format(fig_name)),
                         dpi=hyst_fig.dpi)
        vfig.savefig(os.path.join(fig_dir, '{}_voltage_updated_FCE_new_format.pdf'.format(fig_name)),
                     dpi=hyst_fig.dpi)
        # volt_fig.savefig(os.path.join(fig_dir, '{}_voltage_updated_FCE_new_format.png'.format(fig_name)),
        #                  dpi=hyst_fig.dpi)
    return op_dict


if __name__ == '__main__':
    fix_mpl_backend()
    update_fig = 0
    data_dir = r"\\sol.ita.chalmers.se\groups\batt_lab_data\20200923"
    hc_data = r"\\sol.ita.chalmers.se\groups\batt_lab_data\analysis_directory\TeslaPulseAgeingPaper\half_cell_processed_data"
    data_set = OrganiseRpts(data_dir, proj='BDA')
    hc_data_set = {k.split('.')[0]: pd.read_pickle(os.path.join(hc_data, k)) for k in os.listdir(hc_data)}
    # for test_cond in data_set.ica_dict:
    #     for rpt in data_set.ica_dict[test_cond]:
    #         data_set.ica_dict[test_cond][rpt] = ica_from_raw_rpt(data_set.rpt_raw_dict[test_cond][rpt] )
    data_pts = ['rpt_1', 'rpt_10', 'rpt_19']
    data_pts = [f'rpt_{i}' for i in range(1, 11, 2)]
    # plt_data_2s = {k: data_set.ica_dict['2s_1_4'][k] for k in data_pts}
    # plot_fun_bda(plt_data_2s, '2s pulse')
    plt_data_8s = {k: data_set.ica_dict['8s_1_8'][k] for k in data_pts}
    # plot_fun_bda(plt_data_8s, '8s pulse')
    plt_data_128s = {k: data_set.ica_dict['128s_2_8'][k] for k in data_pts}
    # plot_fun_bda(plt_data_128s, '128s pulse')
    data_set_hysteresis = yield_dataset(20, 9, dataset=data_set, cell='128s_2_8', start=1)
    plot_fun_bda(data_set_hysteresis, '128s_pulse_for_hysteresis', save_bool=0)
    bol_data = yield_dataset(2, 1, dataset=data_set)

    hyst_fig, h_ax = plt.subplots(1, 1, figsize=(8, 6))
    volt_fig, v_ax = plt.subplots(1, 1, figsize=(8, 6))
    gauss_fig, g_ax = plt.subplots(1, 1, figsize=(8, 6))
    dva_fig, d_ax = plt.subplots(1, 1, figsize=(8, 6))
    comb_data = {
        # '8s rpt_1': data_set.ica_dict['8s_1_8']['rpt_1'],
        # '8s rpt_17': data_set.ica_dict['8s_1_8']['rpt_17'],
        '128s rpt_1': data_set.ica_dict['128s_2_8']['rpt_1'],
        '128s rpt_3': data_set.ica_dict['128s_2_8']['rpt_3'],
        '128s rpt_5': data_set.ica_dict['128s_2_8']['rpt_5'],
        '128s rpt_17': data_set.ica_dict['128s_2_8']['rpt_17'],
    }
    for key in comb_data:
        tmp_ica = comb_data[key]
        soc_lvls = re.findall(r'\d+s', key)
        rpt_str = re.search(r'rpt_\d+', key).group()
        soc = (tmp_ica.mAh - tmp_ica.mAh.min()) / (tmp_ica.mAh.max() - tmp_ica.mAh.min())
        tmp_ica.loc[:, 'soc'] = soc
        u_int_chrg = interp1d(tmp_ica[tmp_ica.curr > 0].soc, tmp_ica[tmp_ica.curr > 0].volt)
        u_int_dchg = interp1d(tmp_ica[tmp_ica.curr < 0].soc, tmp_ica[tmp_ica.curr < 0].volt)
        x_low = max(tmp_ica[tmp_ica.curr > 0].soc.min(), tmp_ica[tmp_ica.curr < 0].soc.min())
        x_hi = min(tmp_ica[tmp_ica.curr > 0].soc.max(), tmp_ica[tmp_ica.curr < 0].soc.max())
        x_int = np.linspace(x_low, x_hi, 400)
        v_ax.plot(tmp_ica.soc * 100, tmp_ica.volt,
                  label=f'{soc_lvls[0]} {look_up_fce(rpt_str)} FCE',
                  linewidth=0.85)
        h_ax.plot(x_int * 100, u_int_chrg(x_int) - u_int_dchg(x_int),
                  label=f'{soc_lvls[0]} {look_up_fce(rpt_str)} FCE',
                  linewidth=0.85)
        g_ax.plot(tmp_ica.volt, tmp_ica.ica_gauss,
                  label=f'{soc_lvls[0]} {look_up_fce(rpt_str)} FCE',
                  linewidth=0.85)
        d_ax.plot(tmp_ica.soc, tmp_ica.dva_gauss,
                  label=f'{soc_lvls[0]} {look_up_fce(rpt_str)} FCE',
                  linewidth=0.85)
    v_ax.set_xlabel('SOC [%]', weight='bold')
    v_ax.set_ylabel('Voltage [V]', weight='bold')
    # v_ax.lines[0].set_label('Fresh cell')
    v_ax.legend()
    h_ax.set_xlabel('SOC [%]', weight='bold')
    h_ax.set_ylabel('Voltage hysteresis [V]', weight='bold')
    h_ax.lines[0].set_label('Fresh cell')
    h_ax.legend()
    h_ax.grid(color='grey', alpha=0.5)
    g_ax.set_xlabel('Voltage [V]', weight='bold')
    g_ax.set_ylabel('Incremental capacity dQ/dV [Ah/V]', weight='bold')
    g_ax.lines[0].set_label('Fresh cell')
    g_ax.legend()
    d_ax.set_xlabel('SOC [%]', weight='bold')
    d_ax.set_ylabel('Differential voltage dV/dQ [V/Ah]', weight='bold')
    d_ax.set_ylim(0, 0.5)
    replace_label(d_ax)
    d_ax.legend()

    xaxis_soc = 0
    init_data = data_set.ica_dict['128s_2_8']['rpt_1']
    init_data = init_data[init_data.curr > 0]
    pk_idx, pk_prop = find_peaks(init_data.dva_gauss * 1000, height=200)
    pk_coords = [init_data.reset_index().loc[pk_idx, 'soc'], pk_prop['peak_heights']]
    neg_df = hc_data_set['neg_lithiation']
    pos_df = hc_data_set['pos_lithiation']
    init_fig, init_ax = plt.subplots(1, 1)
    if xaxis_soc:
        neg_scale = 1.17
        pos_scale = 1.17
        pos_of = -0.13
        neg_of = 0
        init_ax.plot(neg_df.soc * neg_scale + neg_of, -neg_df.dva, label='Anode', linestyle='dashed',
                     color=kelly_contrast_colors(0))
        init_ax.plot(pos_df.soc * pos_scale + pos_of, -pos_df.dva, label='Cathode', linestyle='dashed',
                     color=kelly_contrast_colors(1))
        init_ax.plot(init_data.soc, -init_data.dva_gauss * 1000, label='Full cell',
                     color=kelly_contrast_colors(2))
        init_ax.scatter(pk_coords[0], pk_coords[1])
        init_ax.set_xlabel('SOC / -')
        init_ax.set_ylim(0, -500)
        init_ax.set_xlim(0, 1)
        dva_comb_fig_name = 'half_cell_and_fc_dva_soc.pdf'
    else:
        neg_scale = 680
        pos_scale = 640
        pos_of = -790
        neg_of = 75
        init_ax.plot(neg_df.cap.max()*1000*neg_scale - (neg_df.cap*1000 * neg_scale + neg_of), -neg_df.dva,
                     label='Negative electrode', linestyle='dashed', color=kelly_contrast_colors(0))
        init_ax.plot(pos_df.cap*1000 * pos_scale + pos_of, -pos_df.dva, label='Positive electrode', linestyle='dashed',
                     color=kelly_contrast_colors(1))
        init_ax.plot(init_data.cap, -init_data.dva_gauss * 1000, label='Full cell',
                     color=kelly_contrast_colors(2))
        #init_ax.scatter(pk_coords[0], pk_coords[1])
        init_ax.set_xlabel('Full cell capacity / mAh')
        init_ax.set_ylim(0, -500)
        init_ax.set_xlim(0, 4500)
        dva_comb_fig_name = 'half_cell_and_fc_dva_cap.pdf'
    init_ax.set_ylabel('dV/dQ / V mAh$^{-1}$')
    init_ax.legend()
    init_ax.grid(color='grey', alpha=0.5)
    fig_dir = r"\\sol.ita.chalmers.se\groups\batt_lab_data\analysis_directory\TeslaPulseAgeingPaper"
    if update_fig:
        init_fig.savefig(os.path.join(fig_dir, 'freshcell_hc_fc_dva_dchg_soc.pdf'))
        init_fig.savefig(os.path.join(new_fig_folder, dva_comb_fig_name))

    selected_data_pts = yield_dataset(20, 1, dataset=data_set, operation='chrg', cell='128s_2_8', start=1)
    pks = {k: find_peak_coords(selected_data_pts[k], md='chrg') for k in selected_data_pts}
    age_prx_df = calculate_electrode_ageing_proxies(pks)
    age_prx_clean = age_prx_df.dropna(axis=0)
    age_prx_df.loc[9, 'si_gr_proxy'] = np.nan
    prxfig, prax = plt.subplots(1, 1)
    prax.plot(age_prx_df.fce, age_prx_df.si_gr_proxy, linestyle='dashed', marker='*',
              color=kelly_contrast_colors(1),
              label='Negative electrode ageing proxy')
    prax.plot(age_prx_df.fce, age_prx_df.nca_proxy, linestyle='dashed', marker='^',
              color=kelly_contrast_colors(0),
              label='Positive electrode ageing proxy')
    prax.set_xlabel('Full Cycle Equivalents / -')
    prax.set_ylabel('Ageing proxy / mAh')
    prax.set_ylim((0, 3000))
    prax.legend()
    prax.grid(color='grey', alpha=0.5)
    #prax.set_xlim((-20, 220))
    if update_fig:
        prxfig.savefig(os.path.join(new_fig_folder, 'ageing_proxies.pdf'), dpi=300)
    
    group_fig, gr_ax = plt.subplots(1, 1)
    for k in selected_data_pts:
        if look_up_fce(k) % 50 == 0:
            l = gr_ax.plot(selected_data_pts[k].soc, selected_data_pts[k].dva_gauss, label=f'FCE {look_up_fce(k)}')
            loop_col = l[0].get_color()
            gr_ax.scatter(pks[k].soc, pks[k].dva_gauss, color=loop_col, marker='^')
    gr_ax.set_ylim(0, 0.5)


    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    for k in selected_data_pts:
        if look_up_fce(k) % 100 == 0:
            l = ax[0].plot(selected_data_pts[k].cap, selected_data_pts[k].dva_gauss, label=f'FCE {look_up_fce(k)}',
                           linewidth=0.75)
            act_col = l[0].get_color()
            ax[0].scatter(pks[k].cap, pks[k].dva_gauss, color=act_col, marker='.')
            ax[1].plot(selected_data_pts[k].soc, selected_data_pts[k].dva_gauss, label=f'FCE {look_up_fce(k)}',
                       color=act_col, linewidth=0.75)
            ax[1].scatter(pks[k].soc, pks[k].dva_gauss, color=act_col, marker='*')
    ax[0].set_ylim(0, 0.5)
    ax[0].set_xlabel('Capacity [mAh]')
    ax[1].set_ylim(0, 0.5)
    ax[1].set_xlabel('SOC [-]')
    ax[0].legend()
    ax[1].legend()
    fig.text(0.04, 0.5, 'DVA [mAh/V]', va='center', rotation='vertical', fontsize=16)
    if update_fig:
        fig.savefig(os.path.join(fig_dir, '128s_2_7_allrpt_dva_capandsoc_newstyle.png'))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for k in selected_data_pts:
        l = ax.plot(selected_data_pts[k].cap, selected_data_pts[k].dva_gauss, label=k, linewidth=0.75)
        act_col = l[0].get_color()
        ax.scatter(pks[k].cap, pks[k].dva_gauss, color=act_col, marker='.')
    ax.set_ylim(0, 0.5)
    ax.set_xlabel('Capacity [mAh]')
    ax.set_ylabel('DVA [Ah/V]')
    ax.legend()
    if update_fig:
        fig.savefig(os.path.join(fig_dir, '128s_2_8_rpt1_to_7_dva_cap_new_style.png'))
