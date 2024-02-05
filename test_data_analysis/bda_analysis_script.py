import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from PythonScripts.rpt_data_analysis.ReadRptClass import OrganiseRpts
import re
import os
import matplotlib as mpl
from PIL import Image
from io import BytesIO
from PythonScripts.backend_fix import fix_mpl_backend
import natsort
plt.style.use('chalmers_kf')
# import statsmodels.api as sm
# from statsmodels.formula.api import ols

x_width = 8
aspect_rat = 3 / 4
plt.rcParams['figure.figsize'] = x_width, aspect_rat * x_width
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 1.7
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams["text.usetex"]
lbl_font = {'weight': 'normal',
            'size': 20}
plt.rc('legend', fontsize=14)
mark_size = 5
cap_size = 6
plt.rc('font', **{"family": 'sans-serif', 'sans-serif': 'Helvetica'})

def merge_data_sets(dataset1, dataset2):
    dataset1.summary_dict.update(dataset2.summary_dict)
    dataset1.ica_dict.update(dataset2.ica_dict)
    dataset1.rpt_raw_dict.update(dataset2.rpt_raw_dict)
    dataset1.eol_df = dataset1.eol_df.append(dataset2.eol_df)
    dataset1.name_df = dataset1.name_df.append(dataset2.name_df)
    dataset1.eol_df.sort_values(by='test_dur', inplace=True)
    return dataset1


def set_linestyle_bda_dataset(proj_name):
    if 'comp' in proj_name:
        return 'dashed'
    else:
        return 'solid'


def scatter_with_linear_fit(df, save_bool=False):
    fit_poly = np.polyfit(df['freq']*1000, df['eol'], deg=1)
    fit_fig, fax = plt.subplots(1, 1)
    fax.scatter(df['freq']*1000, df['eol'], label='Raw data', color='red', edgecolor='black')
    fax.plot(df['freq']*1000, np.polyval(fit_poly, df['freq']*1000), label='Linear fit', color='blue')
    fax.set_xlabel('Frequency, mHz')
    fax.set_ylabel('Energy throughput at EOL, kWh')
    poly_text = f'Linear fit:\n{fit_poly[1]:.2f} + {fit_poly[0]:.2f} * f'
    box_settings = {
        "boxstyle": "round",
        "fc": "white",
        "ec": "black",
        "pad": 0.3
    }
    plt.text(np.mean(fax.get_xlim()), np.polyval(fit_poly, np.mean(fax.get_xlim())) + 1.5, poly_text, {
        'color': 'black',
        'fontsize': 13,
        'ha': 'left',
        'va': 'top',
        'bbox': box_settings
    })
    if save_bool:
        if df['freq'].min() < 0.1:
            fig_name = 'low'
        else:
            fig_name = 'high'
        fit_fig.savefig(os.path.join(comb_bda_data.analysis_dir,
                                     f'linear_fit_with_{fig_name}_freq_scatter.pdf'), dpi=200)
    return fit_fig


def eol_scatter_plot(df, plot_name='eol_default', x_axis='time', plot_mode='full', save_bool=True):
    # pulse_eol_df = comb_bda_data.eol_df[comb_bda_data.eol_df['test_dur'] != 3600].sort_values(by='test_dur')
    eol_fig, eol_ax = plt.subplots(1, 1)
    if x_axis == 'time':
        x_plot = 'test_dur'
        xlab = 'Pulse duration - log scale'
    else:
        x_plot = 'freq'
        xlab = 'Frequency of pulse test - log scale'
    eol_ax.scatter(df[x_plot], df['eol_avg'], color='blue', edgecolors='black',
                   label='Average of replicates')
    if plot_mode=='full':
        eol_ax.scatter(df[x_plot], df['eol'], color='red', edgecolors='black',
                       label='Unique value')
    eol_ax.set_xscale('log')
    plt.title('Pulse duration v average expected energy throughput at EOL \n'
              'EOL at {}%, data {}.'.format(75, 'interpolated'))
    plt.xlabel(xlab)
    plt.ylabel('Energy throughput at EOL')
    eol_ax.legend()
    eol_fig.savefig(os.path.join(comb_bda_data.analysis_dir, f'{plot_name}_x_axis_{x_plot}.pdf'), dpi=200)
    eol_ax.set_ylim((0, 16))
    if save_bool:
        eol_fig.savefig(os.path.join(comb_bda_data.analysis_dir, f'{plot_name}_x_axis_{x_plot}_large_ax.pdf'), dpi=200)
    return None


def build_test_dict(full_data_set, test_to_find):
    dict = {}
    full_list = full_data_set.summary_dict.keys()
    for t in test_to_find:
        for c in full_list:
            if t == re.findall(r'\d+s', c)[0]:
                dict[c] = full_data_set.summary_dict[c]
    return dict


def calculate_averages(org_rpt_obj, property='cap_relative'):
    tests_in_dataset = org_rpt_obj.summary_dict.keys()
    pulse_cond = [re.search(r'\b\d+s', k).group() for k in tests_in_dataset]
    unique_cond = list(set(pulse_cond))
    grouped_data_full = {}
    for k in unique_cond:
        tmp_test = [org_rpt_obj.summary_dict[key] for key in tests_in_dataset if k==re.search(r'\b\d+s', key).group()]
        grouped_data_full[k] = tmp_test
    averaged_data = {}
    for k in grouped_data_full:
        tmp_df = pd.DataFrame()
        for i, d in enumerate(grouped_data_full[k]):
            cap_data = d.data[property]
            tmp_df.loc[:, f'rep_{i}'] = cap_data
        tmp_df.loc[:, 'mean_val'] = tmp_df.dropna(how='any').mean(axis=1)
        tmp_df.loc[:, 'err'] = tmp_df.filter(like='rep').dropna(how='any').std(axis=1)
        tmp_df.loc[:, 'FCE'] = d.data['FCE']
        tmp_df.set_index('FCE')
        averaged_data[k] = tmp_df
    return averaged_data


def save_fig_multiple_format(fig, filename):
    format_list = ['png', 'pdf', 'jpg', 'eps']
    for fmt in format_list:
        fig.savefig(f'{filename}.{fmt}', format=fmt, dpi=200)
    tmp_png = BytesIO()
    fig.savefig(tmp_png, format='png')
    png_pil = Image.open(tmp_png)
    png_pil.save(f'{filename}.tiff')
    tmp_png.close()
    return None


def color_coding_max_contrast(test_name, col_format='hex'):
    list_of_tests = [f'{dur}s' for dur in 2**np.arange(0, 9, 1)]
    list_of_tests.append('3600s')
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
     # Rest of Kelly's suggested colors
     # '#F2F3F4', '#0067A5', '#F99379', '#604E97', '#F6A600', '#B3446C', '#DCD300', '#882D17', '#8DB600', '#654522',
     # '#E25822', '#2B3D26']
    col_dict = dict(zip(list_of_tests, kelly_colors_hex))
    return col_dict[test_name.split('_')[0]]


if __name__ == '__main__':
    fix_mpl_backend()
    bda_comp = r"\\sol.ita.chalmers.se\groups\batt_lab_data\20210816"
    bda_orig = r"\\sol.ita.chalmers.se\groups\batt_lab_data\20200923_pkl"
    bda_comp_data = OrganiseRpts(bda_comp, proj='bda_comp')
    bda_orig_data = OrganiseRpts(bda_orig)
    new_fig_folder = r"Z:\Documents\Papers\TeslaLowFrequencyPulse\updated_images"

    all_comp_test = [key.split('_')[0] for key in bda_comp_data.ica_dict.keys()]
    comb_bda_data = merge_data_sets(bda_orig_data, bda_comp_data)
    comb_bda_data.eol_df = comb_bda_data.find_eol()
    comb_bda_data.eol_df.loc[:, 'freq'] = 1 / comb_bda_data.eol_df.loc[:, 'test_dur']
    comb_bda_data.plot_eol(savefig=True)
    avg_cap_dict = calculate_averages(comb_bda_data, property='cap_relative')
    avg_res_dict = calculate_averages(comb_bda_data, property='res_dchg_50_relative')

    rpt_fig, rax = plt.subplots(1, 1)
    for test_id in bda_comp_data.summary_dict:
        data_set = bda_comp_data.summary_dict[test_id]
        df = bda_comp_data.summary_dict[test_id].data
        lab = f'{test_id}_{data_set.proj_name}'
        rax.plot(df.egy_thrg, df.cap_relative, label=lab, linestyle=data_set.set_linestyle())
    rax.set_xlabel('Discharge energy throughput')
    rax.set_ylabel('Relative capacity retention')
    rax.legend()

    all_tests = set([key.split('_')[0] for key in comb_bda_data.summary_dict.keys()])
    ref_dict = {k: comb_bda_data.summary_dict[k] for k in comb_bda_data.summary_dict if '3600s' in k}
    for t_id in all_tests:
        t_fig, tax = plt.subplots(1, 1)
        for t in comb_bda_data.summary_dict:
            if t_id == re.findall(r'\d+s', t)[0]:
                data_set = comb_bda_data.summary_dict[t]
                df = comb_bda_data.summary_dict[t].data
                lab = f'{t_id}_{data_set.channel_id}_{data_set.proj_name.lower()}'
                #print(f'Identified {t_id} in {t} with {lab}')
                tax.plot(df.FCE, df.cap_relative, label=lab, linestyle=set_linestyle_bda_dataset(data_set.proj_name))
        for k in ref_dict:
            ref_label = f'1C reference {k.split("s_")[-1]}'
            tax.plot(ref_dict[k].data.FCE, ref_dict[k].data.cap_relative,
                     label=ref_label, linestyle='dotted')
        tax.legend()
        tax.set_ylim((0.6, 1.02))
        tax.set_xlabel('Full cycle equivalents')
        tax.set_ylabel('Relative capacity retention')
        t_fig.savefig(os.path.join(comb_bda_data.analysis_dir, f'all_tests_for_{t_id}_fix_axis_w_reference.pdf'))

    pulse_eol_df = comb_bda_data.eol_df[comb_bda_data.eol_df['test_dur'] != 3600].sort_values(by='test_dur')
    eol_scatter_plot(pulse_eol_df, 'eol_scatter_plot_full')
    long_pulse_df = comb_bda_data.eol_df[(comb_bda_data.eol_df['test_dur'] != 3600) & (comb_bda_data.eol_df['test_dur'] > 10)].sort_values(by='test_dur')
    eol_scatter_plot(long_pulse_df, plot_name='long_pulse_eol')
    eol_scatter_plot(long_pulse_df, plot_name='long_pulse_eol', x_axis='freq')
    eol_scatter_plot(long_pulse_df, plot_name='long_pulse_eol_average', plot_mode='partial', x_axis='freq')
    short_pulse_df = comb_bda_data.eol_df[(comb_bda_data.eol_df['test_dur'] != 3600) & (comb_bda_data.eol_df['test_dur'] < 10)].sort_values(by='test_dur')
    eol_scatter_plot(short_pulse_df, plot_name='short_pulse_eol')
    eol_scatter_plot(short_pulse_df, plot_name='short_pulse_eol', x_axis='freq')
    eol_scatter_plot(short_pulse_df, plot_name='short_pulse_eol_average', plot_mode='partial', x_axis='freq')

    scatter_with_linear_fit(long_pulse_df, save_bool=False)
    poly_long_pulse = np.polyfit(long_pulse_df['freq']*1000, long_pulse_df['eol'], deg=1)
    fit_arr_long = np.linspace((long_pulse_df['freq'].min()-0.008)*1000, 1000*(long_pulse_df['freq'].max() + 0.035), 100)
    fit_arr_short = np.linspace((short_pulse_df['freq'].min()-0.03)*1000, 1000*(short_pulse_df['freq'].max() + 0.1), 100)
    poly_short_pulse = np.polyfit(short_pulse_df['freq']*1000, short_pulse_df['eol'], deg=1)

    eol_fig, eol_ax = plt.subplots(1, 1)
    eol_ax.scatter(pulse_eol_df['freq']*1000, pulse_eol_df['eol'], color='red', edgecolors='black', label='Unique value')
    eol_ax.scatter(pulse_eol_df['freq']*1000, pulse_eol_df['eol_avg'], color='blue',
                   edgecolors='black', label='Average value')
    eol_ax.plot(fit_arr_long, np.polyval(poly_long_pulse, fit_arr_long),
                label='Linear fit low frequency', color='forestgreen')
    eol_ax.plot(fit_arr_short, np.polyval(poly_short_pulse, fit_arr_short),
                label='Linear fit high frequency', color='black')
    eol_ax.set_xlabel('Frequency / mHz', fontdict=lbl_font)
    eol_ax.set_ylabel('Energy throughput at EOL / kWh', fontdict=lbl_font)
    eol_ax.legend()
    #eol_fig.savefig(os.path.join(comb_bda_data.analysis_dir, f'all_data_with_linear_fits_v_freq.pdf'), dpi=200)
    save_fig_multiple_format(eol_fig, os.path.join(comb_bda_data.analysis_dir, 'all_data_with_linear_fits_v_freq_mhz'))

    poly_text_short = f'Linear fit:\n{poly_short_pulse[1]:.2f} + {1000*poly_short_pulse[0]:.2f} * f'
    poly_text_long = f'Linear fit:\n{poly_long_pulse[1]:.2f} + {1000*poly_long_pulse[0]:.2f} * f'

    box_settings = {
        "boxstyle": "round",
        "fc": "white",
        "ec": "black",
        "pad": 0.3
    }
    plt.text(150, 8, poly_text_long, {
        'color': 'black',
        'fontsize': 13,
        'ha': 'left',
        'va': 'top',
        'bbox': box_settings
    })
    plt.text(800, 12, poly_text_short, {
        'color': 'black',
        'fontsize': 13,
        'ha': 'left',
        'va': 'top',
        'bbox': box_settings
    })
    eol_fig.savefig(os.path.join(comb_bda_data.analysis_dir, f'all_data_with_linear_fits_v_freq_mhz_and_text.pdf'), dpi=300)
    save_fig_multiple_format(eol_fig, os.path.join(new_fig_folder, "all_data_with_linear_fits_v_freq_mhz_and_text"))

    bp_fig, bp_ax = plt.subplots(1, 1)
    pulse_eol_df.boxplot(column='eol', by='freq', ax=bp_ax, fontsize='large', grid=False)
    bp_ax.set_ylabel('Energy throughput at EOL / kWh', fontdict=lbl_font)
    bp_ax.set_xlabel('Frequency / mHz', weight='normal', fontdict=lbl_font, labelpad=4)
    bp_ax.set_title('')
    bp_fig.suptitle('', fontsize=18)
    x_tick_labels_mhz = [f'{1000*float(k._get_wrapped_text()):.0f}' for k in bp_ax.get_xticklabels()]
    x_tick_labels_hz = [f'{float(k._get_wrapped_text()):.2e}' for k in bp_ax.get_xticklabels()]
    tick_labels = bp_ax.set_xticklabels(x_tick_labels_mhz)
    #plt.setp(bp_ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    #plt.subplots_adjust(bottom=.17)
    save_fig_multiple_format(bp_fig, os.path.join(comb_bda_data.analysis_dir,
                                                  'all_data_plus_ref_boxplot_no_grid_no_title'))
    save_fig_multiple_format(bp_fig, os.path.join(new_fig_folder, 'all_data_plus_ref_boxplot_no_grid_no_title'))
    save_fig_multiple_format(bp_fig, os.path.join(r"Z:\Documents\Papers\LicentiateThesis\images", "box_plot_bda"))

    # bp_fig.savefig(os.path.join(comb_bda_data.analysis_dir, 'all_data_plus_ref_boxplot_low_res_no_grid_no_title.pdf'),
    #                dpi=200)
    # bp_fig.savefig(os.path.join(comb_bda_data.analysis_dir, 'all_data_plus_ref_boxplot_low_res_no_grid_no_title.pdf'),
    #                dpi=200)

    marker_list = ['v', '.', 's', '^', '*']
    style_list = ['dashed', 'dashdot', (0, (1, 1))]
    m = len(marker_list)
    s = len(style_list)
    ax00_case = [f'{t}s' for t in [1, 2]]
    ax01_case = [f'{t}s' for t in [4, 8]]
    ax10_case = [f'{t}s' for t in [16, 32]]
    ax11_case = [f'{t}s' for t in [64, 128, 256]]
    ax00_dict = build_test_dict(comb_bda_data, ax00_case)
    ax01_dict = build_test_dict(comb_bda_data, ax01_case)
    ax10_dict = build_test_dict(comb_bda_data, ax10_case)
    ax11_dict = build_test_dict(comb_bda_data, ax11_case)
    # Unique figures, for latex
    dict_list = [ax00_dict, ax01_dict, ax10_dict, ax11_dict]
    for dct in dict_list:
        fig_id = dct[list(dct.keys())[0]].test_name
        tmp_fig, tax = plt.subplots(1, 1)
        for k in dct:
            f_ = 1000 / dct[k].test_dur
            tax.plot(dct[k].data.FCE, dct[k].data.cap_relative, label=f'$f={f_:.0f}$mHz')
        for i, l in enumerate(tax.lines):
            l.set_marker(marker_list[i % m])
            l.set_linestyle(style_list[i % s])
        tax.set_xlabel('Full cycle equivalents')
        tax.set_ylabel('Normalised capacity retention')
        tax.plot(avg_cap_dict['3600s'].FCE, avg_cap_dict['3600s']['mean_val'], linestyle=(0, (3, 5, 1, 5, 1, 5)),
                 marker='o', label='1C reference', color='forestgreen')
        tax.set_xlim((0, 800))
        tax.set_ylim((0.60, 1.02))
        tax.legend(loc='lower left')
        save_fig_multiple_format(tmp_fig, os.path.join(comb_bda_data.analysis_dir,
                                                       f'capacity_fade_{fig_id}'))
        tmp_fig.savefig(os.path.join(new_fig_folder, f'capacity_fade_{fig_id}.pdf'), dpi=300)

    f_fig, f_ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(16, 12))
    tmp_lgd_size = 11
    for k in ax00_dict:
        f_lab = 1000 / ax00_dict[k].test_dur
        f_ax[0, 0].plot(ax00_dict[k].data.FCE, ax00_dict[k].data.cap_relative, label=f'$f={f_lab:.0f}$mHz')
    for k in ax01_dict:
        f_lab = 1000 / ax01_dict[k].test_dur
        f_ax[0, 1].plot(ax01_dict[k].data.FCE, ax01_dict[k].data.cap_relative, label=f'$f={f_lab:.0f}$mHz')
    for k in ax10_dict:
        f_lab = 1000 / ax10_dict[k].test_dur
        f_ax[1, 0].plot(ax10_dict[k].data.FCE, ax10_dict[k].data.cap_relative, label=f'$f={f_lab:.0f}$mHz')
    for k in ax11_dict:
        f_lab = 1000 / ax11_dict[k].test_dur
        f_ax[1, 1].plot(ax11_dict[k].data.FCE, ax11_dict[k].data.cap_relative, label=f'$f={f_lab:.0f}$mHz')
    f_ax[0, 1].set_ylim((0.65, 1.02))
    f_ax[0, 1].set_xlim((0, 900))
    # f_fig.text(0.5, 0.05, 'Full cycle equivalents', ha='center', fontsize=24)
    # f_fig.text(0.05, 0.5, 'Normalised capacity retention', va='center', rotation='vertical', fontsize=24)
    f_ax[0, 0].set_ylabel('Normalised capacity retention')
    f_ax[1, 0].set_ylabel('Normalised capacity retention')
    f_ax[1, 0].set_xlabel('Full cycle equivalents')
    f_ax[1, 1].set_xlabel('Full cycle equivalents')
    for ax in f_ax.reshape(-1):
        for i, l in enumerate(ax.lines):
            l.set_marker(marker_list[i % m])
            l.set_linestyle(style_list[i % s])
    f_ax[0, 0].legend(loc='lower left', fontsize=tmp_lgd_size)
    f_ax[0, 1].legend(loc='lower left', fontsize=tmp_lgd_size)
    f_ax[1, 0].legend(loc='lower left', fontsize=tmp_lgd_size)
    f_ax[1, 1].legend(loc='lower left', fontsize=tmp_lgd_size)
    save_fig_multiple_format(f_fig, os.path.join(comb_bda_data.analysis_dir, 'all_data_four_plots_for_pres'))
    # f_fig.savefig(os.path.join(comb_bda_data.analysis_dir, 'all_data_four_plots_for_pres.pdf'), dpi=200)
    for ax in f_ax.reshape(-1):
        ax.plot(ref_dict['3600s_2_8'].data.FCE, ref_dict['3600s_2_8'].data.cap_relative, label='1C reference',
                linestyle=(0, (3, 5, 1, 5, 1, 5)), marker='o', color='forestgreen')
        ax.plot(ref_dict['3600s_2_4'].data.FCE, ref_dict['3600s_2_4'].data.cap_relative, label='1C reference',
                linestyle=(0, (3, 5, 1, 5, 1, 5)), marker='o', color='darkmagenta')
        ax.plot(ref_dict['3600s_2_7'].data.FCE, ref_dict['3600s_2_7'].data.cap_relative, label='1C reference',
                linestyle=(0, (3, 5, 1, 5, 1, 5)), marker='o', color='indigo')
    # for ax in f_ax.reshape(-1):
    #     for k in ref_dict:
    #         ax.plot(ref_dict[k].data.FCE, ref_dict[k].data.cap_relative, label='1C reference',
    #                 linestyle=(0, (3, 5, 1, 5, 1, 5)), marker='o')
    f_ax[0, 0].legend(loc='lower left', fontsize=tmp_lgd_size, ncols=2)
    f_ax[0, 1].legend(loc='lower left', fontsize=tmp_lgd_size, ncols=2)
    f_ax[1, 0].legend(loc='lower left', fontsize=tmp_lgd_size, ncols=2)
    f_ax[1, 1].legend(loc='lower left', fontsize=tmp_lgd_size, ncols=2)
    save_fig_multiple_format(f_fig, os.path.join(comb_bda_data.analysis_dir, 'all_data_four_plots_for_pres_w_allref'))
    save_fig_multiple_format(f_fig, os.path.join(new_fig_folder, 'all_data_four_plots_for_pres_w_allref'))
    # f_fig.savefig(os.path.join(comb_bda_data.analysis_dir, 'all_data_four_plots_for_pres_w_allref.pdf'), dpi=200)

    mark_dict = {
        '1s': '<',
        '2s': 'v',
        '4s': '4',
        '8s': '*',
        '16s': '.',
        '32s': 's',
        '64s': 'p',
        '128s': '>',
        '256s': '^',
        '3600s': 'o'
    }
    color_dict = {
        1: 'maroon',
        2: 'forestgreen',
        4: 'orangered',
        8: 'mediumblue',
        16: 'crimson',
        32: 'darkolivegreen',
        64: 'darkviolet',
        128: 'black',
        256: 'indianred',
        3600: 'yellowgreen'
    }
    linestyle_dict = {
        1: 'dashed',
        2: 'dotted',
        4: 'dashdot',
        8: (0, (5, 10)),
        16: (0, (5, 1)),
        32: (0, (3, 1, 1, 1)),
        64: (0, (3, 5, 1, 5, 1, 5)),
        128: 'dashed',
        256: 'dotted',
        3600: 'solid'
    }
    label_name_dict = {
        '1s': ['1s', f'${1000/1:.0f}$ mHz'],
        '2s': ['2s', f'${1000/2:.0f}$ mHz'],
        '4s': ['4s', f'${1000/4:.0f}$ mHz'],
        '8s': ['8s', f'${1000/8:.0f}$ mHz'],
        '16s': ['16s', f'${1000/16:.0f}$ mHz'],
        '32s': ['32s', f'${1000/32:.0f}$ mHz'],
        '64s': ['64s', f'${1000/64:.0f}$ mHz'],
        '128s': ['128s', f'${1000/128:.0f}$ mHz'],
        '256s': ['256s', f'${1000/256:.0f}$ mHz'],
        '3600s': ['1C Reference', '1C Reference']
    }

    t_dur_list = [2**N for N in np.arange(0, 9, 1)]
    cr_fig, cr_ax = plt.subplots(1, 1)
    cr_ax2 = cr_ax.twinx()
    cvr_fig, cvr_ax = plt.subplots(1, 1)
    # for t_dur in t_dur_list:
    t_dur = 2
    t_list = [8,  128, 3600]
    for t_dur in t_list:
        call_name = f'{t_dur}s'
        cr_plot_dict = {k: comb_bda_data.summary_dict[k] for k in comb_bda_data.summary_dict
                        if f'{t_dur}s'==re.search(r'\b\d+s', k).group()}
        for i, k in enumerate(cr_plot_dict):
            cr_ax.plot(cr_plot_dict[k].data.FCE, cr_plot_dict[k].data.cap_relative,
                       label=f'$f={1/t_dur:.2e}Hz$ replicate {i + 1}',
                       linestyle=(0, (3, 5, 1, 5, 1, 5)),
                       marker=mark_dict[f'{t_dur}s'])
            cr_ax2.plot(cr_plot_dict[k].data.FCE, cr_plot_dict[k].data.res_dchg_50_relative,
                        label=f'$f={1/t_dur:.2e}Hz$ replicate {i + 1}',
                        linestyle='dashdot',
                        marker=mark_dict[f'{t_dur}s'])
            cvr_ax.plot(1 - cr_plot_dict[k].data.cap_relative, cr_plot_dict[k].data.res_dchg_50_relative,
                        label=f'{label_name_dict[call_name][1]} replicate {i + 1}',
                        linestyle=(0, (3, 5, 1, 5, 1, 5)),
                        marker=mark_dict[f'{t_dur}s'])
    cr_ax.legend(loc='upper left')
    cr_ax2.legend(loc='upper right')
    cr_ax.set_ylabel('Normalised capacity retention')
    cr_ax2.set_ylabel('Normalised discharge resistance')
    cr_ax.set_xlabel('Full cycle equivalents')
    cr_ax2.grid(False)
    cr_ax.set_ylim((0.65, 1))
    cr_ax.set_xlim((0, 900))
    cr_ax2.set_ylim((1, 2))
    cvr_ax.legend(loc='upper left')
    cvr_ax.set_xlabel('Relative capacity loss')
    cvr_ax.set_ylabel('Normalised discharge resistance')
    fig_unique_name = "_".join([str(k) for k in t_list])
    save_fig_multiple_format(cr_fig, os.path.join(comb_bda_data.analysis_dir, f'cap_and_res_tdur{fig_unique_name}s'))
    save_fig_multiple_format(cvr_fig, os.path.join(comb_bda_data.analysis_dir, f'cap_v_res_tdur{fig_unique_name}s'))
    # cr_fig.savefig(os.path.join(comb_bda_data.analysis_dir, f'cap_and_res_tdur{fig_unique_name}s.pdf'), dpi=200)
    # cvr_fig.savefig(os.path.join(comb_bda_data.analysis_dir, f'cap_v_res_tdur{fig_unique_name}.pdf'), dpi=200)

    qe_fig, qe_ax = plt.subplots(1, 1)
    for k in natsort.natsorted(avg_cap_dict):
        tmp_df = avg_cap_dict[k]
        qe_ax.errorbar(tmp_df.FCE, tmp_df.mean_val*100, yerr=tmp_df.err*100,
                       color=color_coding_max_contrast(k),
                       elinewidth=1.5,
                       marker='s',
                       markersize=mark_size,
                       capsize=cap_size,
                       label=label_name_dict[k][1])
    qe_ax.legend(loc='lower left', ncols=2)
    qe_ax.set_xlim(-2, 1000)
    qe_ax.set_ylim(55, 103)
    qe_ax.set_xlabel('Number of Full Cycle Equivalents / -', fontdict=lbl_font)
    qe_ax.set_ylabel('Percentage of Capacity Retained / -', fontdict=lbl_font)
    qe_ax.grid(color='grey', alpha=0.5)
    qe_fig.savefig(os.path.join(comb_bda_data.analysis_dir, 'capacity_retention_w_errorbar_bda_study.pdf'))
    qe_fig.savefig(os.path.join(comb_bda_data.analysis_dir, 'capacity_retention_w_errorbar_bda_study.png'), dpi=300)
    qe_fig.savefig(os.path.join(comb_bda_data.analysis_dir, 'capacity_retention_w_errorbar_bda_study.eps'), dpi=300)

    qe_sub_fig, qe_sub_ax = plt.subplots(1, 1)
    for k in natsort.natsorted(avg_cap_dict):
        if float(k.rstrip('s')) > 10:
            tmp_df = avg_cap_dict[k]
            qe_sub_ax.errorbar(tmp_df.FCE, tmp_df.mean_val * 100, yerr=tmp_df.err * 100,
                           color=color_coding_max_contrast(k),
                           elinewidth=1.5,
                           marker='s',
                           markersize=mark_size,
                           capsize=cap_size,
                           label=label_name_dict[k][1])
        qe_sub_ax.legend(loc='lower left', ncols=2)
        qe_sub_ax.set_xlim(-2, 1000)
        qe_sub_ax.set_ylim(55, 103)
        qe_sub_ax.set_xlabel('Number of Full Cycle Equivalents / -', fontdict=lbl_font)
        qe_sub_ax.set_ylabel('Percentage of Capacity Retained / -', fontdict=lbl_font)
        qe_sub_ax.grid(color='grey', alpha=0.5)
        save_fig_multiple_format(qe_sub_fig, os.path.join(r"Z:\Documents\Papers\LicentiateThesis\images", "cap_fade_subset"))

    t_list = 2**np.arange(0, 9, 1)
    # t_list = [1, 128]
    for t_dur in t_list:
        cra_fig, cra_ax = plt.subplots(1, 1)
        cra_ax2 = cra_ax.twinx()
        cra_ax2.grid(False)
        cap_df = avg_cap_dict[f'{t_dur}s'].dropna(how='any')
        res_df = avg_res_dict[f'{t_dur}s'].dropna(how='any')
        case_name = f'$f={1000/t_dur:.0f}mHz$'
        for col in cap_df.columns:
            # if 'mean' in col:
            #     cra_ax.plot(cap_df.FCE, cap_df[col], linewidth=0.9, linestyle=linestyle_dict[t_dur],
            #                 label='$Q$ average ' + case_name, color=color_dict[t_dur])
            #     cra_ax2.plot(res_df.FCE, res_df[col], linewidth=0.9, linestyle=linestyle_dict[t_dur],
            #                  label='$R_{10s}$ average ' + case_name, color=color_dict[t_dur])
            if 'rep' in col:
                cra_ax.plot(cap_df.FCE, cap_df[col],
                            linestyle='None', marker=(5, 2), color=color_dict[t_dur], markersize=6)
                cra_ax2.plot(res_df.FCE, res_df[col],
                             linestyle='None', marker='.', color=color_dict[t_dur], markersize=6)
            if 'mean' in col:
                cra_ax.plot(cap_df.FCE, cap_df[col], linewidth=0.9, linestyle='dashed',
                            label='$Q$ average ' + case_name, color=color_dict[t_dur])
                cra_ax2.plot(res_df.FCE, res_df[col], linewidth=0.9, linestyle='dashdot',
                             label='$R_{10s}$ average ' + case_name, color=color_dict[t_dur])
        cra_ax.lines[0].set_label('$Q$ data ' + case_name)
        cra_ax2.lines[0].set_label('$R_{10s}$ data ' + case_name)
        cra_ax.plot(avg_cap_dict['3600s'].FCE, avg_cap_dict['3600s']['mean_val'],
                    label='$Q$ 1C reference', color='forestgreen')
        cra_ax2.plot(avg_res_dict['3600s'].FCE, avg_res_dict['3600s']['mean_val'],
                     label='$R_{10s}$ 1C reference', color='indianred')
        cra_ax.set_ylim((0.65, 1))
        cra_ax.set_xlim((0, 800))
        cra_ax2.set_ylim((1, 2))
        cra_ax.set_ylabel('Normalised capacity retention')
        cra_ax2.set_ylabel('Normalised discharge resistance')
        cra_ax.set_xlabel('Full cycle equivalents')
        h1, l1 = cra_ax.get_legend_handles_labels()
        h2, l2 = cra_ax2.get_legend_handles_labels()
        cra_ax.legend(h1 + h2, l1 + l2, loc='center left', fontsize=11)
        fig_name = '_'.join([str(k) for k in t_list])
        save_fig_multiple_format(cra_fig, os.path.join(comb_bda_data.analysis_dir,
                                                       f'cap_res_plus_average_tdur_{t_dur}s_800fce_1decpt'))
        save_fig_multiple_format(cra_fig, os.path.join(new_fig_folder, f'cap_res_plus_average_tdur_{t_dur}s_800fce_1decpt'))
        # cra_fig.savefig(os.path.join(comb_bda_data.analysis_dir, f'cap_res_plus_average_tdur_{t_dur}s_800fce.pdf'),
        #                 dpi=200)
