import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import square
import os
from backend_fix import fix_mpl_backend
from scipy.stats import f_oneway
fix_mpl_backend()

my_lrg_font = 16
my_med_font = 14
my_sml_font = 12

x_width = 7
aspect_rat = 9 / 16
plt.rcParams['figure.figsize'] = x_width, aspect_rat * x_width
plt.rcParams['legend.fontsize'] = my_med_font
plt.rcParams['axes.labelsize'] = my_lrg_font
plt.rcParams['axes.titlesize'] = my_lrg_font
plt.rcParams['axes.grid'] = False
plt.rcParams['lines.linewidth'] = 1.7
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams["text.usetex"] = True
lbl_font = {'weight': 'normal',
            'size': my_lrg_font}
plt.rc('legend', fontsize=my_sml_font)
mark_size = 5
cap_size = 6
plt.rc('font', **{"family": 'sans-serif', 'sans-serif': 'Helvetica'})

mrk_font = {'weight': 'normal',
            'size': my_sml_font}

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

output_dir = r"Z:\Documents\Papers\LicentiateThesis\images"

################################################# VISUALISE RPT LAYOUT #################################################
ref_rpt = r"\\sol.ita.chalmers.se\groups\batt_lab_data\20210816\pickle_files_channel_1_2\1_2_rpt_raw_dump_rpt_1.pkl"
rpt_df = pd.read_pickle(ref_rpt)

rpt_df.loc[:, 'float_time'] = rpt_df.loc[:, 'float_time'] - rpt_df.float_time.iloc[0]
cap_end_step = 12
ica_end_step = 14
cap_end_idx = rpt_df[rpt_df.arb_step2 == cap_end_step].last_valid_index()
ica_end_idx = rpt_df[rpt_df.arb_step2 == ica_end_step].last_valid_index()
cap_end_time = rpt_df.loc[cap_end_idx, 'float_time']
ica_end_time = rpt_df.loc[ica_end_idx, 'float_time']
fin_time = rpt_df.loc[rpt_df.last_valid_index(), 'float_time']

rpt_fig, rax = plt.subplots(2, 1, sharex=True, figsize=(x_width, 1.4 * aspect_rat * x_width))
rax[0].plot(rpt_df.float_time / 3600, rpt_df.volt, color='black')
rax[1].plot(rpt_df.float_time / 3600, rpt_df.curr, color='black')
rax[1].set_xlabel('Time, h')
rax[0].set_ylabel('Voltage, V')
rax[1].set_ylabel('Current, A')

# Color code the different parts of the rpt depending on action
rax[0].axvspan(-1, cap_end_time / 3600, facecolor='green', alpha=0.5)
rax[0].axvspan(cap_end_time / 3600, ica_end_time / 3600, facecolor='red', alpha=0.5)
rax[0].axvspan(ica_end_time / 3600, fin_time / 3600 + 4, facecolor='blue', alpha=0.5)
y_text_pos = 4.5
rax[0].text(-0.2, y_text_pos, 'Capacity Test', fontdict=mrk_font)
rax[0].text(20, y_text_pos, 'ICA Test', fontdict=mrk_font)
rax[0].text(52, y_text_pos, 'Impedance \nTest', fontdict=mrk_font)
rax[0].set_ylim(2.5, 5)
rpt_fig_labels = ['(a)', '(b)']
for k, label in enumerate(rpt_fig_labels):
    rax[k].text(-0.12, 1.0, label, transform=rax[k].transAxes, fontsize=14)

# Toggle boolean to output to file
update_file = 0
if update_file:
    rpt_fig.savefig(os.path.join(output_dir, 'rpt_visual_w_label.pdf'), dpi=300)
    rpt_fig.savefig(os.path.join(output_dir, 'rpt_visual_w_label.png'), dpi=300)

rax[1].axvspan(-1, cap_end_time / 3600, facecolor='green', alpha=0.5)
rax[1].axvspan(cap_end_time / 3600, ica_end_time / 3600, facecolor='red', alpha=0.5)
rax[1].axvspan(ica_end_time / 3600, fin_time / 3600 + 4, facecolor='blue', alpha=0.5)

if update_file:
    rpt_fig.savefig(os.path.join(output_dir, 'rpt_visual_all_colored_w_label.pdf'), dpi=300)
    rpt_fig.savefig(os.path.join(output_dir, 'rpt_visual_all_colored_w_label.png'), dpi=300)

################################################# VISUALISE SMART CELL #################################################
x_width = 6
aspect_rat = 12 / 16
plt.rcParams['figure.figsize'] = x_width, aspect_rat * x_width
plt.rcParams['legend.fontsize'] = my_med_font
plt.rcParams['axes.labelsize'] = my_lrg_font
plt.rcParams['axes.titlesize'] = my_lrg_font
plt.rcParams['axes.grid'] = True
ref_1c = r"\\sol.ita.chalmers.se\groups\batt_lab_data\smart_cell_JG\LabTestsMJ1\VCC01\VCC01_Test1436.csv"
ref_1hz = r"\\sol.ita.chalmers.se\groups\batt_lab_data\smart_cell_JG\LabTestsMJ1\VCC02\VCC02_Test1443.csv"
ref_10mhz = r"\\sol.ita.chalmers.se\groups\batt_lab_data\smart_cell_JG\LabTestsMJ1\VCC16\VCC16_Test1576.csv"
cols_ac = ['cyc', 'time', 'volt', 'curr', 'chrg_cap', 'dchg_cap', 'chrg_egy', 'dchg_egy', 'temperature', '']
cols_dc = ['cyc', 'time', 'volt', 'curr', 'chrg_cap', 'dchg_cap', 'chrg_egy', 'dchg_egy', 'temperature_pt100',
           'temperature', '']
df_1c = pd.read_csv(ref_1c, skiprows=42, names=cols_dc)
df_1hz = pd.read_csv(ref_1hz, skiprows=42, names=cols_dc)
df_10mhz = pd.read_csv(ref_10mhz, skiprows=42, names=cols_ac)

## DUE TO SAMPLING INHOMOGENEITY WE GENERATE ARTIFICAL PROFILE
t = np.linspace(0, 600, 30000, endpoint=False)
sq_wave_1hz = square(2 * np.pi * t, 0.5)
sq_wave_10mhz = square(2 / 10 * np.pi * t, 0.5)
curr_1c = 3.6
i_fig, iax = plt.subplots(1, 1, figsize=(x_width * 1.2, aspect_rat * x_width))
iax.plot(t + 0.25, curr_1c * sq_wave_1hz + curr_1c, label='1 Hz', color='#E68FAC')
iax.plot(t, curr_1c * sq_wave_10mhz + curr_1c, label='100 mHz', color='#875692')
iax.plot(t, curr_1c * np.ones_like(t), label='1C Reference', color='#222222')
iax.grid(color='grey', alpha=0.5)
iax.set_xlim(0, 20)
iax.set_xlabel('Time / s')
iax.set_ylabel('Current / A')
iax.set_yticks(np.arange(0, 8, 2))
box = iax.get_position()
iax.set_position([box.x0 - 0.025, box.y0 + .04, box.width * 0.75, box.height])
iax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

update_file = 0
if update_file:
    i_fig.savefig(os.path.join(output_dir, 'smart_cell_visual.pdf'), dpi=300)
    i_fig.savefig(os.path.join(output_dir, 'smart_cell_visual.png'), dpi=300)

## VISUALISE THE SIMULATIONS PERFORMED IN PROJECT
full_sweep = r"\\sol.ita.chalmers.se\groups\batt_lab_data\smart_cell_JG\simulations\output_data_long_chrg_sim_2c.csv"
mid_sweep = r"\\sol.ita.chalmers.se\groups\batt_lab_data\smart_cell_JG\simulations\output_data_05_005hz.csv"
col_names = ['freq', 'time', 'cl_cc', 'cl_sep', 'cs_surf_cc', 'cs_surf_sep', 'cs_center_cc', 'cs_center_sep',
             'dc_cc', 'dc_sep', 'phil_cc', 'phil_sep', 'Eeq_cc', 'Eeq_sep', 'eta_cc', 'eta_sep',
             'i_ct_cc', 'i_ct_sep', 'i_dl_cc', 'i_dl_sep']
full_df = pd.read_csv(full_sweep, header=4, names=col_names)
mid_df = pd.read_csv(mid_sweep, header=4, names=col_names)
gb = full_df.groupby(by='freq')
f_dct = {k: gb.get_group(k) for k in gb.groups}
gb_comp = mid_df.groupby(by='freq')
f_dct_comp = {k: gb_comp.get_group(k) for k in gb_comp.groups}
f_dct.update(f_dct_comp)

sim_fig, sax = plt.subplots(1, 1)
sax.plot(f_dct[1].time, -f_dct[1].phil_sep, color=kelly_colors_hex[2], label='1 Hz', alpha=0.8)
sax.plot(f_dct[0.05].time, -f_dct[0.05].phil_sep, color=kelly_colors_hex[0], label='50 mHz')
sax.plot(f_dct[0.1].time, -f_dct[0.1].phil_sep, color=kelly_colors_hex[3], label='100 mHz')
sax.axhline(0, color='black', label='0 V')
# sax.set_xlim(-3, 80)
sax.set_ylim(-0.11, 0.11)
sax.grid(color='grey', alpha=0.5)
sax.legend(loc='lower left', ncols=2)
sax.set_xlabel('Time [s]')
sax.set_ylabel(r'$\eta_{sep}$ v Li/Li$^+$ [V] ')
box = sax.get_position()
sax.set_position([box.x0 + 0.05, box.y0 + .04, box.width, box.height])

update_file = 1
if update_file:
    sim_fig.savefig(os.path.join(output_dir, 'eta_sep_simulated_full_t.pdf'), dpi=300)
    sim_fig.savefig(os.path.join(output_dir, 'eta_sep_simulated_full_t.png'), dpi=300)

sax.set_ylim(-0.11, 0.1)
sax.set_xlim(230, 270)
if update_file:
    sim_fig.savefig(os.path.join(output_dir, 'eta_sep_simulated_zoom.pdf'), dpi=300)
    sim_fig.savefig(os.path.join(output_dir, 'eta_sep_simulated_zoom.png'), dpi=300)

dc_fig, dcax = plt.subplots(1, 1)
dcax.plot(f_dct[1].time, f_dct[1].dc_sep, color=kelly_colors_hex[2], label='1 Hz', alpha=0.8)
dcax.plot(f_dct[0.05].time, f_dct[0.05].dc_sep, color=kelly_colors_hex[0], label='50 mHz')
dcax.plot(f_dct[0.1].time, f_dct[0.1].dc_sep, color=kelly_colors_hex[3], label='100 mHz')
# dcax.axhline(0, color='black', label='0 V')
dcax.set_xlim(-3, 120)
# dcax.set_ylim()
dcax.grid(color='grey', alpha=0.5)
dcax.legend(loc='upper left', ncols=3)
dcax.set_xlabel('Time [s]')
dcax.set_ylabel(r'$\Delta c$ / mol m$^{-3}$ ')
box = sax.get_position()
dcax.set_position([box.x0, box.y0, box.width, box.height])

if update_file:
    dc_fig.savefig(os.path.join(output_dir, 'dc_sep_simulated.pdf'), dpi=300)
    dc_fig.savefig(os.path.join(output_dir, 'dc_sep_simulated.png'), dpi=300)

cmax_fig, cmax = plt.subplots(1, 1)
cmax.plot(f_dct[1].time, f_dct[1].cs_surf_sep, color=kelly_colors_hex[2], label='1 Hz', alpha=0.8)
cmax.plot(f_dct[0.05].time, f_dct[0.05].cs_surf_sep, color=kelly_colors_hex[0], label='50 mHz')
cmax.plot(f_dct[0.1].time, f_dct[0.1].cs_surf_sep, color=kelly_colors_hex[3], label='100 mHz')

cmax.set_ylim(17000, 29000)
cmax.grid(color='grey', alpha=0.5)
cmax.legend(loc='upper left', ncols=3)
cmax.set_xlabel('Time [s]')
cmax.set_ylabel(r'$c_{s,surf}$ [mol m$^{-3}$] ')
box = sax.get_position()
cmax.set_position([box.x0, box.y0, box.width, box.height])

if update_file:
    cmax_fig.savefig(os.path.join(output_dir, 'cs_surf_sep_simulated_full_t.pdf'), dpi=300)
    cmax_fig.savefig(os.path.join(output_dir, 'cs_surf_sep_simulated_full_.png'), dpi=300)

cmax.set_xlim(230, 270)
cmax.set_ylim(24000, 27000)

if update_file:
    cmax_fig.savefig(os.path.join(output_dir, 'cs_surf_sep_simulated_zoom.pdf'), dpi=300)
    cmax_fig.savefig(os.path.join(output_dir, 'cs_surf_sep_simulated_zoom.png'), dpi=300)

cmax.axhline(34684, color='black', label='$c_{s,max}$')
cmax.set_ylim(17000, 40000)

############################################### VISUALISE DATASET SPREAD ###############################################
mean_vals = [4, 4, 4, 4, 4]
std_vals = np.array([2, 3, 2, 3, 1]) * 2
n_samples = 40
gr = []
for i, item in enumerate(mean_vals):
    gr.append(np.random.normal(mean_vals[i], std_vals[i], n_samples))
aov_results = f_oneway(*gr)

box_fig, bax = plt.subplots(1, 1)
bax.boxplot(gr)
bax.set_title(f'Dataset with {100*(1 - aov_results.pvalue):.1f} probability rejecting $H_0$.')


################################################## VISUALISE BDA FREQ ##################################################
# plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': 'Times New Roman'})  # **{"family": 'sans-serif', 'sans-serif': 'Helvetica'}
font_xticks = {
    'family': 'sans-serif',
    'sans-serif': 'Helvetica'
}
font_legend = {
    'fontsize': 14
}
curr_bda = 3
f_arr = 1 / (2**np.arange(0, 9, 1))
wave_dict = {f'{1/f:.0f}': square(2*np.pi*f*t, 0.25) * curr_bda - curr_bda for f in f_arr}
plot_offset = 6.5
bda_fig, bfax = plt.subplots(1, 1)
mHz = 'mHz'
for i, k in enumerate(wave_dict):
    bfax.plot(t, wave_dict[k] + i*plot_offset, color=kelly_colors_hex[i])   #, label=f'${1000*1/float(k):.0f}$ mHz '
lst_of_legends = [f'{1000*1/float(k):.0f} mHz' for k in wave_dict]
bfax.set_xlim(0, 300)
bfax.set_ylim(-7, 80)
bfax.set_xlabel('Time [s]')
bfax.set_ylabel('Arbitrary unit [-]')
L = bfax.legend(lst_of_legends, ncols=3, fontsize=13)
bda_fig.subplots_adjust(bottom=0.13)
bda_fig.savefig(os.path.join(output_dir, 'bda_pulse_visual_all_freq.pdf'), dpi=300)
# plt.setp(L.texts, family='Times New Roman')

#### WITHOUT OFFSET - REDUCED SAMPLE
wave_fig, wax = plt.subplots(1, 1)
f1 = 1000 / 2
f2 = 1000 / 16
wax.plot(t, wave_dict['2'], label=f'$f={f1:.0f}$ mHz', color=kelly_colors_hex[0])
wax.plot(t - 1, wave_dict['16'], label=f'$f={f2:.0f}$ mHz', color=kelly_colors_hex[1])
wax.legend(ncols=2)
wax.set_xlim(0, 48)
wax.set_ylim(-6.5, 2)
wax.set_xlabel('Time / s')
wax.set_ylabel('Current / A')
wave_fig.subplots_adjust(bottom=0.13)
wave_fig.savefig(os.path.join(output_dir, 'bda_pulse_visual_two_freq.pdf'), dpi=300)
wave_fig.savefig(os.path.join(output_dir, 'bda_pulse_visual_two_freq.png'), dpi=300)

#### PLOTS FOR PRESENTATION
owid_ghg = r"Z:\Documents\Papers\LicentiateThesis\Presentation\Global-GHG-Emissions-by-sector-based-on-WRI-2020.xlsx"
owid_full = r"Z:\Documents\Papers\LicentiateThesis\Presentation\co-emissions-by-sector.csv"

pie_colors = ['#222222', '#F3C300', '#875692', '#F38400', '#A1CAF1', '#BE0032', '#C2B280',
              '#848482', '#008856', '#E68FAC', '#0067A5', '#F99379', '#604E97', '#F6A600',
              '#B3446C', '#DCD300', '#882D17']
wp = {'linewidth': 0.5, 'edgecolor': "black"}
text_size = 14

df_full = pd.read_csv(owid_full)
dfw = df_full.groupby(by='Entity').get_group('World')
dfw['sum'] = dfw.drop('Year', axis=1).sum(axis=1, numeric_only=True)
dfw_norm = dfw.select_dtypes(include=[np.number]).drop('Year', axis=1).div(dfw['sum'], axis=0)
dfw_norm.set_index(dfw.loc[:, 'Year'], inplace=True)
dfw_norm.drop('sum', axis=1, inplace=True)
N = dfw_norm.shape[1]
explode = np.zeros(N)
explode[[4]] = 0.2
pie_fig, pie_ax = plt.subplots(1, 1, figsize=(8,6))
wedges, texts, autotexts = pie_ax.pie(dfw_norm.iloc[-1],
                                   explode=explode,
                                   labeldistance=.6,
                                   pctdistance=1.25,
                                   shadow=True,
                                   autopct='%1.1f%%',
                                   textprops={'fontsize': text_size},
                                   colors=pie_colors,
                                   startangle=90,
                                   wedgeprops=wp
                                   )
pie_ax.set_position([-0.01, 0.15, 0.7, 0.7])
pie_ax.legend(wedges, dfw_norm.columns,
              fontsize=text_size,
              loc='center right',
              bbox_to_anchor=(1.25, -0.35, 0.5, 1))
pie_fig.savefig(os.path.join(output_dir, 'ghg_emissions_by_sector_wexplode.png'), dpi=300, transparent=True)
pie_fig.savefig(os.path.join(output_dir, 'ghg_emissions_by_sector.eps'), dpi=300)

xl_ghg = pd.ExcelFile(owid_ghg)
df_ghg = xl_ghg.parse('Sub-sector')
df_ghg.loc[:, 'labels'] = df_ghg.loc[:1, 'Sub-sector']

pfig, pax = plt.subplots(1, 1, figsize=(8, 6))
explode_vals = np.zeros(df_ghg.shape[0])
explode_vals[:2] = 0.3
wedges, texts, autotexts = pax.pie(df_ghg['Share_ghg'],
                                   explode=explode_vals,
                                   labeldistance=.6,
                                   pctdistance=1.25,
                                   shadow=True,
                                   autopct='%1.1f%%',
                                   textprops={'fontsize': 12},
                                   colors=pie_colors,
                                   startangle=90,
                                   wedgeprops=wp
                                   )
pax.set_position([0, 0.15, 0.7, 0.7])
pax.legend(wedges, df_ghg['labels'].dropna(),
           fontsize=12,
           loc='center right',
           bbox_to_anchor=(1.2, 0, 0.5, 1))
pfig.savefig(os.path.join(output_dir, 'ghg_emissions_pie.png'), dpi=300, transparent=True)
