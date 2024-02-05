import pandas as pd
import numpy as np
from PythonScripts.rpt_data_analysis.ReadRptClass import OrganiseRpts, ReadRpt
import matplotlib.pyplot as plt
import os
import re
from natsort import natsort_keygen


class AlineOrganiseRpts(OrganiseRpts):
    def __init__(self, directory):
        super().__init__(directory)

    def fill_data(self, directory, clean_data=False):
        name_dict = {}
        for root, dirs, files in os.walk(directory):
            if 'pickle' in root:
                tmp_ica = {}
                tmp_raw = {}
                for file in files:
                    try:
                        rpt_key = re.findall(r'rpt_\d+', file)[0]
                    except IndexError:
                        if 'summary' in file:
                            rpt_data = AlineRptReader(os.path.join(root, file))
                        continue
                    if 'ica_dump' in file:
                        tmp_ica[rpt_key] = pd.read_pickle(os.path.join(root, file))
                    if 'rpt_raw' in file:
                        tmp_raw[rpt_key] = pd.read_pickle(os.path.join(root, file))
                test_name = f'{rpt_data.test_info_dict["TEST CASE"].rstrip(" ")}'
                name_dict[test_name] = [rpt_data.test_name, rpt_data.channel_id]
                self.ica_dict[test_name] = tmp_ica
                self.rpt_raw_dict[test_name] = tmp_raw
                self.summary_dict[test_name] = rpt_data
        self.name_df = self.fill_name_df(name_dict)
        try:
            self.eol_df = self.find_eol()
        except AttributeError:
            print('No extrapolation available due to few data points.')
        return None


class AlineRptReader(ReadRpt):
    def __init__(self, file_name):
        super().__init__(file_name=file_name)
        self.base_dir = os.path.dirname(file_name)
        self.test_info_dict = {}
        self.read_data()
        self.read_test_info()

    def read_data(self):
        if self.call_name.endswith('.xlsx'):
            self.data = pd.read_excel(self.call_name, index_col=0)
        elif self.call_name.endswith('.pkl'):
            self.data = pd.read_pickle(self.call_name)
        if 'Unnamed: 0' in self.data.columns:
            self.data.set_index('Unnamed: 0', inplace=True)
        return None

    def check_cell_id(self):
        self.cell_id = re.search(r'cell\d+', self.call_name).group()
        return None

    def read_test_info(self):
        d = {}
        fname = self.find_test_info_file()
        with open(fname) as f:
            for line in f:
                (key, val) = line.split(": ")
                d[key] = val.strip()
        self.test_info_dict = d
        return None

    def find_test_info_file(self):
        f_list = os.listdir(self.base_dir)
        info_fname = [f for f in f_list if 'test_info' in f][0]
        f_path = os.path.join(self.base_dir, info_fname)
        return f_path


# DATA_ID = [f'RPT_240095_1_{k}' for k in range(1, 9)]
# TEST_SOC = ['5-15', '15-25', '15-25', '25-35', '25-35', '35-45', '35-45', '85-95']
# CELL_ID = [680, 672, 691, 681, 683, 694, 697, 660]
# TEST_ID = [f'{tn}% SOC_cell{cid}' for tn, cid in zip(TEST_SOC, CELL_ID)]
# TEST_ID_DICT = dict(zip(DATA_ID, TEST_ID))
ALL_TESTS = ['5-15',
             '15-25',
             '25-35',
             '35-45',
             '45-55',
             '55-65',
             '65-75',
             '75-85',
             '85-95']
COLOR_RGB = np.array([(170, 111, 158),
                      (136, 46, 114),
                      (67, 125, 191),
                      (123, 175, 222),
                      (144, 201, 135),
                      (247, 240, 86),
                      (244, 167, 54),
                      (230, 85, 24),
                      (165, 23, 14)]) / 255
COLOR_HEX = np.array(['#aa6f9e',
                      '#882e72',
                      '#437dbf',
                      '#7bafde',
                      '#90c987',
                      '#f7f056',
                      '#f4a736',
                      '#e65518',
                      '#a5170e'
                      ])
COLOR_DICT = dict(zip(ALL_TESTS, COLOR_HEX))



file_loc = r"\\sol.ita.chalmers.se\groups\batt_lab_data\ALINE_data\After_long_storage_RPT"
output_ = r"\\sol.ita.chalmers.se\groups\batt_lab_data\analysis_directory\Aline_after_rest"
my_data = AlineOrganiseRpts(file_loc)

for k, rpt_obj in my_data.summary_dict.items():
    t_case = rpt_obj.test_info_dict['TEST CASE']
    soc_lvl = re.search(r'\d+-\d+', t_case).group()
    rpt_obj.test_info_dict['SOC LEVEL'] = soc_lvl
    rpt_obj.test_info_dict['PLOT COLOR'] = COLOR_DICT[soc_lvl]

plt.style.use('kelly_colors')
fig = plt.figure()
for k, ica_obj in my_data.ica_dict.items():
    for n, df in ica_obj.items():
        plt.plot(df[df['mode'] == 'CC_Chg'].cap,
                 df[df['mode'] == 'CC_Chg'].dva_gauss,
                 label=k,
                 color=my_data.summary_dict[k].test_info_dict['PLOT COLOR'])
plt.ylim(0, 0.7)
plt.xlabel('Capacity [mAh]')
plt.ylabel(r'DV, dV dQ$^{-1}$ [V mAh$^{-1}$]')
plt.legend()
plt.savefig(os.path.join(output_, 'dva_all_cells.png'), dpi=300)

fig = plt.figure()
for k, ica_obj in my_data.ica_dict.items():
    for n, df in ica_obj.items():
        plt.plot(df[df['mode'] == 'CC_Chg'].volt,
                 df[df['mode'] == 'CC_Chg'].ica_gauss,
                 dashes=[4, 1],
                 label=k,
                 color=my_data.summary_dict[k].test_info_dict['PLOT COLOR'])
        c = plt.gca().lines[-1].get_color()
        plt.plot(df[df['mode'] == 'CC_DChg'].volt,
                 df[df['mode'] == 'CC_DChg'].ica_gauss,
                 dashes=[4, 1],
                 color=c)
plt.xlabel('Voltage [V]')
plt.ylabel(r'IC, dQ dV$^{-1}$ [mAh V$^{-1}$]')
plt.legend()
plt.savefig(os.path.join(output_, 'ica_all_cells.png'), dpi=300)


plt.style.use('ml_colors')
cap_data_ = pd.DataFrame.from_dict({k: rp.data.cap for k, rp in my_data.summary_dict.items()})
cap_data_sorted_ = cap_data_.sort_index(axis=1, key=natsort_keygen())
cap_data_sorted_.columns = [k.replace('_', ' ') for k in cap_data_sorted_]
soc_lvls = [re.search(r'\d+-\d+', t_str).group() for t_str in cap_data_sorted_.columns]
col_lst = [COLOR_DICT[sc] for sc in soc_lvls]
ax = cap_data_sorted_.transpose().plot.bar(ylim=(3700, 4100), edgecolor='#e0e0e0', linewidth=0.5)
for p, c in zip(ax.patches, col_lst):
    p.set_color(c)
ax.grid(False)
ax.get_legend().remove()
ax.set_ylabel('Capacity retention [mAh]')
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.subplots_adjust(bottom=.17)
plt.subplots_adjust(left=0.2)
plt.savefig(os.path.join(output_, 'bar_chart_all_cells.png'), dpi=300)
plt.savefig(os.path.join(output_, 'bar_chart_all_cells.pdf'))
