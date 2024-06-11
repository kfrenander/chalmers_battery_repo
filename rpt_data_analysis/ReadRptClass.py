import re
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import os
import matplotlib as mpl
from scipy.stats import norm
from backend_fix import fix_mpl_backend
plt.rcParams['axes.grid'] = True
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['figure.figsize'] = 10, 8


def look_up_fce(rpt_str):
    rpt_num = int(re.search(r'\d+', rpt_str).group())
    fce = 50*(rpt_num - 1)
    return fce


class OrganiseRpts(object):

    def __init__(self, directory, clean_data=False, proj='BDA'):
        self.name_df = pd.DataFrame()
        self.ica_dict = {}
        self.rpt_raw_dict = {}
        self.summary_dict = {}
        self.eol_val = ''
        self.proj_name = proj
        self.fill_data(directory, clean_data)
        self.cmap = plt.get_cmap('tab20b')
        analysis_date = dt.datetime.now().strftime('%Y-%m-%d')
        self.analysis_dir = os.path.join(r"\\sol.ita.chalmers.se\groups\batt_lab_data\analysis_directory",
                                         analysis_date)

    def calc_replicate_average(self, col_name):
        df = pd.DataFrame()
        for rpt_obj in self.summary_dict.values():
            name = rpt_obj.test_dur
            tmp_df = pd.DataFrame(rpt_obj.data[col_name]).rename(columns={col_name: name})
            df = pd.concat([df, tmp_df], axis=1)
        df = df.groupby(by=df.columns, axis=1).mean()
        return df

    def fill_data(self, directory, clean_data):
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
                            rpt_data = ReadRpt(os.path.join(root, file),
                                               clean_data=clean_data,
                                               proj_name=self.proj_name)
                        continue
                    if 'ica_dump' in file:
                        tmp_ica[rpt_key] = pd.read_pickle(os.path.join(root, file))
                    if 'rpt_raw' in file:
                        tmp_raw[rpt_key] = pd.read_pickle(os.path.join(root, file))
                test_name = '{}_{}'.format(rpt_data.test_name, rpt_data.channel_id)
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

    def fill_name_df(self, name_dict: dict):
        name_df = pd.DataFrame(name_dict).transpose()
        name_df.columns = ['test_name', 'channel_id']
        name_df['cell_nbr'] = np.zeros_like(name_df['test_name'])
        name_df.loc[:, 'cell_id'] = [self.summary_dict[k].cell_id for k in self.summary_dict]
        for name in name_df.test_name.unique():
            sub_df = name_df[name_df.test_name == name]
            nbr_of_cells = len(sub_df)
            idx = sub_df.index
            name_df.loc[idx, 'cell_nbr'] = [(i + 1) for i in range(nbr_of_cells)]
        return name_df

    def find_dist_param(self, data_set):
        mu = np.mean(data_set)
        sigma = np.std(data_set)
        return mu, sigma

    def plot_cap_distribution(self, rpt_num, mah=True):
        if mah:
            data_set = self.find_arb_cap(rpt_num)
        else:
            data_set = self.find_arb_cap(rpt_num) / 1000
        fig, ax = plt.subplots(1, 1)
        d_max = data_set.max().values
        d_min = data_set.min().values
        span = d_max - d_min
        x_vals = np.arange(d_min - span/10, d_max + span/10, span/1000)
        mu, sigma = self.find_dist_param(data_set)
        pdf = norm.pdf(x_vals, loc=mu, scale=sigma)
        ax.plot(x_vals, pdf)
        ax.hist(data_set, bins=6, rwidth=1/2, density=True)
        ax.set_ylabel('Density [-]')
        ax.set_xlabel('Capacity [-]')
        return fig

    def find_init_cap(self):
        tmp = self.summary_dict
        cap_dct = {k: tmp[k].data.loc['rpt_1', 'cap'] for k in tmp}
        init_cap_df = pd.DataFrame.from_dict(cap_dct, orient='index', columns=['Rpt_cap'])
        return init_cap_df

    def find_arb_cap(self, rpt_num):
        tmp = self.summary_dict
        cap_dct = {k: tmp[k].data.loc[f'rpt_{rpt_num}', 'cap'] for k in tmp}
        init_cap_df = pd.DataFrame.from_dict(cap_dct, orient='index', columns=[f'Rpt_cap'])
        return init_cap_df

    def find_arb_normal_cap(self, rpt_num):
        tmp = self.summary_dict
        cap_dct = {k: tmp[k].data.loc[f'rpt_{rpt_num}', 'cap_relative'] for k in tmp}
        init_cap_df = pd.DataFrame.from_dict(cap_dct, orient='index', columns=[f'Rpt_cap'])
        return init_cap_df

    def plot_average_cap(self,
                         test_name=['1s', '2s', '4s', '8s', '16s', '32s', '64s', '128s'],
                         savefig=False):
        cap_avg = self.calc_replicate_average('cap_relative')
        egy_avg = self.calc_replicate_average('egy_dchg')
        avg_fig, ax = plt.subplots(1, 1)
        # if re.findall(r'\d+s', key)[0] in test_name:
        for dur in egy_avg.columns:
            if re.findall(r'\d+s', '{}s'.format(dur))[0] in test_name:
                ax.plot(egy_avg[dur], cap_avg[dur], label='Pulse duration {}s'.format(dur), linestyle='solid', marker='.')
        ax.set_xlabel('Discharge energy throughput')
        ax.set_ylabel('Relative capacity retention')
        ax.set_title('Average of two replicates at each testing point')
        ax.lines[-1].set_color('black')
        plt.legend()
        if savefig:
            tests = '_'.join(test_name)
            if not os.path.exists(self.analysis_dir):
                os.makedirs(self.analysis_dir)
            avg_fig.savefig(os.path.join(self.analysis_dir,
                                         'average_capacity_plot_tests_{}.png'.format(tests)),
                            dpi=1200)
        return avg_fig

    def plot_dva(self, test_name, rpt_num=None, cell_nbr=[1, 2], savefig=False):
        import natsort
        dva_fig, dva_ax = plt.subplots(1, 1)
        for key in self.ica_dict:
            if self.check_to_plot(key, cell_nbr, test_name):
            # if re.findall(r'\d+s', key)[0] in test_name and self.name_df.loc[key, 'cell_nbr'] in cell_nbr:
                if rpt_num is None:
                    rpt_num = ['rpt_{}'.format(i + 1) for i in range(len(self.ica_dict[key]))]
                test_ica = self.ica_dict[key]
                for test in natsort.natsorted(test_ica):
                    if test in rpt_num:
                        plt.plot(test_ica[test].cap, test_ica[test].dva_gauss,
                                 label='{}_{} FCE'.format(key, self.fce_converter(test)),
                                 linewidth=0.85)
        plt.legend()
        plt.xlabel('Capacity [mAh]')
        plt.ylabel('DVA [dv/dQ]')
        plt.title('Differential Voltage Analysis')
        plt.ylim([-0.7, 0.6])
        plt.tight_layout()
        if savefig:
            if not os.path.exists(self.analysis_dir):
                os.makedirs(self.analysis_dir)
            op_name = self.name_output('DVA', test_name, rpt_num, cell_nbr)
            dva_fig.savefig(op_name, dpi=1200)
            # tests = '_'.join(test_name)
            # rpts = '_'.join(rpt_num)
            # cells = '_'.join(['{}'.format(cell) for cell in cell_nbr])
            # dva_fig.savefig(os.path.join(self.analysis_dir,
            #                              'DVA_tests_{}_rpt_{}_cells_{}.png'.format(tests, rpts, cells)), dpi=1200)
        return dva_ax

    def plot_ica(self, test_name, rpt_num=None, cell_nbr=[1, 2], savefig=False):
        import natsort
        ica_fig, ica_ax = plt.subplots(1, 1)
        for key in self.ica_dict:
            if self.check_to_plot(key, cell_nbr, test_name):
            # if re.findall(r'\d+s', key)[0] in test_name and self.name_df.loc[key, 'cell_nbr'] in cell_nbr:
                if rpt_num is None:
                    rpt_num = ['rpt_{}'.format(i + 1) for i in range(len(self.ica_dict[key]))]
                test_ica = self.ica_dict[key]
                for test in natsort.natsorted(test_ica):
                    if test in rpt_num:
                        plt.plot(test_ica[test].volt, test_ica[test].ica_gauss,
                                 label='{}_{} FCE'.format(key, self.fce_converter(test)),
                                 linewidth=0.85)
        plt.legend()
        plt.xlabel('Voltage [V]')
        plt.ylabel('ICA [dQ/dV]')
        plt.title('Incremental Capacity Analysis')
        plt.tight_layout()
        if savefig:
            if not os.path.exists(self.analysis_dir):
                os.makedirs(self.analysis_dir)
            op_name = self.name_output('ICA', test_name, rpt_num, cell_nbr)
            ica_fig.savefig(op_name, dpi=1200)
            # tests = '_'.join(test_name)
            # rpts = '_'.join(rpt_num)
            # cells = '_'.join(['{}'.format(cell) for cell in cell_nbr])
            # ica_fig.savefig(os.path.join(self.analysis_dir,
            #                              'ICA_tests_{}_rpt_{}_cells_{}.png'.format(tests, rpts, cells)), dpi=1200)
        return ica_ax

    def plot_hysteresis(self, test_name, rpt_num=None, cell_nbr=[1, 2], savefig=False):
        import natsort
        from scipy.interpolate import interp1d
        hyst_fig, hyst_ax = plt.subplots(2, 1)
        for key in self.ica_dict:
            if self.check_to_plot(key, cell_nbr, test_name):
            # if re.findall(r'\d+s', key)[0] in test_name and self.name_df.loc[key, 'cell_nbr'] in cell_nbr:
                if rpt_num is None:
                    rpt_num = ['rpt_{}'.format(i + 1) for i in range(len(self.ica_dict[key]))]
                test_ica = self.ica_dict[key]
                for test in natsort.natsorted(test_ica):
                    tmp_ica = test_ica[test]
                    if tmp_ica.empty or tmp_ica.curr.mean() < -0.2:
                        print('No ICA found in {} for test {}.'.format(test, key))
                        continue
                    soc = (tmp_ica.mAh - tmp_ica.mAh.min()) / (tmp_ica.mAh.max() - tmp_ica.mAh.min())
                    tmp_ica.loc[:, 'soc'] = soc
                    u_int_chrg = interp1d(tmp_ica[tmp_ica.curr > 0].soc, tmp_ica[tmp_ica.curr > 0].volt)
                    u_int_dchg = interp1d(tmp_ica[tmp_ica.curr < 0].soc, tmp_ica[tmp_ica.curr < 0].volt)
                    x_low = max(tmp_ica[tmp_ica.curr > 0].soc.min(), tmp_ica[tmp_ica.curr < 0].soc.min())
                    x_hi = min(tmp_ica[tmp_ica.curr > 0].soc.max(), tmp_ica[tmp_ica.curr < 0].soc.max())
                    x_int = np.linspace(x_low, x_hi, 400)
                    if test in rpt_num:
                        hyst_ax[0].plot(tmp_ica.soc, tmp_ica.volt,
                                        label='Voltage {}_{} FCE'.format(key, self.fce_converter(test)),
                                        linewidth=0.85)
                        hyst_ax[1].plot(x_int, u_int_chrg(x_int) - u_int_dchg(x_int),
                                        label='Hysteresis {}_{} FCE'.format(key, self.fce_converter(test)),
                                        linewidth=0.85)
        plt.legend()
        hyst_ax[0].set_xlabel('SOC [-]')
        hyst_ax[1].set_ylabel('Hysteresis [V]')
        hyst_ax[0].set_ylabel('Voltage [V]')
        hyst_ax[1].set_ylim([0, 0.35])
        plt.title('Voltage and hysteresis')
        plt.tight_layout()
        if savefig:
            if not os.path.exists(self.analysis_dir):
                os.makedirs(self.analysis_dir)
            op_name = self.name_output('Hysteresis_plot', test_name, rpt_num, cell_nbr)
            hyst_fig.savefig(op_name, dpi=1200)
            # tests = '_'.join(test_name)
            # rpts = '_'.join(rpt_num)
            # cells = '_'.join(['{}'.format(cell) for cell in cell_nbr])
            # hyst_fig.savefig(os.path.join(self.analysis_dir,
            #                              'Hysteresis_plot_tests_{}_rpt_{}_cells_{}.png'.format(tests, rpts, cells)), dpi=1200)
        return hyst_ax

    def plot_ah_volt(self, test_name, rpt_num=None, cell_nbr=[1, 2], savefig=False):
        import natsort
        ah_fig, ah_ax = plt.subplots(1, 1)
        colors = iter(self.cmap(np.linspace(0, 1, 12)))
        for key in self.ica_dict:
            if self.check_to_plot(key, cell_nbr, test_name):
            # if re.findall(r'\d+s', key)[0] in test_name and self.name_df.loc[key, 'cell_nbr'] in cell_nbr:
                if rpt_num is None:
                    rpt_num = ['rpt_{}'.format(i + 1) for i in range(len(self.ica_dict[key]))]
                test_ica = self.ica_dict[key]
                for test in natsort.natsorted(test_ica):
                    if test in rpt_num:
                        try:
                            plt.plot((test_ica[test].mAh - test_ica[test].mAh.min()) / 1000, test_ica[test].volt,
                                     label='{}_{} FCE'.format(key, self.fce_converter(test)),
                                     linewidth=0.7, color=next(colors))
                        except StopIteration:
                            colors = iter(self.cmap(np.linspace(0.1, 1, 12)))
                            plt.plot((test_ica[test].mAh - test_ica[test].mAh.min()) / 1000, test_ica[test].volt,
                                     label='{}_{} FCE'.format(key, self.fce_converter(test)),
                                     linewidth=0.85, color=next(colors))
        plt.legend()
        plt.xlabel('Capacity [Ah]')
        plt.ylabel('Voltage [V]')
        plt.title('Capacity v Voltage')
        plt.tight_layout()
        if savefig:
            if not os.path.exists(self.analysis_dir):
                os.makedirs(self.analysis_dir)
            op_name = self.name_output('ah_volt', test_name, rpt_num, cell_nbr)
            ah_fig.savefig(op_name, dpi=1200)
            # tests = '_'.join(test_name)
            # rpts = '_'.join(rpt_num)
            # cells = '_'.join(['{}'.format(cell) for cell in cell_nbr])
            # ah_fig.savefig(os.path.join(self.analysis_dir,
            #                              'ah_volt_tests_{}_rpt_{}_cells_{}.png'.format(tests, rpts, cells)), dpi=1200)
        return ah_ax

    def plot_soc_volt(self, test_name, rpt_num=None, cell_nbr=[1, 2], savefig=False):
        import natsort
        soc_fig, soc_ax = plt.subplots(1, 1)
        colors = iter(self.cmap(np.linspace(0, 1, 12)))
        for key in self.ica_dict:
            if self.check_to_plot(key, cell_nbr, test_name):
            # if re.findall(r'\d+s', key)[0] in test_name and self.name_df.loc[key, 'cell_nbr'] in cell_nbr:
                if rpt_num is None:
                    rpt_num = ['rpt_{}'.format(i + 1) for i in range(len(self.ica_dict[key]))]
                test_ica = self.ica_dict[key]
                for test in natsort.natsorted(test_ica):
                    if test in rpt_num:
                        step_cap = test_ica[test].mAh.max() - test_ica[test].mAh.min()
                        try:
                            plt.plot((test_ica[test].mAh - test_ica[test].mAh.min()) / step_cap, test_ica[test].volt,
                                     label='{}_{} FCE'.format(key, self.fce_converter(test)),
                                     linewidth=0.7, color=next(colors))
                        except StopIteration:
                            colors = iter(self.cmap(np.linspace(0.1, 1, 12)))
                            plt.plot((test_ica[test].mAh - test_ica[test].mAh.min()) / step_cap, test_ica[test].volt,
                                     label='{}_{} FCE'.format(key, self.fce_converter(test)),
                                     linewidth=0.7, color=next(colors))
        plt.legend()
        plt.xlabel('SOC [-]')
        plt.ylabel('Voltage [V]')
        plt.title('SOC v Voltage')
        plt.tight_layout()
        if savefig:
            if not os.path.exists(self.analysis_dir):
                os.makedirs(self.analysis_dir)
            op_name = self.name_output('soc_volt', test_name, rpt_num, cell_nbr)
            soc_fig.savefig(op_name, dpi=1200)
            # tests = '_'.join(test_name)
            # rpts = '_'.join(rpt_num)
            # cells = '_'.join(['{}'.format(cell) for cell in cell_nbr])
            # soc_fig.savefig(os.path.join(self.analysis_dir,
            #                              'soc_volt_tests_{}_rpt_{}_cells_{}.png'.format(tests, rpts, cells)), dpi=1200)
        return soc_ax

    def plot_rpt_data(self, test_name,
                      cell_nbr=[1, 2],
                      x_mode='dchg',
                      y_mode='cap',
                      savefig=False,
                      plot_title='',
                      cell_id_label=False):
        rpt_fig = plt.figure()
        test_name = self.fix_list(test_name)
        for key in self.summary_dict:
            if any([self.name_df.loc[key, 'test_name'] == name for name in test_name]) \
                    and self.name_df.loc[key, 'cell_nbr'] in cell_nbr:
                self.summary_dict[key].plot_data(x_mode, y_mode, cell_id_label=cell_id_label)
        rpt_fig.gca().set_title(plot_title)
        if savefig:
            tests = '_'.join(test_name)
            cells = '_'.join(['{}'.format(cell) for cell in cell_nbr])
            rpt_fig.savefig(os.path.join(self.analysis_dir, 'RPT_tests_{}_cells_{}_x_{}_y_{}.png'.format(
                tests, cells, x_mode, y_mode
            )), dpi=1200)
        return rpt_fig

    def check_to_plot(self, key, cell_nbr, test_name):
        if 'SOC' in test_name:
            if test_name in key:
                return True
            else:
                return False
        else:
            try:
                if re.findall(r'\d+s', key)[0] in test_name and self.name_df.loc[key, 'cell_nbr'] in cell_nbr:
                    return True
                else:
                    return False
            except IndexError:
                print(f'Did not find relevant data when seeking to plot test {test_name} for key {key}')
                return False

    def name_output(self, plot_name, test_name, rpt_num, cell_nbr):
        if 'SOC' in test_name:
            rpts = '_'.join(rpt_num)
            cells = '_'.join(['{}'.format(cell) for cell in cell_nbr])
            return os.path.join(self.analysis_dir,
                                '{}_tests_{}_rpt_{}_cells_{}.png'.format(plot_name, test_name, rpts, cells))
        else:
            tests = '_'.join(test_name)
            rpts = '_'.join(rpt_num)
            cells = '_'.join(['{}'.format(cell) for cell in cell_nbr])
            return os.path.join(self.analysis_dir,
                                '{}_tests_{}_rpt_{}_cells_{}.png'.format(plot_name, tests, rpts, cells))

    @staticmethod
    def fix_list(maybe_list):
        if isinstance(maybe_list, list):
            return maybe_list
        else:
            return [maybe_list]

    @staticmethod
    def fce_converter(rpt_str):
        rpt_num = int(re.search(r'\d+', rpt_str).group())
        return (rpt_num - 1) * 50

    def plot_eol(self, mode='full', savefig=False, eol_val=75, data_use='interpolation'):
        if eol_val != self.eol_val:
            self.find_eol(eol_val=eol_val)
        eol_fig = plt.figure()
        ax = plt.gca()
        if data_use == 'interpolation':
            un_col_name = 'eol_interpolated'
            avg_col_name = 'eol_interpolated_avg'
            data_label = 'interpolated'
        else:
            un_col_name = 'eol'
            avg_col_name = 'eol_avg'
            data_label = 'extrapolated'
        if mode == 'full':
            ax.scatter(self.eol_df.test_dur, self.eol_df[avg_col_name], color='blue', edgecolors='black',
                       label='Average of replicates')
            ax.scatter(self.eol_df.test_dur, self.eol_df[un_col_name], color='red', edgecolors='black',
                       label='Unique value')
        elif mode == 'avg':
            ax.scatter(self.eol_df.test_dur, self.eol_df[avg_col_name], color='blue', edgecolors='black',
                       label='Average of replicates')
        else:
            ax.scatter(self.eol_df.test_dur, self.eol_df[un_col_name], color='red', edgecolors='black',
                       label='Unique value')
        plt.title('Pulse duration v average expected energy throughput at EOL \n'
                  'EOL at {}%, data {}.'.format(self.eol_val, data_label))
        plt.xlabel('Logarithm of pulse duration')
        plt.ylabel('Energy throughput at EOL')
        ax.set_xscale('log')
        plt.legend()
        if savefig:
            if not os.path.exists(self.analysis_dir):
                os.makedirs(self.analysis_dir)
            eol_fig.savefig(os.path.join(self.analysis_dir,
                                         'test_duration_v_eol_data_{}_mode_{}_eol_{}.png'.format(
                                             data_label, mode, self.eol_val)), dpi=1200)
        return eol_fig

    def find_eol(self, eol_val=75):
        self.eol_val = eol_val
        eol_dict = {k: self.summary_dict[k].find_eol(self.eol_val) for k in self.summary_dict}
        eol_interpolate_dict = {k: self.summary_dict[k].find_eol_interpolation(self.eol_val) for k in self.summary_dict}
        eol_df = self.name_df.copy()
        eol_df['eol'] = pd.DataFrame.from_dict(eol_dict, orient='index')
        eol_df['eol_interpolated'] = pd.DataFrame.from_dict(eol_interpolate_dict, orient='index')
        avg_eol = {nm: [eol_df[eol_df.test_name == eol_df.loc[nm, 'test_name']].eol.mean(),
                        eol_df[eol_df.test_name == eol_df.loc[nm, 'test_name']].eol_interpolated.mean()]
                   for nm in eol_df.index.unique()}
        avg_df = pd.DataFrame.from_dict(avg_eol, orient='index', columns=['eol_avg', 'eol_interpolated_avg'])
        eol_df = pd.concat([eol_df, avg_df], axis=1)
        dur_dict = {k: self.summary_dict[k].test_dur for k in self.summary_dict}
        eol_df['test_dur'] = pd.DataFrame(dur_dict.values(), index=dur_dict.keys(), columns=['test_duration'])
        return eol_df

    @staticmethod
    def combine_plots(list_of_figs):
        fig, ax_fig = plt.subplots(1, 1)
        ax_list = [fig.gca() for fig in list_of_figs]
        for ax in ax_list:
            ax_fig.plot(ax.lines[0].get_xdata(), ax.lines[0].get_ydata())
            scat = ax.get_children()[0]
            ax_fig.scatter(scat.get_offsets()[:, 0], scat.get_offsets()[:, 1], marker='x')
        return ax_fig


class ReadRpt(object):

    def __init__(self, file_name=None, clean_data=False, proj_name='BDA'):
        self.call_name = file_name
        self.test_dur = []
        self.proj_name = proj_name
        self.channel_id = []
        self.data = pd.DataFrame()
        self.clean_data = clean_data
        self.eol_val = 75
        try:
            self.check_name_validity()
            print('Validity of file {} asserted'.format(self.call_name))
        except AssertionError as e:
            print(e)
        else:
            self.find_duration()
            self.find_ch_id()
            self.test_id = self.find_test_id()
            if isinstance(self.test_dur, int):
                self.test_name = '{}s'.format(self.test_dur)
            else:
                self.test_name = self.test_id
            self.read_data()
            self.calc_time_delta()
            # if self.data.shape[0] > 2:
            try:
                self.extrap_param = self.fit_extrapolation()
                self.exp_eol = self.find_eol(eol_val=self.eol_val)
                self.act_eol = self.find_eol_interpolation(eol_val=self.eol_val)
            except:
                print('Not able to fit eol')
            try:
                self.cell_id = self.check_cell_id()
                cell_id_nbr = re.sub("^0+(?!$)", "", re.search(r'\d+', self.cell_id).group())
                self.cell_id_pretty = f'Pulse duration {self.test_dur}s Cell_{cell_id_nbr}'
            except:
                self.cell_id = ''

    def check_cell_id(self):
        data_sheet = r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\Aline_BAD\Cell_Inventory\Tesla2170CellsFromVCC201909_Updated_2021_06_10.xlsx"
        df = pd.read_excel(data_sheet)
        return df[(df['Notes'] == 'BDA_Test') & (df['Channel'] == self.channel_id)]['Bar Code Number'].iloc[0]

    def check_name_validity(self):
        assert ((self.call_name.endswith('.xlsx') or self.call_name.endswith('.pkl'))
                and re.findall(r'[1-5]_[1-8]', self.call_name) and re.findall(r'summar', self.call_name)), \
            'File name {} does not follow expected pattern, AssertionError raised.'.format(self.call_name)
        return None

    def find_duration(self):
        if self.call_name.endswith('.xlsx'):
            dur_substr = re.findall(r'\d+_sec', self.call_name)[0]
            self.test_dur = int(re.findall(r'\d+', dur_substr)[0])
        elif self.call_name.endswith('.pkl'):
            self.find_ch_id()
            duration_str = self.look_up_test_name(self.channel_id, proj_name=self.proj_name)
            if 'second' in duration_str:
                self.test_dur = int(re.findall(r'\d+', duration_str)[0])
            else:
                self.test_dur = duration_str
        return None

    def find_test_id(self):
        self.find_ch_id()
        test_name = self.look_up_test_name(self.channel_id, proj_name=self.proj_name)
        return test_name

    def find_ch_id(self):
        try:
            self.channel_id = re.findall(r'\d+_[1-5]_[1-8]', self.call_name)[0]
        except IndexError:
            self.channel_id = re.findall(r'[1-5]_[1-8]', self.call_name)[0]

    def read_data(self):
        if self.call_name.endswith('.xlsx'):
            self.data = pd.read_excel(self.call_name, index_col=0)
        elif self.call_name.endswith('.pkl'):
            self.data = pd.read_pickle(self.call_name)
        if 'FCE' not in self.data.columns:
            self.data.loc[:, 'FCE'] = [look_up_fce(idx) for idx in self.data.index]
        if 'Unnamed: 0' in self.data.columns:
            self.data.set_index('Unnamed: 0', inplace=True)
        if self.clean_data:
            self.data = self.data[(self.data.cap.diff() < 0) | (self.data.cap.diff().isnull())]
        return self

    def calc_time_delta(self):
        dates = []
        for day in self.data['date'].astype(str):
            try:
                dates.append(dt.datetime.strptime(day, '%Y-%m-%d'))
            except ValueError:
                dates.append(dt.datetime.strptime('2020-02-10', '%Y-%m-%d'))
        # dates = [dt.datetime.strptime(day, '%Y-%m-%d') for day in self.data['date'].astype(str)]
        time_diff = [(day - dates[0]).days for day in dates]
        self.data['time_diff'] = time_diff
        return None

    def fit_extrapolation(self, x_mode='dchg'):
        from scipy.optimize import curve_fit
        df = self.data
        if x_mode.lower() == 'dchg':
            x_data = df['egy_dchg']
            try:
                popt, pcov = curve_fit(self.extrap_fun, x_data, df['cap'], p0=[1.5, 15, 4500])
            except ValueError:
                print('Value error found, due to NaN in data, should be dropped.')
                df.dropna(subset=['cap'], inplace=True)
                popt, pcov = curve_fit(self.extrap_fun, df['egy_dchg'], df['cap'], p0=[1.5, 15, 4500])
        else:
            x_data = df['time_diff']
            popt, pcov = curve_fit(self.extrap_fun, x_data, df['cap'], p0=[1., 50, 4500])
        return popt

    def find_eol(self, eol_val=75):
        if eol_val != self.eol_val:
            self.eol_val = eol_val
        from scipy.optimize import fsolve
        if not self.extrap_param.any():
            self.extrap_param = self.fit_extrapolation()
        zero_fun = lambda x, coeff: abs((eol_val / 100)*self.data['cap'].iloc[0] - self.extrap_fun(x, *coeff))
        x_zero = fsolve(zero_fun, x0=10, args=self.extrap_param)
        return x_zero

    def find_eol_interpolation(self, eol_val=75):
        if eol_val != self.eol_val:
            self.eol_val = eol_val
        from scipy.optimize import fsolve, minimize
        from scipy.interpolate import interp1d
        cap_interp = interp1d(self.data['egy_dchg'], self.data['cap_relative'], bounds_error=False,
                              fill_value='extrapolate')
        zero_fun = lambda x: cap_interp(x) - self.eol_val / 100
        x_zero = fsolve(zero_fun, x0=5)
        return x_zero

    def plot_extrapolation(self, x_mode='dchg', y_mode='limited'):
        if not self.extrap_param.any():
            self.extrap_param = self.fit_extrapolation()
        if x_mode.lower() == 'dchg':
            x_data = self.data['egy_dchg']
            x_label = 'Discharge energy throughput [kWh]'
        else:
            x_data = self.data['time_diff']
            x_label = 'Relative time [days]'
        fit_fig, ax = plt.subplots()
        scat = ax.scatter(x_data, self.data['cap'], label='Original data points', marker='x', color='blue')
        if y_mode.lower() == 'lim' or y_mode.lower() == 'limited':
            x_fit_data = np.linspace(x_data.min(), x_data.max(), 1000)
        else:
            x_fit_data = np.linspace(0, x_data.max()*6, 1000)
            y_intersect = self.eol_val / 100 * self.data.cap.max()
            x_intersect = self.find_eol(self.eol_val)
            ax.axhline(y_intersect, linewidth=0.8, c='blue')
            ax.axvline(x_intersect, linewidth=0.8, c='black')
            ax.text(x_intersect + 0.2, y_intersect + 10, 'Expected EOL \nfrom extrapolation',
                    {'va': 'bottom', 'ha': 'left', 'fontsize': 10})
        ax.plot(x_fit_data, self.extrap_fun(x_fit_data, *self.extrap_param), label='Fitted data', linewidth=0.8,
                linestyle='dashed', color='orange')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Capacity decay')
        plt.title('Extrapolation compared to original data for test {}s'.format(self.test_dur))
        plt.legend(loc='lower left')
        return fit_fig

    @staticmethod
    def extrap_fun(t, beta, tau, q0):
        return q0*np.exp(-(t/tau)**beta)

    def plot_data(self, x_mode='dchg', y_mode='cap', cell_id_label=False):
        if x_mode.lower() == 'dchg':
            x_data = self.data['egy_dchg']
            x_label = 'Discharge energy throughput [kWh]'
            plt.xlim([-0.10, 12])
        elif x_mode.lower() == 'chrg':
            x_data = self.data['egy_chrg']
            x_label = 'Charge energy throughput [kWh]'
        elif x_mode.lower() == 'tot':
            x_data = self.data['egy_thrg']
            x_label = 'Total energy throughput [kWh]'
        elif x_mode.lower() == 'time':
            x_data = self.data['time_diff']
            x_label = 'Relative time [days]'
        elif x_mode.lower() == 'cap_loss':
            x_data = 1 - self.data['cap_relative']
            x_label = 'Capacity lost'
        elif x_mode.lower() == 'cycles':
            x_data = self.data['FCE']
            x_label = 'Full cycle equivalents'
        else:
            print('Unknown x_mode provided, please re-run with correct mode: \n \'dchg\', \n \'chrg\', \n \'tot\', '
                  '\n \'time\', \n \'cycles\' or \n \'cap_loss\'')
            return

        if y_mode.lower() == 'cap':
            y_data = self.data['cap_relative']
            y_label = 'Relative capacity retention'
            plt.ylim([0.65, 1.01])
        elif y_mode.lower() == 'cap_abs':
            y_data = self.data['cap']
            y_label = 'Absolute capacity'
        elif y_mode.lower() == 'res_chrg':
            y_data = self.data['res_chrg_50_relative']
            y_label = 'Relative 50% SOC 10s charge resistance'
        elif y_mode.lower() == 'res_dchg':
            y_data = self.data['res_dchg_50_relative']
            y_label = 'Relative 50% SOC 10s discharge resistance'
        else:
            print('Unknown y_mode provided, please re-run with correct mode: \n \'cap\' or \n \'res_chrg\' or '
                  'or \n \'res_dchg\'')
            return
        if cell_id_label:
            label = self.cell_id_pretty
        else:
            label = '{}_cell{}'.format(self.test_name, self.channel_id)
        try:
            c = self.look_up_color(self.test_dur)
        except KeyError:
            print('No color specified for test based on pulse duration')
            c = self.look_up_color_storage(self.test_name)
        plt.plot(x_data,
                 y_data,
                 label=label,
                 marker='*',
                 linewidth=0.8,
                 color=c,
                 linestyle=self.set_linestyle())
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        return None

    def set_linestyle(self):
        chn = int(re.findall(r'\d', self.channel_id)[-1])
        if chn % 2 == 0:
            return 'solid'
        else:
            return 'dashed'

    @staticmethod
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
            256: 'indianred',
            3600: 'yellowgreen'
        }
        return color_dict[test_dur]

    @staticmethod
    def look_up_color_storage(test_name):
        storage_colors = {
            'Storage 15 SOC': 'mediumblue',
            'Storage 50 SOC': 'crimson',
            'Storage 85 SOC': 'forestgreen'
        }
        return storage_colors[test_name]

    @staticmethod
    def look_up_test_name(chan_key, proj_name='BDA'):
        try:
            if 'aline' in proj_name.lower():
                name_dict_aline = {
                    '1_1': 'Storage 15 SOC',
                    '1_2': 'Storage 15 SOC',
                    '1_3': 'Storage 50 SOC',
                    '1_4': 'Storage 50 SOC',
                    '1_5': 'Storage 85 SOC',
                    '240119_1_6': 'Storage 85 SOC',
                    '2_1': '5 to 15 SOC',
                    '2_2': '5 to 15 SOC',
                    '2_3': '15 to 25 SOC',
                    '240119_2_4': '15 to 25 SOC',
                    '240119_2_5': '25 to 35 SOC',
                    '2_6': '25 to 35 SOC',
                    '2_7': '35 to 45 SOC',
                    '2_8': '35 to 45 SOC',
                    '3_1': '45 to 55 SOC',
                    '3_2': '45 to 55 SOC',
                    '3_3': '55 to 65 SOC',
                    '3_4': '55 to 65 SOC',
                    '3_5': '65 to 75 SOC',
                    '3_6': '65 to 75 SOC',
                    '3_7': '75 to 85 SOC',
                    '3_8': '75 to 85 SOC',
                    '4_1': '85 to 95 SOC',
                    '4_2': '85 to 95 SOC',
                    '4_3': '0 to 100 SOC room temp',
                    '4_4': '0 to 100 SOC room temp',
                    '4_5': '50 to 100 SOC room temp',
                    '4_6': '50 to 100 SOC room temp',
                    '4_7': '0 to 50 SOC room temp',
                    '4_8': '0 to 50 SOC room temp',
                    '5_1': '0 to 50 SOC high temp',
                    '5_2': '0 to 100 SOC high temp',
                    '5_3': '0 to 50 SOC high temp',
                    '5_4': '50 to 100 SOC high temp',
                    '5_5': '50 to 100 SOC high temp',
                    '5_6': '0 to 100 SOC high temp',
                    'FCE': '3600 seconds'
                }
                return name_dict_aline[chan_key]
            elif 'bda_comp' in proj_name.lower():
                name_dict_bda_comp = {
                    '1_1': '2 seconds',
                    '1_2': '2 seconds',
                    '1_3': '4 seconds',
                    '1_4': '4 seconds',
                    '1_5': '2 seconds',
                    '1_6': '16 seconds',
                    '1_7': '64 seconds',
                    '1_8': '64 seconds',
                    '2_1': '256 seconds',
                    '2_2': '256 seconds',
                    '2_3': '256 seconds',
                    '2_4': '3600 seconds',
                    '2_5': 'Broken test',
                    '2_6': 'Broken test',
                    '2_7': '3600 seconds',
                    '2_8': '3600 seconds',
                    '3_1': '16 seconds'
                }
                return name_dict_bda_comp[chan_key]
            elif 'bda' in proj_name.lower():
                name_dict_bda = {
                    '1_1': '1 second',
                    '1_2': '1 second',
                    '1_3': '2 seconds',
                    '1_4': '2 seconds',
                    '1_5': '4 seconds',
                    '1_6': '4 seconds',
                    '1_7': '8 seconds',
                    '1_8': '8 seconds',
                    '2_1': '16 seconds',
                    '2_2': '16 seconds',
                    '2_3': '32 seconds',
                    '2_4': '32 seconds',
                    '2_5': '64 seconds',
                    '2_6': '64 seconds',
                    '2_7': '128 seconds',
                    '2_8': '128 seconds'
                }
                return name_dict_bda[chan_key]
            elif 'stat' in proj_name.lower():
                name_dict_stat = {
                    '240095_1_1': 'Test1_1',
                    '240095_1_2': 'Test1_1',
                    '240095_1_3': 'Test1_1',
                    '240095_1_4': 'Test1_1',
                    '240095_1_5': 'Test1_1',
                    '240095_1_6': 'Test1_1',
                    '240095_1_7': 'Test1_1',
                    '240095_1_8': 'Test1_1',
                    '240095_2_1': 'Test1_2',
                    '240095_2_2': 'Test1_2',
                    '240095_2_3': 'Test1_2',
                    '240095_2_4': 'Test1_2',
                    '240095_2_5': 'Test1_2',
                    '240095_2_6': 'Test1_2',
                    '240095_2_7': 'Test1_2',
                    '240095_2_8': 'Test1_2',
                    '240095_3_1': 'Test2_1',
                    '240095_3_2': 'Test2_1',
                    '240095_3_3': 'Test2_1',
                    '240095_3_4': 'Test2_1',
                    '240095_3_5': 'Test2_1',
                    '240095_3_6': 'Test2_1',
                    '240095_3_7': 'Test2_1',
                    '240095_3_8': 'Test2_1',
                    '240046_2_1': 'Test2_2',
                    '240046_2_2': 'Test2_2',
                    '240046_2_3': 'Test2_2',
                    '240046_2_4': 'Test2_2',
                    '240046_2_5': 'Test2_2',
                    '240046_2_6': 'Test2_2',
                    '240046_2_7': 'Test2_2',
                    '240046_2_8': 'Test2_2'
                }
                return name_dict_stat[chan_key]
            else:
                raise KeyError('Unknown unit used for test, update dictionaries')
        except KeyError:
            print('Channel not in list, return \'RPT\'')
            return 'RPT'


if __name__ == '__main__':
    fix_mpl_backend()
    ALINE_50DOD = OrganiseRpts(r"\\sol.ita.chalmers.se\groups\batt_lab_data\ALINE_data\50_dod_data", proj='aline')
    stat_test = OrganiseRpts(r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\cycling_data", proj='stat')
    test = OrganiseRpts(r"\\sol.ita.chalmers.se\groups\batt_lab_data\20210816", proj='bda_comp')
    clean_test = OrganiseRpts(r"\\sol.ita.chalmers.se\groups\batt_lab_data\20210816", clean_data=True, proj='bda_comp')
    all_tests = ['{}s'.format(num) for num in 2 ** np.arange(0, 8, 1)]
    all_rpts = ['rpt_{}'.format(num) for num in np.arange(1, 19, 1)]
    third_rpts = ['rpt_{}'.format(num) for num in np.arange(1, 19, 3)]
    fourth_rpts = ['rpt_{}'.format(num) for num in np.arange(1, 19, 4)]
    # avg_fig = test.plot_average_cap(savefig=True)
    ica_fig = test.plot_ica(test_name=['1s', '4s', '128s'], rpt_num=['rpt_1', 'rpt_9', 'rpt_17'],
                            cell_nbr=[2], savefig=True)
    ica_post = test.plot_ica(test_name=['1s', '128s'], rpt_num=['rpt_1', 'rpt_9', 'rpt_18'],
                             cell_nbr=[1], savefig=True)
    ah_fig_rpt1 = test.plot_ah_volt(test_name=['1s', '4s', '128s'], rpt_num=['rpt_1'],
                                    cell_nbr=[1], savefig=True)
    ah_fig_rpt12 = test.plot_ah_volt(test_name=['1s', '4s', '128s'], rpt_num=['rpt_12'],
                                     cell_nbr=[1], savefig=True)
    ica_64s = test.plot_ica(test_name=['64s'], rpt_num=third_rpts, cell_nbr=[1], savefig=True)
    #ah_fig_1s = test.plot_ah_volt(test_name=['1s'], rpt_num=['rpt_1', 'rpt_12'], cell_nbr=[1, 2])
    #ah_fig_all = test.plot_ah_volt(test_name=all_tests)
    soc_fig_first_cell = test.plot_soc_volt(test_name=['64s'], rpt_num=third_rpts, cell_nbr=[1], savefig=True)
    soc_fig_hysteresis = test.plot_soc_volt(test_name=['32s'], rpt_num=['rpt_1', 'rpt_6', 'rpt_12'],
                                            savefig=True, cell_nbr=[2])
    soc_fig_full = test.plot_soc_volt(test_name=['32s', '128s'], rpt_num=['rpt_1', 'rpt_18'],
                                      savefig=True, cell_nbr=[2])
    hyst_fig = test.plot_hysteresis(test_name=['64s'], rpt_num=third_rpts, cell_nbr=[1], savefig=False)
    comp_fig_dict = {}
    for test_name in all_tests:
        comp_fig_dict[test_name] = test.plot_rpt_data(test_name=test_name, cell_nbr=[1, 2], savefig=True)
    # for nbr in [1, 2, 3, 4]:
    #     ica_all = test.plot_ica(test_name=all_tests, rpt_num=['rpt_{}'.format(nbr)], cell_nbr=[1], savefig=True)
    # rpt_fig1 = test.plot_rpt_data(test_name=['1s', '2s', '4s', '8s'], cell_nbr=[1], savefig=True)
    # rpt_fig2 = test.plot_rpt_data(test_name=['16s', '32s', '64s', '128s'], cell_nbr=[1], savefig=True)
    rpt_fig3 = test.plot_rpt_data(test_name=['1s', '32s', '64s', '128s'], savefig=True)
    rpt_figx = test.plot_rpt_data(test_name=['1s', '4s', '32s', '128s'], savefig=True)
    #rpt_fig4 = clean_test.plot_rpt_data(test_name=all_tests, cell_nbr=[1], savefig=True)
    #if len(rpt_fig4.gca().lines) > 7:
    #    print('color reused')
    #    rpt_fig4.gca().lines[-1].set_color('black')
    #    [line.set_linewidth(0.7) for line in rpt_fig4.gca().lines]
    #    rpt_fig4.gca().legend()
    # rpt_fig5 = test.plot_rpt_data(test_name=['1s', '16s', '32s', '128s'], cell_nbr=[1], y_mode='res_dchg', savefig=True)
    # rpt_fig6 = test.plot_rpt_data(test_name=['1s', '16s', '32s', '128s'], cell_nbr=[1], y_mode='res_chrg', savefig=True)
    extrap_fig1 = test.summary_dict['64s_2_5'].plot_extrapolation(y_mode='limited')
    extrap_fig2 = test.summary_dict['64s_2_6'].plot_extrapolation(y_mode='limited')
    extrap_fig3 = test.summary_dict['128s_2_8'].plot_extrapolation(y_mode='limited')
    extrap_fig4 = test.summary_dict['1s_1_1'].plot_extrapolation(y_mode='limited')
    #extrap_fig4s1 = clean_test.summary_dict['4s_1_5'].plot_extrapolation(y_mode='limited')
    #extrap_fig4s2 = clean_test.summary_dict['4s_1_6'].plot_extrapolation(y_mode='limited')
    ax1 = extrap_fig1.gca()
    ax2 = extrap_fig2.gca()
    ax1.plot(ax2.lines[0].get_xdata(), ax2.lines[0].get_ydata(), color='brown', linewidth=0.8)
    scat = ax2.get_children()[0]
    ax1.scatter(scat.get_offsets()[:, 0], scat.get_offsets()[:, 1], marker='x', color='black')
    ax1.legend(['Fit data cell1', 'Raw data cell1', 'Fit data cell2', 'Raw data cell2'])
    # extrap_fig1.savefig(os.path.join(test.analysis_dir, '64s_extrapolation.png'), dpi=1200)
    # extrap_fig2.savefig(os.path.join(test.analysis_dir, '64s_cell2_extrapolation.png'), dpi=1200)
    eol_fig = test.plot_eol(mode='avg', data_use='interpolation', savefig=True)
    #clean_eol = clean_test.plot_eol(mode='full', data_use='extrapolation', savefig=True)
    eol_full_int = test.plot_eol(mode='full', data_use='interpolation', savefig=True)
    #comb_clean = clean_test.combine_plots([extrap_fig4s1, extrap_fig4s2])
    all_keys = list(test.summary_dict.keys())
    pulse_time = ['{}s'.format(t) for t in 2**np.arange(0, 8, 1)]
    consistent_results = ['1s', '8s', '32s', '128s']
    consistent_plot = test.plot_average_cap(test_name=consistent_results, savefig=True)
    test_points = test.summary_dict.keys()
    for pt in pulse_time:
        res_fig = test.plot_rpt_data(test_name=[pt], cell_nbr=[1, 2, 3], savefig=False, y_mode='res_dchg')
        test_list = [d_set for d_set in test_points if pt in d_set]
        lines = [line for line in res_fig.gca().lines]
        for data_set in test_list:
            if test.summary_dict[data_set].test_name == pt:
                for ln in lines:
                    if test.summary_dict[data_set].channel_id in ln.get_label():
                        res_fig.gca().plot(test.summary_dict[data_set].data['egy_dchg'],
                                           test.summary_dict[data_set].data['cap_relative'],
                                           linewidth=0.8, linestyle='dashed', marker=ln.get_marker(),
                                           color=ln.get_c(), label='{}_capacity'.format(ln.get_label()))
        plt.legend()
        res_fig.savefig(os.path.join(test.analysis_dir, 'res_and_cap_for_{}.png'.format(pt)), dpi=1200)
        # res_v_cap_fig = test.plot_rpt_data(test_name=[pt], cell_nbr=[1, 2], savefig=True, x_mode='cap_loss',
        #                                    y_mode='res_dchg', plot_title='Capacity loss plotted vs resistance increase for {} pulse duration'.format(pt))
        # dva_fig = test.plot_dva(test_name=[pt], rpt_num=['rpt_1'],
        #                         cell_nbr=[1], savefig=True)
    #     hyst_fig = test.plot_hysteresis(test_name=[pt], rpt_num=['rpt_1', 'rpt_9', 'rpt_17'],
    #                                     cell_nbr=[1, 2], savefig=True)
    #     ica_fig = test.plot_ica(test_name=[pt], rpt_num=['rpt_1', 'rpt_9', 'rpt_17'],
    #                             cell_nbr=[1], savefig=True)
    eis_set = [f"{t}s" for t in [1, 16, 64, 128]]
    test.plot_rpt_data(test_name=eis_set, cell_nbr=[1], x_mode='cycles', savefig=True, cell_id_label=True)
    test.plot_rpt_data(test_name=all_tests, cell_nbr=[1, 2], x_mode='cycles', savefig=True, cell_id_label=True)
    plt.close('all')
