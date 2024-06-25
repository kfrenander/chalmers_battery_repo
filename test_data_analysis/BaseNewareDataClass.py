import pandas as pd
from scipy.integrate import cumtrapz
import datetime as dt
import numpy as np
import re
from natsort.natsort import natsorted
import os
from collections import OrderedDict


class BaseNewareDataSet(object):

    def __init__(self, data_dir):
        print(f'Running analysis for directory {data_dir}')
        # print('Initiating data for directory {:%Y-%m-%d}, please hold'.format(dt.datetime.strptime(
        #     re.findall(r'\d+', os.path.split(dir)[-1])[0], '%Y%m%d')))
        self.data_dict = {}
        self.chan_file_dict = {}
        self.sort_files_to_channel(data_dir)
        print('Files have been sorted')

    def sort_files_to_channel(self, data_dir):
        file_list = os.listdir(data_dir)
        chan_list = []
        for f in file_list:
            if f.endswith('.xls') or f.endswith('.xlsx'):
                unit = re.search(r'\d{6}', f).group()
                chan_list.append('{2}-{0}-{1}'.format(*[x.strip("-") for x in re.findall(r'-\d+', f) if len(x) < 3],
                                                      unit))
        chan_list = list(set(chan_list))
        self.chan_file_dict = {k: [os.path.join(data_dir, x) for x in file_list if f'{k}-' in x] for k in
                               chan_list}
        return None

    def write_summaries(self):
        for entry in self.data_dict:
            if not isinstance(self.data_dict[entry], str):
                self.data_dict[entry].write_rpt_summary()
        return None


class BaseNewareData(object):

    def __init__(self, list_of_files, n_cores=8):
        import pickle
        self.file_names = list_of_files
        self.ica_step_list = []
        self.pkl_dir = ''
        self.machine_id = ''
        self.channel_id = ''
        self.meta_data = None
        self.unit_name = self.find_unit_name()
        self.machine_id, self.channel_id = self.find_machine_and_channel_id()
        self.channel_name = self.make_channel_name()
        self.set_pkl_dir()
        self.test_id = self.find_test_id()
        # self.channel_name = list(set(['{}_{}_{}'.format(*[x.strip('-') for x in re.findall(r'\d+-', file)])
        #                              for file in list_of_files]))[0]

    def pickle_data_dump(self):
        if not self.pkl_dir:
            self.set_pkl_dir()
        op_dir = self.pkl_dir
        rpt_dict = self.find_rpt_dict()
        if not os.path.isdir(op_dir):
            os.makedirs(op_dir)
        try:
            for key in self.rpt_analysis.ica_dict:
                ica_file = f'{self.channel_name}_ica_dump_{key}.pkl'
                res_file = f'{self.channel_name}_res_dump_{key}.pkl'
                cap_file = f'{self.channel_name}_cap_dump_{key}.pkl'
                rpt_file = f'{self.channel_name}_rpt_raw_dump_{key}.pkl'
                self.rpt_analysis.ica_dict[key].to_pickle(os.path.join(op_dir, ica_file))
                self.rpt_analysis.res[key].to_pickle(os.path.join(op_dir, res_file))
                self.rpt_analysis.cap[key].to_pickle(os.path.join(op_dir, cap_file))
                rpt_dict[key].to_pickle(os.path.join(op_dir, rpt_file))
        except:
            print('Unable to pickle ICA and/or res')
        rpt_summary_pickle = f'{self.channel_name}_rpt_summary_dump.pkl'
        dynamic_pickle = f'{self.channel_name}_dyn_df_dump.pkl'
        char_pickle = f'{self.channel_name}_step_char_dump.pkl'
        self.rpt_analysis.summary.to_pickle(os.path.join(op_dir, rpt_summary_pickle))
        self.dyn_df.to_pickle(os.path.join(op_dir, dynamic_pickle))
        self.step_char.to_pickle(os.path.join(op_dir, char_pickle))
        return None

    def make_meta_data_dict(self):
        self.meta_data = {
            'UNIT': self.unit_name,
            'MACHINE': self.machine_id,
            'CHANNEL': self.channel_id,
            'TEST': self.test_id
        }

    def write_meta_data(self):
        if not self.meta_data:
            self.make_meta_data_dict()
        if not self.pkl_dir:
            self.set_pkl_dir()
            if not os.path.isdir(self.pkl_dir):
                os.makedirs(self.pkl_dir)
        fname = os.path.join(self.pkl_dir, f'metadata_{self.channel_name}_test_{self.test_id}.txt')
        with open(fname, 'w+') as f:
            for key, value in self.meta_data.items():
                f.write(f'{key}_ID\t\t\t: {value}\n')
        print(f'Metadata written for unit {self.unit_name} and channel {self.machine_id}_{self.channel_id}')

    def set_pkl_dir(self):
        self.pkl_dir = os.path.join(os.path.split(self.file_names[0])[0], f'pickle_files_channel_{self.channel_name}')
        return None

    def step_characteristics(self, n_cores=8):
        from test_data_analysis.rpt_analysis import characterise_steps
        from multiprocessing import Pool
        char_start = dt.datetime.now()
        df_split = np.array_split(self.dyn_df, n_cores)
        pool = Pool(n_cores)
        self.step_char = pd.concat(pool.map(characterise_steps, df_split))
        pool.close()
        pool.join()
        self.fix_split_errors()
        print('Time for characterisation was {} seconds.'.format((dt.datetime.now() - char_start).seconds))
        return None

    def fix_split_errors(self):
        self.step_char.reset_index(inplace=True, drop=True)
        idx_to_drop = self.step_char[self.step_char.step_nbr.duplicated()].index - 1
        self.step_char.drop(idx_to_drop, axis=0, inplace=True)
        self.step_char.reset_index(inplace=True, drop=True)
        return None

    def find_unit_name(self):
        unit_list = [re.search(r'\d{6}-', f).group() for f in self.file_names]
        if len(list(set(unit_list))) > 1:
            raise ValueError("More than one unit found in test list, check data")
        return list(set(unit_list))[0].strip('-')

    def find_machine_and_channel_id(self):
        # Regex pattern that looks for both leading AND trailing hyphen
        pattern = r'(?<=-)\d+(?=-)'
        all_cases = [re.findall(pattern, f) for f in self.file_names]
        unique_lists = list(OrderedDict.fromkeys(tuple(lst) for lst in all_cases))
        # Convert back to list of lists
        unique_list = [list(tpl) for tpl in unique_lists][0]
        machine_id = unique_list[0]
        channel_id = unique_list[1]
        return machine_id, channel_id

    def make_channel_name(self):
        return f'{self.unit_name}_{self.machine_id}_{self.channel_id}'

    def find_test_id(self):
        # Regex pattern to identify only the trailing umber preceded by hyphen
        pattern = r'(?<=-)\d+(?=[^0-9]*$)'
        # Regex pattern will only work if trailing number in file name can be removed, this requires the path to be
        # reduced to only file name
        fnames = [os.path.split(f)[-1] for f in self.file_names]
        list_of_ids = [re.search(pattern, fname.split('_')[0]).group() for fname in fnames]
        return np.unique(list_of_ids)[0]

    def read_dynamic_data(self):
        df = pd.DataFrame()
        temperature_df = pd.DataFrame()
        col_names = ['Measurement', 'mode', 'step', 'arb_step1', 'arb_step2',
                     'curr', 'volt', 'cap', 'step_egy', 'rel_time', 'abs_time']
        col_names_temp = ['Measurement', 'mode', 'rel_time', 'abs_time', 'temperature', 'aux_temp']
        for xl_file in self.xl_files:
            for sheet in xl_file.sheet_names:
                if 'detail_' in sheet.lower():
                    if df.empty:
                        df = xl_file.parse(sheet, names=col_names)
                        if df['curr'].abs().max() > 10000:
                            df.loc[:, 'curr'] = df.loc[:, 'curr'] / 10
                            df.loc[:, 'cap'] = df.loc[:, 'cap'] / 10
                            df.loc[:, 'step_egy'] = df.loc[:, 'step_egy'] / 10
                    else:
                        temp_df = xl_file.parse(sheet, names=col_names)
                        if temp_df['curr'].abs().max() > 10000:
                            temp_df.loc[:, 'curr'] = temp_df.loc[:, 'curr'] / 10
                            temp_df.loc[:, 'cap'] = temp_df.loc[:, 'cap'] / 10
                            temp_df.loc[:, 'step_egy'] = temp_df.loc[:, 'step_egy'] / 10
                        df = df.append(temp_df, ignore_index=True)
                elif 'detailtemp' in sheet.lower():
                    if temperature_df.empty:
                        temperature_df = xl_file.parse(sheet, names=col_names_temp)
                    else:
                        tdf = xl_file.parse(sheet, names=col_names_temp)
                        temperature_df = temperature_df.append(tdf, ignore_index=True)
        mA = [x for x in df.columns if '(mA)' in x]
        if df['curr'].abs().max() > 10000:
            df.loc[:, 'curr'] = df.loc[:, 'curr'] / 10
            df.loc[:, 'cap'] = df.loc[:, 'cap'] / 10
            df.loc[:, 'step_egy'] = df.loc[:, 'step_egy'] / 10
        df['mode'].replace(['搁置', '恒流充电', '恒流恒压充电', '恒流放电'],
                           ['Rest', 'CC_Chg', 'CCCV_Chg', 'CC_DChg'], inplace=True)
        df['mode'].replace(['Rest', 'CC Chg', 'CC DChg', 'CCCV Chg'],
                           ['Rest', 'CC_Chg', 'CC_DChg', 'CCCV_Chg'], inplace=True)
        df['step_time'] = pd.to_timedelta(df.rel_time)
        df['abs_time'] = pd.to_datetime(df['abs_time'], format='%Y-%m-%d %H:%M:%S')
        df['float_time'] = (df.abs_time - df.abs_time[0]).dt.total_seconds().astype(float)
        try:
            df.loc[:, 'temperature'] = temperature_df.loc[:, 'temperature']
        except:
            print('No temperature measurement available')
        df['pwr'] = df.curr / 1000 * df.volt
        df['pwr_chrg'] = df.pwr.mask(df.pwr < 0, 0)
        df['pwr_dchg'] = df.pwr.mask(df.pwr > 0, 0)
        df['egy_tot'] = cumtrapz(df.pwr.abs() / (1000*3600), df.float_time, initial=0)
        df['egy_chrg'] = cumtrapz(df.pwr_chrg.abs() / (1000*3600), df.float_time, initial=0)
        df['egy_dchg'] = cumtrapz(df.pwr_dchg.abs() / (1000*3600), df.float_time, initial=0)
        if not df.arb_step2.is_monotonic_increasing:
            df = self.sum_idx(df, 'arb_step2')
            # reset_idx = df.arb_step2[df.arb_step2.diff() < 0].index.values[0]
            # df.loc[reset_idx:, 'arb_step2'] = df.loc[reset_idx:, 'arb_step2'] + df.loc[reset_idx - 1, 'arb_step2']
            # df.loc[reset_idx:, 'Measurement'] = df.loc[reset_idx:, 'Measurement'] + df.loc[reset_idx - 1, 'Measurement']
        # df = df.sort_values(by='Measurement')
        if df['curr'].abs().max() > 100:
            df['mAh'] = cumtrapz(df.curr, df.float_time, initial=0) / 3600
            df['curr'] = df.curr / 1000
        else:
            df['mAh'] = cumtrapz(df.curr, df.float_time, initial=0) * 1000 / 3600
            df['cap'] = df['cap'] * 1000
        self.dyn_df = df
        return self

    def read_cycle_statistics(self):
        for xl_file in self.xl_files:
            for sheet in xl_file.sheet_names:
                if 'statis' in sheet.lower():
                    if self.stat.empty:
                        col_names = [
                            'Channel',
                            'CyCle',
                            'Step',
                            'Raw Step ID',
                            'Status',
                            'Step Voltage(V)',
                            'End Voltage(V)',
                            'Start Current(mA)',
                            'End Current(mA)',
                            'CapaCity(mAh)',
                            'Endure Time(h:min:s.ms)',
                            'Relative Time(h:min:s.ms)',
                            'Absolute Time',
                            'Discharge_Capacity(mAh)',
                            'Charge_Capacity(mAh)',
                            'Discharge_Capacity(mAh)',
                            'Net Engy_DChg(mWh)',
                            'Engy_Chg(mWh)',
                            'Engy_DChg(mWh)'
                        ]
                        self.stat = xl_file.parse(sheet, names=col_names)
                        print('Statistics df intiated')
                    else:
                        print('Statistics df appended')
                        temp_df = xl_file.parse(sheet)
                        col_match = {x: y for x, y in zip(temp_df.columns, self.stat.columns)}
                        self.stat = self.stat.append(temp_df.rename(columns=col_match), ignore_index=True)
                    t_dur = self.stat['Endure Time(h:min:s.ms)'].apply(lambda x: self.calc_t_delta(x))
                    self.stat['t_dur'] = t_dur
                    curr_diff = (self.stat['Start Current(mA)'] - self.stat['End Current(mA)']).abs()
                    self.stat['curr_diff'] = curr_diff
                    self.stat['Status'].replace(['搁置', '恒流充电', '恒流恒压充电', '恒流放电'],
                                                ['Rest', 'CC_Chg', 'CCCV_Chg', 'CC_DChg'], inplace=True)
                    self.stat = self.sum_idx(self.stat, 'Step')
                if 'cycle' in sheet.lower():
                    cyc_names = ['channel', 'cycle', 'chrg_cap', 'dchg_cap', 'cap_decay']
                    if self.cyc.empty:
                        self.cyc = xl_file.parse(sheet, names=cyc_names)
                    else:
                        temp_df = xl_file.parse(sheet, names=cyc_names)
                        col_match = {x: y for x, y in zip(temp_df.columns, self.cyc.columns)}
                        self.cyc = self.cyc.append(temp_df.rename(columns=col_match), ignore_index=True)
                    # self.cyc.rename(columns={
                    #     '通道': 'Channel',
                    #     '循环序号': 'ToTal of Cycle',
                    #     '充电容量(mAh)': ' Capacity of charge(mAh)',
                    #     '放电容量(mAh)': 'Capacity of discharge(mAh)',
                    #     '放电容量衰减率（%）': 'Cycle Life(%)'
                    # }, inplace=True)
                    # self.cyc.rename(columns={'ToTal of Cycle': 'cycle'}, inplace=True)
                    self.cyc = self.sum_idx(self.cyc, 'cycle')
        return self

    @staticmethod
    def sum_idx(df, col):
        reset_idx = df.where(df[col].diff() < 0).dropna().index.to_list()
        if reset_idx:
            increment = df.loc[[idx - 1 for idx in reset_idx], col]
            for t in zip(reset_idx, increment):
                df.loc[t[0]:, col] = df.loc[t[0]:, col] + t[1]
        return df

    def find_rpt_dict(self):
        # Find rest steps which have rpt duration (15 minutes).
        start_idx = self.step_char[(self.step_char.step_duration == 28) &
                                   (self.step_char.step_mode == 'Rest')].index
        stop_idx = self.step_char[(self.step_char.step_duration == 34) &
                                  (self.step_char.step_mode == 'Rest')].index
        if stop_idx.empty:
            rpt_rest_steps = self.step_char[(self.step_char.step_mode == 'Rest') & (self.step_char.step_duration == 900)]
            # Find discharge steps that are long enough to be RPT and C/3 (only present in RPT)
            long_dchg_steps = self.step_char[(self.step_char.step_duration > 4000) & (abs(self.step_char.curr + 1.53) < 0.2)]
            pulse_chrg_steps = self.step_char[(self.step_char.curr > 5.5) & (self.step_char.step_duration == 10)]

            # From rest steps, find the ones that are far enough apart to signify new rpt has started
            new_rpt_step = rpt_rest_steps[rpt_rest_steps.step_nbr.diff() > 90].step_nbr.tolist()
            new_rpt_step.append(self.dyn_df.arb_step2.max())
            stop_idx = [int(rpt_rest_steps[rpt_rest_steps.step_nbr < int(val)].last_valid_index()) for val in new_rpt_step]

            pulse_idx = pulse_chrg_steps.index.to_numpy()
            # dchg_30_soc_idx = self.step_char.loc[pulse_idx - 1, :][self.step_char.loc[pulse_idx - 1, 'minV'] < 3.46].index
            dchg_30_soc_idx = self.step_char.loc[pulse_idx - 4, :][self.step_char.loc[pulse_idx - 4, 'minV'] < 3.5].index
            stop_idx = dchg_30_soc_idx + 7


            # Similary find the long discharge steps far enough apart.
            # start_idx = long_dchg_steps[long_dchg_steps.step_nbr.diff().fillna(200) > 100].step_nbr.values.tolist()
            start_idx = long_dchg_steps[long_dchg_steps.stp_date.diff().fillna(dt.timedelta(days=15)) >
                                        dt.timedelta(days=3)].index

        stop_idx = stop_idx.drop_duplicates()
        start_idx = start_idx.drop_duplicates()


        # Since some tests do not properly start from scratch the first found RPT might be the second, so this must be
        # kept despite not fulfilling the diff requirement.
        if start_idx[0] > stop_idx[0]:
            start_idx.insert(0, int(long_dchg_steps.first_valid_index()))

        # All tests should start with rpt
        # if start_idx[0] > 10:
        #     start_idx.insert(0, 0)

        # Some test show inexplicable c/3 discharges and 15 minute rests that must sorted out
        for i, idx in enumerate(start_idx):
            if stop_idx[i] - start_idx[i] < 20:
                stop_idx.pop(i)
                start_idx.pop(i)

        # Sample out the dynamic data between each of the start and stop indices
        rpt_dict = {'rpt_{}'.format(i + 1): self.dyn_df[(self.dyn_df.arb_step2 > start_idx[i]) &
                                                        (self.dyn_df.arb_step2 < stop_idx[i])]
                    for i in range(len(stop_idx)) if stop_idx[i] - start_idx[i] > 20}
        rpt_dict = {k: v for k, v in rpt_dict.items() if not v.empty}
        return rpt_dict

    def find_ica_steps(self):
        self.ica_step_list = self.step_char[self.step_char.step_duration > 10 * 3600].step_nbr.tolist()

    def write_rpt_summary(self):
        op_dir = os.path.join(os.path.split(self.file_names[0])[0], 'rpt_summaries')
        if not os.path.isdir(op_dir):
            os.makedirs(op_dir)
        op_file = '{}_pulse_time_channel_{}.xlsx'.format(self.test_name, self.channel_name).replace(' ', '_')
        self.rpt_analysis.summary.to_excel(os.path.join(op_dir, op_file))
        return None

    @staticmethod
    def calc_t_delta(my_time):
        fmt = '%H:%M:%S.%f'
        t = dt.datetime.strptime(my_time, fmt)
        return dt.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)


class BaseRptData(object):

    def __init__(self, rpt_dict, ica_step_list):
        from test_data_analysis.rpt_analysis import find_step_characteristics, find_res
        from test_data_analysis.capacity_test_analysis import find_cap_meas
        self.rpt_dict = rpt_dict
        # self.sort_individual_rpt(rpt_df)
        self.char_dict = {key: find_step_characteristics(self.rpt_dict[key]) for key in self.rpt_dict}
        self.res = {key: find_res(self.rpt_dict[key], self.char_dict[key]) for key in self.rpt_dict}
        self.egy = {key: self.char_dict[key].filter(like='egy').iloc[0] for key in self.rpt_dict}
        self.date = {key: '{:%Y-%m-%d}'.format(
            self.rpt_dict[key].abs_time[self.rpt_dict[key].abs_time.first_valid_index()]) for key in self.rpt_dict}
        self.ica_dict = {key: self.rpt_dict[key][self.rpt_dict[key].arb_step2.isin(ica_step_list)]
                         for key in self.rpt_dict}
        self.ica_analysis()

    def sort_individual_rpt(self, df):
        self.rpt_dict = {'rpt_{:.0f}'.format(marks): df[df.mark == marks] for marks in df.mark.unique()}
        return None

    def ica_analysis(self):
        from test_data_analysis.ica_analysis import perform_ica
        for key in self.ica_dict:
            ica_df = self.ica_dict[key]
            try:
                gb = ica_df.groupby('step')
                ica_dchg = [perform_ica(gb.get_group(x)) for x in gb.groups if gb.get_group(x).curr.mean() < 0][0]
                ica_chrg = [perform_ica(gb.get_group(x)) for x in gb.groups if gb.get_group(x).curr.mean() > 0][0]
                processed_df = pd.concat([ica_chrg, ica_dchg])
            except IndexError:
                print('Something failed when calculating gradient')
            except ValueError as e:
                print('ICA for this RPT probably empty')
                print(e)
            self.ica_dict[key] = processed_df
        return self

    def create_cell_summary(self):
        # summary_df = pd.DataFrame(columns=['date', 'cap', 'res_dchg_50', 'res_chrg_50'])
        date_df = pd.DataFrame(data=[self.date[key] for key in self.date if self.date], columns=['date'])
        dchg_df = pd.DataFrame(data=[self.res[key].loc['soc_50', 'R10_dchg'] for key in self.res
                                     if 'soc_50' in self.res[key].index], columns=['res_dchg_50'])
        chrg_df = pd.DataFrame(data=[self.res[key].loc['soc_50', 'R10_chrg'] for key in self.res
                                     if 'soc_50' in self.res[key].index], columns=['res_chrg_50'])
        cap_df = pd.DataFrame(data=[(self.cap[key].loc['mean', 'cap'], self.cap[key].loc['var_normed', 'cap'])
                                    for key in self.cap if not self.cap[key].empty],
                              columns=['cap', 'sigma_cap'])
        egy_df = pd.DataFrame(data=[self.egy[key] for key in self.egy if not self.egy[key].empty],
                              columns=self.egy[list(self.egy.keys())[0]].index, index=[i for i in range(len(self.egy))])
        summary_df = pd.concat([date_df, cap_df, dchg_df, chrg_df, egy_df], axis=1)
        for entry in summary_df:
            if not entry == 'date' and 'egy' not in entry:
                summary_df['{}_relative'.format(entry)] = summary_df[entry] / summary_df[entry][0]
            summary_df.index = [key for key in self.rpt_dict if self.rpt_dict]
        return summary_df

    @staticmethod
    def flatten_df(df):
        s = df.stack()
        df1 = pd.DataFrame([s.values], columns=[f'{j}-{i}' for i, j in s.index])
        return df1


if __name__ == '__main__':
    test_case = BaseNewareDataSet(r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\cycling_data")
