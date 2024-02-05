from test_data_analysis.BaseNewareDataClass import BaseNewareDataSet, BaseNewareData
from test_data_analysis.NewareDataTesla import TeslaRptData
from test_data_analysis.rpt_analysis import characterise_steps
import os
import re
import datetime as dt
import pandas as pd


class NewareDataSetAline(BaseNewareDataSet):

    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.fill_data()

    def sort_files_to_channel(self, data_dir):
        file_list = os.listdir(data_dir)
        chan_list = [os.path.splitext(f)[0] for f in file_list if f.endswith('xlsx')]
        # for f in file_list:
        #     if f.endswith('.xls') or f.endswith('.xlsx'):
        #         unit = re.search(r'\d{6}', f).group()
        #         m_c_list = [x.strip("-") for x in re.findall(r'-\d+', f) if len(x) < 3]
        #         test_id = re.search(r'\d{10}', f).group()
        #         machine = m_c_list[0]
        #         channel = m_c_list[1]
        #         chan_list.append(f'{unit}-{machine}-{channel}-{test_id}')
        # chan_list = list(set(chan_list))
        self.chan_file_dict = {k: [os.path.join(data_dir, f) for f in file_list if f'{k}' in f] for k in
                               chan_list}
        return None

    def fill_data(self):
        temp_dict = {}
        for key, file_list in self.chan_file_dict.items():
            root_dir = os.path.split(file_list[0])[0]
            chan_id = re.findall(r'\b\d\b', key)
            chan_number = '_'.join(chan_id)
            exp_name = f'pickle_files_channel_{chan_number}'
            print('Calling Neware data with {}'.format(key))
            tic = dt.datetime.now()
            pkl_dir = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if name == exp_name]
            if os.path.exists(os.path.join(root_dir, exp_name)):
                print('Files already read. \n Read pickle dumps instead')
            else:
                try:
                    temp_dict[key] = NewareDataAline(file_list)
                except OSError as e:
                    print('Probably not enough memory')
                    print(e)
                    temp_dict[key] = 'Placeholder due to OSError'
                except MemoryError:
                    print('Running out of RAM')
                    temp_dict[key] = 'Placeholder due to MemoryError'
            toc = dt.datetime.now()
            print('Time elapsed for test {} was {:.2f} min.'.format(key, (toc - tic).total_seconds() / 60))
        self.data_dict = temp_dict
        return None


class NewareDataAline(BaseNewareData):
    def __init__(self, list_of_files, n_cores=8):

        import numpy as np
        from natsort.natsort import natsorted
        from multiprocessing import Pool
        super().__init__(list_of_files, n_cores)
        self.channel_name = ['{}_{}_{}'.format(*[x.strip('-') for x in re.findall(r'\d+-', file)])
                                      for file in list_of_files][0]
        self.test_name = [re.findall(r'\d{10}', f)[0] for f in list_of_files][0]
        self.test_case = self.look_up_test_name()
        self.cell_id = re.search(r'cell\d+', self.test_case).group()
        self.op_fname = self.test_case.replace(" ", "_")
        self.stat = pd.DataFrame()
        self.cyc = pd.DataFrame()
        print('Starting read in')
        self.xl_files = [pd.ExcelFile(file_name) for file_name in natsorted(self.file_names)]
        print('Read in of data to pandas finished')
        self.read_dynamic_data()
        super().read_cycle_statistics()
        super().step_characteristics(n_cores=n_cores)
        super().find_ica_steps()
        self.rpt_analysis = TeslaRptData(self.find_rpt_dict(), self.ica_step_list)
        self.pickle_data_dump()
        self.write_rpt_summary()
        self.write_test_info()

    def write_test_info(self):
        test_info_file = os.path.join(self.pkl_dir, 'test_info.txt')
        read_me_file = os.path.join(self.pkl_dir, 'README.txt')
        with open(test_info_file, 'w') as f_:
            f_.write(f'TEST CASE: {self.test_case} \n')
            f_.write(f'CELL ID: {self.cell_id} \n')
            f_.write(f'TEST NBR: {self.test_name} \n')
            f_.write(f'CHANNEL ID: {self.channel_name}\n')
            f_.write(f'TEST DATE: {self.step_char.stp_date.iloc[0].strftime("%Y-%m-%d")}')
        return None

    def look_up_test_name(self):
        if self.test_name == '2818575215':
            data_id = [f'240095_1_{k}' for k in range(1, 9)]
            test_soc = ['5-15', '15-25', '15-25', '25-35', '25-35', '35-45', '35-45', '85-95']
            cell_id = [680, 672, 691, 681, 683, 694, 697, 660]
            test_id = [f'{tn}% SOC_cell{cid}' for tn, cid in zip(test_soc, cell_id)]
            test_id_dict = dict(zip(data_id, test_id))
            return test_id_dict[self.channel_name]
        elif self.test_name == '2818575216':
            data_id = [f'240095_1_{k}' for k in range(1, 9)]
            test_soc = ['55-65', '65-75', '65-75', '55-65', '75-85', '75-85', '45-55', '45-55']
            cell_id = [665, 666, 667, 674, 677, 678, 686, 695]
            test_id = [f'{tn}% SOC_cell{cid}' for tn, cid in zip(test_soc, cell_id)]
            test_id_dict = dict(zip(data_id, test_id))
            return test_id_dict[self.channel_name]

    def read_dynamic_data(self):
        from scipy.integrate import cumtrapz
        df = pd.DataFrame()
        temperature_df = pd.DataFrame()
        col_names = ['Measurement', 'mode', 'step', 'arb_step1', 'arb_step2',
                     'curr', 'volt', 'cap', 'step_egy', 'rel_time', 'abs_time']
        col_names_temp = ['Measurement', 'mode', 'rel_time', 'abs_time', 'temperature', 'aux_temp', 'aux_temp2']
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
                        temperature_df = self.pick_temp_measurement(temperature_df)
                    else:
                        tdf = xl_file.parse(sheet, names=col_names_temp)
                        tdf = self.pick_temp_measurement(tdf)
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
        df['float_time'] = (df.abs_time - df.abs_time[0]).astype('timedelta64[s]').dt.total_seconds()
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

    def pickle_data_dump(self):
        self.pkl_dir = os.path.join(os.path.split(self.file_names[0])[0],
                                    f'pickle_files_channel_{self.op_fname}')
        rpt_dict = self.find_rpt_dict()
        if not os.path.isdir(self.pkl_dir):
            os.makedirs(self.pkl_dir)
        try:
            for key in self.rpt_analysis.ica_dict:
                ica_file = f'{self.channel_name}_{self.op_fname}_ica_dump_{key}.pkl'
                res_file = f'{self.channel_name}_{self.op_fname}_res_dump_{key}.pkl'
                cap_file = f'{self.channel_name}_{self.op_fname}_cap_dump_{key}.pkl'
                rpt_file = f'{self.channel_name}_{self.op_fname}_rpt_raw_dump_{key}.pkl'
                self.rpt_analysis.ica_dict[key].to_pickle(os.path.join(self.pkl_dir, ica_file))
                self.rpt_analysis.res[key].to_pickle(os.path.join(self.pkl_dir, res_file))
                self.rpt_analysis.cap[key].to_pickle(os.path.join(self.pkl_dir, cap_file))
                rpt_dict[key].to_pickle(os.path.join(self.pkl_dir, rpt_file))
        except:
            print('Unable to pickle ICA and/or res')
        rpt_summary_pickle = f'{self.channel_name}_{self.op_fname}_rpt_summary_dump.pkl'
        dynamic_pickle = f'{self.channel_name}_{self.op_fname}_dyn_df_dump.pkl'
        char_pickle = f'{self.channel_name}_{self.op_fname}_step_char_dump.pkl'
        self.rpt_analysis.summary.to_pickle(os.path.join(self.pkl_dir, rpt_summary_pickle))
        self.dyn_df.to_pickle(os.path.join(self.pkl_dir, dynamic_pickle))
        self.step_char.to_pickle(os.path.join(self.pkl_dir, char_pickle))
        return None

    def write_rpt_summary(self):
        op_dir = os.path.join(os.path.split(self.file_names[0])[0], 'rpt_summaries')
        if not os.path.isdir(op_dir):
            os.makedirs(op_dir)
        op_file = f'{self.op_fname}.xlsx'
        self.rpt_analysis.summary.to_excel(os.path.join(op_dir, op_file))
        return None

    def pick_temp_measurement(self, temperature_df):
        tmp_df = temperature_df.filter(like='temp').mean()
        idx_to_drop = tmp_df[tmp_df > 100].index
        rename_index = tmp_df[tmp_df < 100].index[0]
        op_df = temperature_df.drop(idx_to_drop, axis=1).rename(columns={rename_index: 'temperature'})
        return op_df


if __name__ == '__main__':
    aline_after_rest = r"\\sol.ita.chalmers.se\groups\batt_lab_data\ALINE_data\After_long_storage_RPT"
    test_ = NewareDataSetAline(aline_after_rest)
