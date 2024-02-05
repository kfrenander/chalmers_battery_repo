import sys
sys.path.append(r'C:\battery-model\PythonScripts')
from test_data_analysis.BaseNewareDataClass import BaseNewareDataSet, BaseNewareData, BaseRptData
import os
import re
import pandas as pd
import datetime as dt
from natsort.natsort import natsorted
import numpy as np
from test_data_analysis.rpt_analysis import characterise_steps
from multiprocessing import Pool
import pickle


class TeslaNewareDataSet(BaseNewareDataSet):

    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.fill_data()

    def fill_data(self):
        temp_dict = {}
        for key in self.chan_file_dict:
            remaining_list = ['2-4']  #['1-1', '1-3', '1-5', '1-6', '2-1', '2-5', '2-6']
            #if any(s in key for s in remaining_list):
            root_dir = os.path.split(self.chan_file_dict[key][0])[0]
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
                    temp_dict[key] = TeslaNewareData(self.chan_file_dict[key])
                except OSError as e:
                    print('Probably not enough memory')
                    print(e)
                    temp_dict[key] = 'Placeholder due to OSError'
                except MemoryError:
                    print('Running out of RAM')
                    temp_dict[key] = 'Placeholder due to MemoryError'
            toc = dt.datetime.now()
            print('Time elapsed for test {} was {:.2f} min.'.format(key, (toc-tic).total_seconds() / 60))
        self.data_dict = temp_dict
        return None


class TeslaNewareData(BaseNewareData):
    def __init__(self, list_of_files, n_cores=8):
        super().__init__(list_of_files, n_cores)
        self.test_name = self.look_up_test_name(self.channel_name)
        self.stat = pd.DataFrame()
        self.cyc = pd.DataFrame()
        print('Starting read in')
        self.xl_files = [pd.ExcelFile(file_name) for file_name in natsorted(self.file_names)]
        print('Read in of data to pandas finished')
        BaseNewareData.read_dynamic_data(self)
        super().read_cycle_statistics()
        debug_start = dt.datetime.now()
        df_split = np.array_split(self.dyn_df, n_cores)
        pool = Pool(n_cores)
        self.step_char = pd.concat(pool.map(characterise_steps, df_split))
        pool.close()
        pool.join()
        print('Time for characterisation was {} seconds.'.format((dt.datetime.now() - debug_start).seconds))
        super().find_ica_steps()
        self.rpt_analysis = TeslaRptData(self.find_rpt_dict(), self.ica_step_list)
        super().pickle_data_dump()
        super().write_rpt_summary()
        self.dyn_df = pd.DataFrame()
        self.xl_files = []

    def look_up_test_name(self, chan_key):
        name_dict_bda = {
            '240046_1_1': '1 second',
            '240046_1_2': '1 second',
            '240046_1_3': '2 seconds',
            '240046_1_4': '2 seconds',
            '240046_1_5': '4 seconds',
            '240046_1_6': '4 seconds',
            '240046_1_7': '8 seconds',
            '240046_1_8': '8 seconds',
            '240046_2_1': '16 seconds',
            '240046_2_2': '16 seconds',
            '240046_2_3': '32 seconds',
            '240046_2_4': '32 seconds',
            '240046_2_5': '64 seconds',
            '240046_2_6': '64 seconds',
            '240046_2_7': '128 seconds',
            '240046_2_8': '128 seconds'
        }
        name_dict_aline = {
            '240119_1_1': 'Storage 15 SOC',
            '240119_1_2': 'Storage 15 SOC',
            '240119_1_3': 'Storage 50 SOC',
            '240119_1_4': 'Storage 50 SOC',
            '240119_1_5': 'Storage 85 SOC',
            '240119_1_6': 'Storage 85 SOC',
            '240119_2_1': '5 to 15 SOC',
            '240119_2_2': '5 to 15 SOC',
            '240119_2_3': '15 to 25 SOC',
            '240119_2_4': '15 to 25 SOC',
            '240119_2_5': '25 to 35 SOC',
            '240119_2_6': '25 to 35 SOC',
            '240119_2_7': '35 to 45 SOC',
            '240119_2_8': '35 to 45 SOC',
            '240119_3_1': '45 to 55 SOC',
            '240119_3_2': '45 to 55 SOC',
            '240119_3_3': '55 to 65 SOC',
            '240119_3_4': '55 to 65 SOC',
            '240119_3_5': '65 to 75 SOC',
            '240119_3_6': '65 to 75 SOC',
            '240119_3_7': '75 to 85 SOC',
            '240119_3_8': '75 to 85 SOC',
            '240119_4_1': '85 to 95 SOC',
            '240119_4_2': '85 to 95 SOC',
            '240119_4_3': '3600 seconds room temp',
            '240119_4_4': '3600 seconds room temp',
            '240119_4_5': '50 to 100 SOC room temp',
            '240119_4_6': '50 to 100 SOC room temp',
            '240119_4_7': '0 to 50 SOC room temp',
            '240119_4_8': '0 to 50 SOC room temp',
            '240119_5_1': '0 to 50 SOC high temp',
            '240119_5_2': '3600 seconds high temp',
            '240119_5_3': '0 to 50 SOC high temp',
            '240119_5_4': '50 to 100 SOC high temp',
            '240119_5_5': '50 to 100 SOC high temp',
            '240119_5_6': '3600 seconds high temp',
            'FCE': '3600 seconds'
        }
        name_dict_bda_comp = {
            '240095_1_1': '2 seconds',
            '240095_1_2': '2 seconds',
            '240095_1_3': '4 seconds',
            '240095_1_4': '4 seconds',
            '240095_1_5': '2 seconds',
            '240095_1_6': '16 seconds',
            '240095_1_7': '64 seconds',
            '240095_1_8': '64 seconds',
            '240095_2_1': '256 seconds',
            '240095_2_2': '256 seconds',
            '240095_2_3': '256 seconds',
            '240095_2_4': 'inf seconds',
            '240095_2_5': 'Broken test',
            '240095_2_6': 'Broken test',
            '240095_2_7': 'inf seconds',
            '240095_2_8': 'inf seconds',
            '240095_3_1': '16 seconds'
        }
        try:
            if '240119' in self.unit_name:
                return name_dict_aline[chan_key]
            elif '240046' in self.unit_name:
                return name_dict_bda[chan_key]
            elif '240095' in self.unit_name:
                return name_dict_bda_comp[chan_key]
            else:
                raise KeyError('Unknown unit used for test, update dictionaries')
        except KeyError:
            print('Channel not in list, return \'RPT\'')
            return 'RPT'

class TeslaRptData(BaseRptData):
    def __init__(self, rpt_dict, ica_step_list):
        BaseRptData.__init__(self, rpt_dict, ica_step_list)
        self.cell_lim = {
            'Umax': 4.18,
            'Umin': 2.55,
            'C_rate': -1.53
        }
        self.cap = {key: self.find_capacity_measurement(key) for key in self.rpt_dict}
        self.summary = BaseRptData.create_cell_summary(self)

    def find_capacity_measurement(self, key):
        cap_df = pd.DataFrame(columns=['cap'])
        i = 1
        for stp in self.char_dict[key].step_nbr:
            try:
                ref_df = self.char_dict[key][self.char_dict[key].step_nbr == stp - 2]
                step_df = self.char_dict[key][self.char_dict[key].step_nbr == stp]
                if (step_df['step_mode'][0] == 'CC_DChg' or step_df['step_mode'][0] == 'CC DChg') \
                        and ref_df['step_mode'][0] == 'CCCV_Chg':
                    if abs(step_df.curr[0] + 1.53) < 0.2 and step_df['maxV'][0] > 4 and step_df['minV'][0] < 3:
                        print('Cap is {:.2f} mAh'.format(step_df.cap[0]))
                        cap_df = pd.concat([cap_df, pd.DataFrame(data=step_df.cap.values,
                                                                 columns=['cap'], index=[f'cap_meas_{i}'])])
                        i += 1
            except:
                pass
        cap_df.loc['mean'] = cap_df.filter(like='cap_meas', axis=0).mean()
        cap_df.loc['var_normed'] = cap_df.filter(like='cap_meas', axis=0).std(ddof=1) / cap_df.loc['mean']
        print(f'Normalised variance is {cap_df.loc["var_normed"]}')
        return cap_df


if __name__ == '__main__':
    aline_10_dod = r"\\sol.ita.chalmers.se\groups\batt_lab_data\ALINE_data\10_dod_data"
    aline_50_dod = r"\\sol.ita.chalmers.se\groups\batt_lab_data\ALINE_data\50_dod_data"
    aline_after_rest = r"\\sol.ita.chalmers.se\groups\batt_lab_data\ALINE_data\After_long_storage_RPT"
    test_case = TeslaNewareDataSet(aline_after_rest)
