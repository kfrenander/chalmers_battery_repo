import datetime as dt
import glob
import re
import time
import pandas as pd
from test_data_analysis.BaseNewareDataClass import BaseNewareDataSet, BaseNewareData, BaseRptData


class LgNewareDataSet(BaseNewareDataSet):

    def __init__(self, neware_data_dir):
        super().__init__(neware_data_dir)
        self.fill_data()

    def fill_data(self):
        temp_dict = {}
        for key, itm_lst in self.chan_file_dict.items():
            # remaining_list = ['2-2']  #['1-1', '1-3', '1-5', '1-6', '2-1', '2-5', '2-6']
            # if any(s in key for s in remaining_list):
            root_dir = os.path.split(itm_lst[0])[0]
            # chan_id = re.findall(r'\b\d\b', key)
            # chan_number = '_'.join(chan_id)
            # exp_name = f'pickle_files_channel_{chan_number}'
            exp_name = f'pickle_files_channel_{key}'
            print(f'Calling Neware data with {key}')
            tic = dt.datetime.now()
            pkl_dir = os.path.join(root_dir, exp_name.replace('-', '_'))
            if os.path.exists(pkl_dir):
                print('Files already read. \nWill check if metadata update needed')
                metadata_files = glob.glob(os.path.join(pkl_dir, 'metadata*.txt'))
                if metadata_files:
                    print('Metadata files already generated, no update needed')
                else:
                    base_data = BaseNewareData(itm_lst)
                    base_data.write_meta_data()
            else:
                try:
                    temp_dict[key] = LgNewareData(itm_lst)
                except OSError as e:
                    print('Probably not enough memory')
                    print(e)
                    temp_dict[key] = 'Placeholder due to OSError'
                except MemoryError as e:
                    print(f'MemoryError with message {e}')
                    temp_dict[key] = 'Placeholder due to MemoryError'
            # except:
            #     print('General error')
            #     temp_dict[key] = 'Placeholder due to unknown error'
            toc = dt.datetime.now()
            print(f'Time elapsed for test {key} was {(toc - tic).total_seconds() / 60:.2f} min.\n\n')
        self.data_dict = temp_dict
        return None


class LgNewareData(BaseNewareData):
    def __init__(self, list_of_files, n_cores=8):
        super().__init__(list_of_files, n_cores)
        self.test_name = self.look_up_test_name(self.channel_name)
        debug_start = time.time()
        super().step_characteristics()
        # df_split = np.array_split(self.dyn_df, n_cores)
        # pool = Pool(n_cores)
        # self.step_char = pd.concat(pool.map(characterise_steps, df_split))
        # pool.close()
        # pool.join()
        print(f'Time for characterisation was {time.time() - debug_start:.2f} seconds.')
        super().find_ici_steps()
        super().find_ica_steps()
        self.rpt_analysis = LgRptData(self.find_rpt_dict(), self.ica_step_list)
        super().pickle_data_dump()
        super().write_rpt_summary()
        super().write_meta_data()
        # self.dyn_df = pd.DataFrame()
        self.xl_files = []

    def look_up_test_name(self, chan_key):
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
        try:
            if '240046' in self.unit_name:
                return name_dict_stat[chan_key]
            elif '240095' in self.unit_name:
                return name_dict_stat[chan_key]
            else:
                raise KeyError('Unknown unit used for test, update dictionaries')
        except KeyError:
            print('Channel not in list, return \'RPT\'')
            return 'RPT'


class LgRptData(BaseRptData):
    def __init__(self, rpt_dict, ica_step_list):
        BaseRptData.__init__(self, rpt_dict, ica_step_list)
        self.cell_lim = {
            'Umax': 4.2,
            'Umin': 2.5,
            'C_rate': -1.15
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
                if ((step_df['step_mode'][0] == 'CC_DChg' or step_df['step_mode'][0] == 'CC DChg') and (ref_df['step_mode'][0] == 'CCCV_Chg' or ref_df['step_mode'][0] == 'CCCV Chg')):
                    if abs(step_df.curr[0] + 1.15) < 0.2 and step_df['maxV'][0] > 4 and step_df['minV'][0] < 3:
                        print(f'Cap at rpt {i:.0f} is {step_df.cap[0]:.2f} mAh')
                        cap_df.loc[f'cap_meas_{i}'] = step_df.cap.values
                        # tmp_cap_df = pd.DataFrame(data=step_df.cap.values, columns=['cap'], index=[f'cap_meas_{i}'])
                        # cap_df = pd.concat([cap_df, tmp_cap_df])
                        i += 1
            except:
                pass
        cap_df.loc['mean'] = cap_df.mean()
        cap_df.loc['var_normed'] = cap_df.filter(like='cap_meas', axis=0).std(ddof=1) / cap_df.loc['mean']
        return cap_df


if __name__ == '__main__':
    from check_current_os import get_base_path_batt_lab_data
    import os

    outer_tic = dt.datetime.now()
    BASE_PATH = get_base_path_batt_lab_data()
    stat_test = "stat_test/cycling_data"
    pulse_charge = "pulse_chrg_test/cycling_data_ici"
    test_case = LgNewareDataSet(os.path.join(BASE_PATH, pulse_charge))
    outer_toc = dt.datetime.now()
    print(f'Total elapsed time was {(outer_toc - outer_tic).total_seconds() / 60:.2f} min.')
