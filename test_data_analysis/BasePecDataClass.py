import pandas as pd
import re
import pandas.errors
from scipy.integrate import cumtrapz
import datetime as dt


def find_step_characteristics(df):
    step_dict = {}
    date_fmt = '%m/%d/%Y %H:%M:%S'
    for stp in df.unq_step.unique():
        sub_df = df[df.unq_step == stp]
        avg_curr = sub_df.curr.mean()
        max_volt = sub_df.volt.max()
        min_volt = sub_df.volt.min()
        dur = sub_df.float_time.max() - sub_df.float_time.min()
        step_dur = sub_df.step_time.max()
        stp_cap = sub_df.step_cap.abs().max()
        stp_mode = BasePecData.check_step_mode(avg_curr)
        stp_date = sub_df['abs_time'].iloc[0]
        data_pts = {
            'stp_date': stp_date,
            'maxV': round(max_volt, 3),
            'minV': round(min_volt, 3),
            'cap': stp_cap,
            'curr': avg_curr,
            'duration': dur,
            'step_duration': step_dur,
            'step_nbr': stp,
            'step_mode': stp_mode,
        }
        step_dict['{0}'.format(stp)] = data_pts
    df_out = pd.DataFrame(step_dict).T
    return df_out


class BasePecData(object):

    def __init__(self, fname, data_init_row=24):
        self.rpt_dict = {}
        self.ici_dict = {}
        self.meta_data_dict = {}
        self.data_file = fname
        self.data_init_row = data_init_row
        self.test_nbr = self.find_test_nbr()
        self.col_name_dict = self.make_rename_column_dict()
        self.dyn_data = self.read_pec_file(fname)
        self.meta_data_dict = self.read_pec_metadata()
        self.dyn_data = self.fill_step_cap()
        self.step_info = find_step_characteristics(self.dyn_data)
        # self.find_all_rpt()
        # self.find_ici()

    def find_test_nbr(self):
        return re.search(r'Test\d+', self.data_file).group()

    def make_rename_column_dict(self):
        orig_name_list =   ['Step',
                            'Instruction Name',
                            'Cycle',
                            'Total Time (Seconds)',
                            'Step Time (Seconds)',
                            'Real Time',
                            'Voltage (V)',
                            'Current (A)',
                            'Charge Capacity (mAh)',
                            'Discharge Capacity (mAh)',
                            'Charge Energy (mWh)',
                            'Discharge Energy (mWh)',
                            'CycNbrOuter [Var18]',
                            'CycNbrRpt [Var19]',
                            'RPT_fast_bool [Var20]',
                            'RPT_comp_bool [Var21]',
                            'TC_1 (°C)',
                            'Unnamed: 16',
                            'Unnamed: 17',
                            'Unnamed: 18',
                            'ICI_bool [Var22]',
                            'Unnamed: 13',
                            'LiPlating_bool [Var26]'
                            ]
        new_name_list = ['step',
                         'instruction',
                         'cycle',
                         'float_time',
                         'step_time',
                         'abs_time',
                         'volt',
                         'curr',
                         'chrg_cap',
                         'dchg_cap',
                         'chrg_egy',
                         'dcgh_egy',
                         'CycNbrOuter',
                         'CycNbrRpt',
                         'rpt_fast_bool',
                         'rpt_comp_bool',
                         'temperature',
                         'unknown1',
                         'unknown2',
                         'unknown3',
                         'ici_bool',
                         'unknown4',
                         'li_plating_bool']
        name_update_dict = dict(zip(orig_name_list, new_name_list))
        return name_update_dict

    def read_pec_file(self, fname):
        from test_data_analysis.read_pec_file import find_data_init_row
        try:
            raw_df = pd.read_csv(fname, skiprows=self.data_init_row)
            raw_df.columns = [self.col_name_dict[c] for c in raw_df.columns]
        except (pd.errors.ParserError, KeyError):
            print(f'Error detected for initrow = {self.data_init_row}')
            self.data_init_row = find_data_init_row(fname)
            raw_df = pd.read_csv(fname, skiprows=self.data_init_row)
            raw_df.columns = [self.col_name_dict[c] for c in raw_df.columns]
        raw_df['abs_time'] = pd.to_datetime(raw_df.abs_time)
        raw_df.loc[:, 'mAh'] = cumtrapz(raw_df.curr, raw_df.float_time, initial=0) / 3600
        nbr_of_unique_steps = raw_df[raw_df.step.diff() != 0].shape[0]
        raw_df.loc[raw_df.step.diff() != 0, 'unq_step'] = range(1, nbr_of_unique_steps + 1)
        df = raw_df.fillna(method='ffill').fillna(method='bfill').dropna(how='all', axis=1).copy()
        return df

    def read_pec_metadata(self):
        tmp_dct = {}
        with open(self.data_file) as f:
            for cnt, line in enumerate(f):
                if cnt < self.data_init_row:
                    ln_ = line.split(',')
                    tmp_dct[ln_[0].strip()] = [l.strip() for l in ln_[1:]]
                else:
                    f.close()
                    break
        return tmp_dct

    @staticmethod
    def check_step_mode(i_mean):
        if abs(i_mean) < 1e-5:
            return 'idle'
        elif i_mean > 1e-5:
            return 'chrg'
        elif i_mean < -1e-5:
            return 'dchg'
        else:
            print(f'Unknown format of mean current {i_mean}, return None')
            return None

    @staticmethod
    def set_unique_number_pec(df):
        nbr_of_unique_steps = df[df.step.diff() != 0].shape[0]
        df.loc[df.step.diff() != 0, 'unq_step'] = range(1, nbr_of_unique_steps + 1)
        return df.copy()

    @staticmethod
    def calc_step_cap(group):
        group['step_cap'] = cumtrapz(group.curr, group.step_time, initial=0) / 3.6
        return group

    @staticmethod
    def calc_step_egy(group):
        group['step_egy'] = cumtrapz(group.curr*group.volt, group.step_time, initial=0) / 3.6
        return group

    def fill_step_cap(self):
        return self.dyn_data.groupby(by='unq_step', group_keys=False).apply(self.calc_step_cap)

    def fill_step_egy(self):
        return self.dyn_data.groupby(by='unq_step', group_keys=False).apply(self.calc_step_egy)

    def find_all_rpt(self):
        fast_rpt_dict = self.find_rpt(rpt_type='fast')
        comp_rpt_dict = self.find_rpt(rpt_type='comp')
        self.rpt_dict = {
            'fast': fast_rpt_dict,
            'comp': comp_rpt_dict
        }
        return None

    def find_rpt(self, rpt_type):
        if rpt_type == 'comp':
            bool_col = 'rpt_comp_bool'
        elif rpt_type == 'fast':
            bool_col = 'rpt_fast_bool'
        else:
            print(f'No analysis possible for rpt key of \'{rpt_type}\'. Ending.')
            return None
        rpt_df = self.dyn_data[self.dyn_data[bool_col] == 1]
        gb = rpt_df.groupby(by='CycNbrOuter')
        rpt_dict = {k: gb.get_group(k) for k in gb.groups if gb.get_group(k).shape[0] > 100}
        return rpt_dict

    def find_ici(self):
        ici_df = self.dyn_data[self.dyn_data['ici_bool'] == 1]
        gb = ici_df.groupby(by='CycNbrOuter')
        self.ici_dict = {k: gb.get_group(k) for k in gb.groups}
        return None


class BasePecRpt(object):

    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.step_info_dict = {}
        self._step_info()

    def _step_info(self):
        tmp_dct = {}
        for k, dct in self.data_dict.items():
            for j, df in dct.items():
                tmp_dct[f'{k}_{j}'] = find_step_characteristics(df)
        self.step_info_dict = tmp_dct
        return None

    @staticmethod
    def flatten_df(df):
        s = df.stack()
        df1 = pd.DataFrame([s.values], columns=[f'{j}-{i}' for i, j in s.index])
        return df1


if __name__ == '__main__':
    test_case = r"\\sol.ita.chalmers.se\groups\batt_lab_data\smart_cell_JG\TestBatch2_autumn2023\Test2441_Cell-1.csv"
    fault_trace_data = BasePecData(test_case)
