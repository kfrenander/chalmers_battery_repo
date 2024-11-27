import pandas as pd
import re
from scipy.integrate import cumtrapz


def find_step_characteristics(df):
    step_dict = {}
    date_fmt = '%m/%d/%Y %H:%M:%S'
    for stp in df.unq_step.unique():
        sub_df = df[df.unq_step == stp]
        avg_curr = sub_df.curr.mean()
        max_volt = sub_df.volt.max()
        min_volt = sub_df.volt.min()
        dur = sub_df.float_time.max() - sub_df.float_time.min()
        step_dur = sub_df.float_step_time.max()
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


def find_step_characteristics_fast(df):
    date_fmt = '%m/%d/%Y %H:%M:%S'

    # Group by `unq_step` and calculate required metrics
    grouped = df.groupby('unq_step')

    # Precompute values using groupby and aggregation
    step_data = grouped.agg(
        avg_curr=('curr', 'mean'),
        max_volt=('volt', 'max'),
        min_volt=('volt', 'min'),
        dur=('float_time', lambda x: x.max() - x.min()),
        step_dur=('float_step_time', 'max'),
        stp_cap=('step_cap', lambda x: x.abs().max()),
        stp_date=('abs_time', 'first'),
    ).reset_index()

    # Add the step mode as a separate column
    step_data['step_mode'] = step_data['avg_curr'].apply(BasePecData.check_step_mode)

    # Create the resulting DataFrame with rounded values
    step_data['maxV'] = step_data['max_volt'].round(3)
    step_data['minV'] = step_data['min_volt'].round(3)

    # Reorganize columns to match the original structure
    df_out = step_data.rename(columns={
        'unq_step': 'step_nbr',
        'avg_curr': 'curr',
        'dur': 'duration',
        'step_dur': 'step_duration',
        'stp_cap': 'cap'
    })[['stp_date', 'maxV', 'minV', 'cap', 'curr', 'duration', 'step_duration', 'step_nbr', 'step_mode']]

    return df_out.set_index('step_nbr', drop=False)


class BasePecData(object):

    def __init__(self, fname, data_init_row=24):
        self.meta_data_dict = {}
        self.data_file = fname
        self.data_init_row = data_init_row
        self.meta_data_dict = self.read_pec_metadata()
        self.test_nbr = self.find_test_nbr()
        self.cell_nbr = self.find_cell_nbr()
        self.col_name_dict = self.make_rename_column_dict()
        self.dyn_data = self.read_pec_file(fname)
        self.dyn_data = self.fill_step_cap_fast()
        self.dyn_data = self.fill_step_egy_fast()
        self.step_info = find_step_characteristics_fast(self.dyn_data)

    def find_test_nbr(self):
        return self.meta_data_dict['Test']

    def find_cell_nbr(self):
        cell_nbr_raw = self.meta_data_dict['Split Results'][-1]
        return re.search('\d+', cell_nbr_raw).group()

    def make_rename_column_dict(self):
        orig_name_list = ['Step',
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
                         'float_step_time',
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
                         'li_plating_bool'
                         ]
        name_update_dict = dict(zip(orig_name_list, new_name_list))
        return name_update_dict

    def read_pec_file(self, fname):
        from test_data_analysis.read_pec_file import find_data_init_row
        try:
            raw_df = pd.read_csv(fname, skiprows=self.data_init_row)
        except (pd.errors.ParserError, KeyError):
            self.data_init_row = find_data_init_row(fname)
            print(f'Error detected for default initrow \nRe-initiating initrow to {self.data_init_row}.')
            raw_df = pd.read_csv(fname, skiprows=self.data_init_row)
            # raw_df.columns = [self.col_name_dict[c] for c in raw_df.columns]
        raw_df.rename(self.col_name_dict, axis=1, inplace=True)
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
                    if len(ln_) != 2:
                        tmp_dct[ln_[0].strip().strip(':')] = [ln.strip() for ln in ln_[1:]]
                    else:
                        tmp_dct[ln_[0].strip().strip(':')] = ln_[1].strip()
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
        group['step_cap'] = cumtrapz(group.curr, group.float_step_time, initial=0) / 3.6
        return group

    @staticmethod
    def calc_step_egy(group):
        group['step_egy'] = cumtrapz(group.curr * group.volt, group.float_step_time, initial=0) / 3.6
        return group

    def fill_step_cap(self):
        return self.dyn_data.groupby(by='unq_step', group_keys=False).apply(self.calc_step_cap)

    def fill_step_cap_fast(self):
        df = self.dyn_data
        df['step_cap'] = (
            df.groupby(by='unq_step')
            .apply(lambda g: cumtrapz(g['curr'], g['float_step_time'], initial=0) / 3.6)
            .explode()
            .to_numpy()
        )
        return df

    def fill_step_egy_fast(self):
        df = self.dyn_data
        df['step_egy'] = (
            df.groupby(by='unq_step')
            .apply(lambda group: cumtrapz(group.curr * group.volt, group.float_step_time, initial=0)/3.6).
            explode().to_numpy()
        )
        return df

    def fill_step_egy(self):
        return self.dyn_data.groupby(by='unq_step', group_keys=False).apply(self.calc_step_egy)


class BasePecRpt(object):

    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.step_info_dict = {}
        # self._step_info()

    def _step_info(self):
        tmp_dct = {}
        for k, dct in self.data_dict.items():
            for j, df in dct.items():
                tmp_dct[f'{k}_{j}'] = find_step_characteristics_fast(df)
        self.step_info_dict = tmp_dct
        return None

    @staticmethod
    def flatten_df(df):
        s = df.stack()
        df1 = pd.DataFrame([s.values], columns=[f'{j}-{i}' for i, j in s.index])
        return df1


if __name__ == '__main__':
    from check_current_os import get_base_path_batt_lab_data
    import os
    BASE_PATH = get_base_path_batt_lab_data()
    test_case = os.path.join(BASE_PATH, "smart_cell_JG\TestBatch2_autumn2023\Test2441_Cell-1.csv")
    fault_trace_data = BasePecData(test_case)
