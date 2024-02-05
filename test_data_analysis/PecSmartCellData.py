import pandas as pd
import numpy as np
from PythonScripts.test_data_analysis.BasePecDataClass import BasePecData, BasePecRpt, find_step_characteristics
import os


class PecSmartCellData(BasePecData):

    def __init__(self, fname, op_bool=0):
        super().__init__(fname)
        self.op_folder = ''
        self.write_output = op_bool
        self.test_name = self.lookup_test_()
        self.rpt_obj = PecSmartCellRpt(self.rpt_dict)
        self.set_op_dir()
        self.write_files()

    def set_op_dir(self):
        dir_name, fname = os.path.split(self.data_file)
        self.op_folder = os.path.join(dir_name, f'pickle_files_{fname.split("_")[0]}')
        return None

    def write_files(self):
        if self.write_output:
            if not os.path.isdir(self.op_folder):
                os.makedirs(self.op_folder)
            self.dyn_data.to_pickle(os.path.join(self.op_folder, 'full_data.pkl'))
            for k, df in self.ici_dict.items():
                f_name = f'ici_from_rpt_{k:.0f}.pkl'
                df.to_pickle(os.path.join(self.op_folder, f_name))
            for k, dct in self.rpt_dict.items():
                for nbr, df in dct.items():
                    f_name_rpt = f'rpt_{nbr:.0f}_{k}_raw.pkl'
                    df.to_pickle(os.path.join(self.op_folder, f_name_rpt))
            self.rpt_obj._output_rpt_summary(os.path.join(self.op_folder, 'rpt_summary'))
        return None

    def lookup_test_(self):
        test_name_dict = {
            'Test2441': 'FaultTrace',
            'Test2443': 'FaultTrace',
            'Test2445': 'FaultTraceForMetaData',
            'Test2447': 'FaultTrace',
            'Test2449': 'FaultTrace',
            'Test2461': 'FaultTrace',
            'Test2463': 'FaultTrace',
            'Test2465': 'FaultTrace_pretest',
            'Test2468': 'FaultTrace_pretest',
            'Test2469': 'FaultTrace_pretest',
            'Test2470': 'Implement_step_wise_reduction',
            'Test2477': 'Full pre-test',
            'Test2484': 'Failed Li plating test',
            'Test2498': 'Test with slow rise'
        }
        return test_name_dict[self.test_nbr]


class PecSmartCellRpt(BasePecRpt):

    def __init__(self, data_dict):
        super().__init__(data_dict)
        self.cap_max_volt = 4.0
        self.cap_min_volt = 2.8
        self.rpt_summary = pd.DataFrame()
        self.comp_rpt_dict = {k: self.comp_rpt_analysis(df) for k, df in self.data_dict['comp'].items()}
        self.fast_rpt_dict = {k: self.fast_rpt_analysis(df) for k, df in self.data_dict['fast'].items()}
        self._make_rpt_summary()

    def fast_rpt_analysis(self, df):
        stp_info = find_step_characteristics(df)
        FCE = self.find_fce(df)
        date, q_mean, q_err = self.find_capacity_measurement(stp_info)
        rpt_df = pd.DataFrame(data=[date, FCE, q_mean, q_err], index=['date', 'fce', 'cap', 'sig_cap']).T
        return rpt_df

    def comp_rpt_analysis(self, df):
        stp_info = find_step_characteristics(df)
        date, q_mean, q_err = self.find_capacity_measurement(stp_info)
        FCE = self.find_fce(df)
        res_df = self.find_resistance_vals(df, stp_info)
        flat_rdf = self.flatten_df(res_df)
        rpt_df = pd.DataFrame(data=[date, FCE, q_mean, q_err], index=['date', 'fce', 'cap', 'sig_cap']).T
        return pd.concat([rpt_df, flat_rdf], axis=1)

    @staticmethod
    def flatten_df(df):
        s = df.stack()
        df1 = pd.DataFrame([s.values], columns=[f'{j}-{i}' for i, j in s.index])
        return df1

    def find_capacity_measurement(self, df):
        v_margin = 0.1
        cap_idx = df[(df.maxV >= self.cap_max_volt - v_margin) &
                     (df.minV <= self.cap_min_volt + v_margin) &
                     (df.step_mode == 'dchg')].index
        cap_vals = df.loc[cap_idx, 'cap']
        cap_date = df.loc[cap_idx[0], 'stp_date'].strftime("%Y-%m-%d")
        cap_mean = cap_vals.mean()
        cap_err = cap_vals.std(ddof=1) / cap_mean
        return cap_date, cap_mean, cap_err

    def find_fce(self, df):
        fce_ = df['cycle'].unique()
        return fce_[0]

    def find_resistance_vals(self, df, stp_info):
        from scipy.interpolate import interp1d
        pulse_steps = stp_info[(stp_info.duration < 11) & (stp_info.curr.abs() > 5)].step_nbr
        soc_arr = [30, 30, 50, 50, 70, 70]
        volt_lvl = [2.8, 3.5, 3.55, 3.7, 3.75, 4.2]
        soc_lookup = interp1d(volt_lvl, soc_arr, kind='previous')
        R0_dchg = {}
        R0_chrg = {}
        R10_dchg = {}
        R10_chrg = {}
        for stp in pulse_steps:
            ocv = df.loc[df[df.unq_step == stp - 1].last_valid_index(), 'volt']
            print('Found ocv of {:.2f}'.format(ocv))
            round_soc = soc_lookup(ocv)
            print(f'Found rounded soc of {round_soc}')
            step_label = f'soc_{round_soc:.0f}'
            curr = df[df.unq_step== stp].curr.mean()
            start_index_pulse = df[df.unq_step == stp].first_valid_index()
            fin_index_pulse = df[df.unq_step == stp].last_valid_index()
            R10 = (df.loc[fin_index_pulse, 'volt'] - ocv) / curr
            R0 = (df.loc[start_index_pulse, 'volt'] - ocv) / curr
            if curr > 0:
                R0_chrg[step_label] = R0
                R10_chrg[step_label] = R10
            else:
                R0_dchg[step_label] = R0
                R10_dchg[step_label] = R10
        op = pd.DataFrame([R10_dchg, R0_dchg, R10_chrg, R0_chrg]).T
        op.columns = ['R10_dchg', 'R0_dchg', 'R10_chrg', 'R0_chrg']
        return op

    def _make_rpt_summary(self):
        df_fast = pd.concat(self.fast_rpt_dict.values())
        df_comp = pd.concat(self.comp_rpt_dict.values())
        rpt_summary = pd.concat([df_comp, df_fast]).sort_values(by='fce')
        rpt_summary.reset_index(drop=True, inplace=True)
        for c in rpt_summary.columns:
            if c.startswith('cap') or c.startswith('R0') or c.startswith('R10'):
                normal_idx = rpt_summary[c].first_valid_index()
                rpt_summary[f'{c}_normalised'] = rpt_summary[c] / rpt_summary[c].iloc[normal_idx]
        self.rpt_summary = rpt_summary
        return None

    def _output_rpt_summary(self, fpath):
        if self.rpt_summary.empty:
            self._make_rpt_summary()
        self.rpt_summary.to_excel(f'{fpath}.xlsx', index=False)
        self.rpt_summary.to_pickle(f'{fpath}.pkl')


if __name__ == '__main__':
    test_file = r"\\sol.ita.chalmers.se\groups\batt_lab_data\smart_cell_JG\TestBatch2_autumn2023\Test2498_Cell-1.csv"
    class_test = PecSmartCellData(test_file, op_bool=1)
