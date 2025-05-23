import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from test_data_analysis.BasePecDataClass import BasePecRpt, find_step_characteristics_fast
from test_data_analysis.pec_lifetest import PecLifeTestData
from misc_classes.pulse_charge_style_class import TestCaseStyler
import os
import re


class PecSmartCellData(PecLifeTestData):

    def __init__(self, filename, op_bool=0):
        super().__init__(filename)
        self.op_folder = ''
        self.write_output = op_bool
        self.test_name = self.meta_data_dict['TestRegime Name']
        self.rpt_obj = PecSmartCellRpt(self.rpt_dict)
        self.set_op_dir()
        self.write_files()
        self.line_styler = TestCaseStyler()
        self.style = self.line_styler.get_abbrv_style(self.formatted_metadata['OUTPUT_NAME'])
        self.popt = None
        self.pcov = None
        self._complement_metadata()

    def _q_function(self, fce, q0, tau, beta):
        """The function to fit: Q(fce) = q0 * exp(-(fce/tau)^beta)"""
        fce = np.asarray(fce)
        return q0 * np.exp(-(fce / tau) ** beta)

    def fit_degradation_function(self):
        """
            Fit the Q(fce) function to data.
            Args:
               fce (array-like): FCE data
               q (array-like): Capacity data
            Returns:
               popt (tuple): Optimized parameters (q0, tau, beta)
               pcov (2D array): Covariance matrix of the fit
        """
        x_data = self.rpt_obj.rpt_summary.fce.astype('float')
        y_data = self.rpt_obj.rpt_summary.cap_normalised.astype('float')

        # Define the objective function for differential evolution
        def objective(params):
            q0, tau, beta = params
            return np.sum((y_data - self._q_function(x_data, q0, tau, beta)) ** 2)

        # Use differential evolution to get initial parameters
        bounds = [(0, 10), (0.1, 1e10), (0.1, 5)]
        result = differential_evolution(objective, bounds, strategy='best1bin', tol=1e-5)
        initial_guess = result.x  # Use this as initial guess for curve_fit

        popt, pcov = curve_fit(self._q_function, x_data, y_data, p0=[1, 1000, 1])
        self.popt = popt
        self.pcov = pcov
        return popt, pcov

    def visualise_fit_function(self):
        if self.popt is None:
            _, _ = self.fit_degradation_function()
        labels = [r"$q_0$", r"$\tau$", r"$\beta$"]
        plt.figure()
        x_data = self.rpt_obj.rpt_summary.fce.astype('float')
        y_data = self.rpt_obj.rpt_summary.cap_normalised.astype('float')
        x_model = np.linspace(x_data.min(), x_data.max(), 100)
        y_model = self._q_function(x_model, *self.popt)
        plt.plot(x_model, y_model, label='Fit')
        plt.scatter(x_data, y_data, color='red', label='Raw')
        text_str = "\n".join(f"{label} = {value:.2f}" for label, value in zip(labels, self.popt))
        plt.text(20, 0.9, text_str, verticalalignment='top',
                 bbox={'facecolor': 'white', 'boxstyle': 'round'})
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        ymin = min(ymin, 0.85)
        ax.set_ylim((ymin, ymax))
        plt.show()

    def find_fce_at_given_q(self, q_target):
        """
        Find the time at which the fitted Q(t) function yields a given Q.
        Args:
            q_target (float): Target capacity
        Returns:
            t (float): Time corresponding to the target Q
        """
        if self.popt is None:
            raise ValueError("You must fit the function before calling this method.")

        # Solve for t such that Q(t) = q_target
        from scipy.optimize import root_scalar

        def equation(fce):
            return q_target - self._q_function(fce, *self.popt)

        result = root_scalar(equation, bracket=[1e-5, 1e5], method='brentq')
        if result.converged:
            return result.root
        else:
            raise RuntimeError("Failed to converge to a solution.")

    def set_op_dir(self):
        dir_name, fname = os.path.split(self.data_file)
        if self.formatted_metadata:
            op_name = (f'pickle_files_{self.formatted_metadata["OUTPUT_NAME"].replace(" ", "_")}_'
                       f'cell_{self.formatted_metadata["CELL_ID"]}')
            self.op_folder = os.path.join(dir_name, op_name)
        else:
            self.op_folder = os.path.join(dir_name, f'pickle_files_{fname.split(".")[0].replace("-","_")}')
        return None

    def write_files(self):
        if self.write_output:
            if not os.path.isdir(self.op_folder):
                os.makedirs(self.op_folder)
            self.dyn_data.to_pickle(os.path.join(self.op_folder, 'full_data.pkl'))
            for k, ici_obj in self.ici_dict.items():
                f_name_raw = f'raw_ici_from_rpt_{k:.0f}.pkl'
                ici_obj.raw_data.to_pickle(os.path.join(self.op_folder, f_name_raw))
                f_name_res = f'processed_ici_from_rpt_{k:.0f}.pkl'
                ici_obj.ici_result_df.to_pickle(os.path.join(self.op_folder, f_name_res))
            for k, dct in self.rpt_dict.items():
                for nbr, df in dct.items():
                    f_name_rpt = f'rpt_{nbr:.0f}_{k}_raw.pkl'
                    df.to_pickle(os.path.join(self.op_folder, f_name_rpt))
            self.rpt_obj._output_rpt_summary(os.path.join(self.op_folder, 'rpt_summary'))
            metadata_fname = f'metadata_test{self.test_nbr}_cell{self.cell_nbr}.txt'
            with open(os.path.join(self.op_folder, metadata_fname), 'w') as f:
                for key, value in self.formatted_metadata.items():
                    f.write(f"{key}: {value}\n")
        return None

    def lookup_test(self):
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
        try:
            return test_name_dict[self.test_nbr]
        except KeyError:
            print(f'Key{self.test_nbr} not found. Return default.')
            return 'Default_unknown_test'

    def filter_ici_on_cap(self, cap_vals):
        rpt_df = self.rpt_obj.rpt_summary.dropna(subset='R0_dchg-soc_70')
        filtered_ici = {}
        for value in cap_vals:
            # Find the absolute difference between the value and the column
            diffs = np.abs(rpt_df['cap_normalised'] - value).astype(float)
            # Find the index of the minimum difference
            closest_idx = diffs.idxmin()
            # Append the identified ICI
            if self.ici_dict[closest_idx].ici_result_df.maxV.max() > 4:
                filtered_ici[closest_idx] = self.ici_dict[closest_idx]
            else:
                closest_idx = diffs.nsmallest(n=2).index[-1]
                filtered_ici[closest_idx] = self.ici_dict[closest_idx]
        return filtered_ici

    def _complement_metadata(self):
        pattern = r'(?P<Duty>\d+(\.\d+)?)duty (?P<C_rate>\d+(\.\d+)?)C (?P<Frequency>\d+(\.\d+)?)Hz'

        # Using re.search to find the matches
        match = re.search(pattern, self.formatted_metadata['TEST_CONDITION'])

        if match:
            self.formatted_metadata.update(match.groupdict())
        else:
            self.formatted_metadata.update({
                'Duty': 100,
                'C_rate': 1,
                'Frequency': 0
            })


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
        stp_info = find_step_characteristics_fast(df)
        FCE = self.find_fce(df)
        date, q_mean, q_err = self.find_capacity_measurement(stp_info)
        rpt_df = pd.DataFrame(data=[date, FCE, q_mean, q_err], index=['date', 'fce', 'cap', 'sig_cap']).T
        return rpt_df

    def comp_rpt_analysis(self, df):
        stp_info = find_step_characteristics_fast(df)
        date, q_mean, q_err = self.find_capacity_measurement(stp_info)
        FCE = self.find_fce(df)
        res_df = self.find_resistance_vals(df, stp_info)
        flat_rdf = BasePecRpt.flatten_df(res_df)
        rpt_df = pd.DataFrame(data=[date, FCE, q_mean, q_err], index=['date', 'fce', 'cap', 'sig_cap']).T
        return pd.concat([rpt_df, flat_rdf], axis=1)

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
        volt_lvl = [2.8, 3.65, 3.66, 3.84, 3.85, 4.2]
        soc_lookup = interp1d(volt_lvl, soc_arr, kind='previous')
        R0_dchg = {}
        R0_chrg = {}
        R10_dchg = {}
        R10_chrg = {}
        for stp in pulse_steps:
            ocv = df.loc[df[df.unq_step == stp - 1].last_valid_index(), 'volt']
            # print('Found ocv of {:.2f}'.format(ocv))
            round_soc = soc_lookup(ocv)
            step_label = f'soc_{round_soc:.0f}'
            curr = df[df.unq_step == stp].curr.mean()
            start_index_pulse = df[df.unq_step == stp].first_valid_index()
            fin_index_pulse = df[df.unq_step == stp].last_valid_index()
            if abs(stp_info.loc[stp, 'duration'] - 10) > 0.5:
                R10 = None
            else:
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
        try:
            df_fast = pd.concat(self.fast_rpt_dict.values())
        except ValueError as e:
            print(f'Not possible to concatenate empty dict. Return empty df instead.')
            df_fast = pd.DataFrame()
        try:
            df_comp = pd.concat(self.comp_rpt_dict.values())
        except ValueError as e:
            print(f'Not possible to concatenate empty dict. Return empty df instead.')
            df_comp = pd.DataFrame()
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

    def __call__(self, t):
        """
        Make the fitted function callable, using the fitted parameters.
        Args:
            t (array-like): Time data
        Returns:
            Q (array-like): Fitted capacity values
        """
        if self.popt is None:
            raise ValueError("You must fit the function before calling it.")
        q0, tau, beta = self.popt
        return self._q_function(t, q0, tau, beta)


if __name__ == '__main__':
    from check_current_os import get_base_path_batt_lab_data
    plt.style.use('widthsixinches')
    BASE_PATH = get_base_path_batt_lab_data()
    test_file_path = r"pulse_chrg_test\high_frequency_testing\PEC_export\Test2818_Cell-1.csv"
    test_file = os.path.join(BASE_PATH, test_file_path)
    # test_file = r"D:\PEC_logs\Test2767_Cell-1.csv"
    class_test = PecSmartCellData(test_file, op_bool=0)
    df = class_test.rpt_obj.rpt_summary
    class_test.fit_degradation_function()
    class_test.visualise_fit_function()
    filtered_ici = class_test.filter_ici_on_cap(np.arange(1, 0.8, -0.1))
