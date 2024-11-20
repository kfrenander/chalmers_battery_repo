import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from test_data_analysis.BasePecDataClass import find_step_characteristics_fast
from test_data_analysis.rpt_analysis import characterise_steps
import matplotlib.pyplot as plt


def fit_lin_vs_sqrt_time(df, col):
    fit_df = df[(df['float_step_time'] > 1)] # & (df['step_time_float'] <= 5)]
    coeffs = np.polyfit(np.sqrt(fit_df['float_step_time']), fit_df[col], 1)
    return coeffs


def fit_lin_ocp_slope(df, col):
    fit_df = df[df['float_step_time'] > df['float_step_time'].max() - 120]
    fit_tuple = np.polyfit(fit_df['float_step_time'], fit_df[col], 1, full=True)
    coeffs = fit_tuple[0]
    residual = fit_tuple[1]
    return coeffs, residual


class ICIAnalysis:
    def __init__(self, input_data):
        """
        Initializes the ICI analysis class.

        Parameters:
        - input_data: str (path to pickle) or pd.DataFrame (direct dataframe)
        - settings: dict, optional (user-defined settings for columns and modes)
        """
        self.settings = None
        self.rest_dict = {}
        self.chrg_dict = {}
        self.dchg_dict = {}
        if isinstance(input_data, str):
            self.ici_df = pd.read_pickle(input_data)
        elif isinstance(input_data, pd.DataFrame):
            self.ici_df = input_data
        else:
            raise ValueError("Input data must be either a string (file path) or a pandas DataFrame")
        self.settings = self.initialise_settings()
        self.ch_df = self.settings['characterize_fun'](self.ici_df)
        self.split_df_to_dicts()

    def initialise_settings(self):
        if 'instruction' in self.ici_df.columns:
            return {
                'mode_column': 'instruction',  # Default is 'mode', can be overridden to 'instruction'
                'charge_instructions': 'I Charge',
                'discharge_instructions': 'I Disch.',
                'chrg_indicator': 'chrg',
                'dchg_indicator': 'dchg',
                'rest_instructions': 'idle',
                'step_counter': 'unq_step',
                'characterize_fun': find_step_characteristics_fast
            }
        elif 'mode' in self.ici_df.columns:
            return {
                'mode_column': 'mode',  # Default is 'mode', can be overridden to 'instruction'
                'charge_instructions': 'CC Chg',  # Default charge instruction for 'mode'
                'discharge_instructions': 'CC DChg',  # Default discharge instruction for 'mode'
                'chrg_indicator': 'CC Chg',
                'dchg_indicator': 'CC DChg',
                'rest_instructions': 'Rest',
                'step_counter': 'arb_step2',
                'characterize_fun': characterise_steps
            }

    def categorize_step(self):
        for k, row in self.ch_df.iterrows():
            if row['step_mode'] == self.settings['rest_instructions']:
                if self.ch_df.loc[k - 1, 'step_mode'] == self.settings['chrg_indicator']:
                    self.ch_df.loc[k, 'ici_mode'] = 'chrg'
                elif self.ch_df.loc[k - 1, 'step_mode'] == self.settings['dchg_indicator']:
                    self.ch_df.loc[k, 'ici_mode'] = 'dchg'
                self.ch_df.loc[k, 'current_mode'] = 'interrupt'
            else:
                self.ch_df.loc[k, 'current_mode'] = 'current'
        return self.ch_df

    def find_ici_parameters(self, df, ch_df):
        df['stp_diff'] = df[self.settings['step_counter']].diff().fillna(0)
        for k, row in ch_df.iterrows():
            if row['current_mode'] == 'interrupt':
                stp = row['step_nbr']
                idf = self.rest_dict[stp]
                if ch_df.loc[k - 1, 'step_mode'] == self.settings['chrg_indicator']:
                    cdf = self.chrg_dict[stp - 1]
                else:
                    cdf = self.dchg_dict[stp - 1]
                curr = cdf['curr'].mean()
                pol_volt = cdf['volt'].iloc[-1]
                # idf = df[df[self.settings['step_counter']] == stp]
                # curr = df[df[self.settings['step_counter']] == stp - 1]['curr'].mean()
                # pol_volt = df[df[self.settings['step_counter']] == stp - 1]['volt'].iloc[-1]
                beg_volt = idf['volt'].iloc[0]
                fin_volt = idf['volt'].iloc[-1]
                R0 = (pol_volt - beg_volt) / curr
                R10 = (pol_volt - fin_volt) / curr
                ch_df.loc[k, 'R0_mohm'] = R0 * 1e3
                ch_df.loc[k, 'R10_mohm'] = R10 * 1e3
                slope, v_intcpt = fit_lin_vs_sqrt_time(idf, 'volt')
                ch_df.loc[k, 'k'] = - slope / curr
                ch_df.loc[k, 'R_reg_mohm'] = 1e3 * (pol_volt - v_intcpt) / curr
            else:
                stp = row['step_nbr']
                if row['step_mode'] == self.settings['chrg_indicator']:
                    cdf = self.chrg_dict[stp]
                else:
                    cdf = self.dchg_dict[stp]
                # cdf = df[df[self.settings['step_counter']] == stp]
                fit_coeffs, residual = fit_lin_ocp_slope(cdf, 'volt')
                if residual[0] < 1e-5:
                    ch_df.loc[k, 'dOCPdT'] = fit_coeffs[0]
        return ch_df

    def check_ici_step(self, step):
        stp_df = self.ici_df[self.ici_df.step == step]
        mean_curr = stp_df.curr.mean()
        first_curr = stp_df.loc[stp_df.first_valid_index(), 'curr']
        last_curr = stp_df.loc[stp_df.last_valid_index(), 'curr']
        duration = stp_df.float_time.max() - stp_df.float_time.min()
        if abs((first_curr + last_curr) / 2 - mean_curr) < 1e-2 and abs(mean_curr) > 0 and abs(duration - 300) < 10:
            return True
        else:
            return False

    def calc_step_res(self):
        dchg_curr = 1
        for st in self.ici_df.step.unique():
            if st > 1:
                beg_ind = self.ici_df[self.ici_df.step == st].first_valid_index()
                stp_ind = self.ici_df[self.ici_df.step == st].last_valid_index()
                stp_curr = self.ici_df[self.ici_df.step == st]['curr'].mean()
                if stp_curr != 0:
                    dchg_curr = stp_curr
                    fin_volt = self.ici_df.loc[stp_ind, 'volt']
                elif stp_curr == 0 and self.check_ici_step(st - 1):
                    self.ici_df.loc[beg_ind, 'R0'] = 1e3 * (fin_volt - self.ici_df.loc[beg_ind, 'volt']) / dchg_curr
                    self.ici_df.loc[stp_ind, 'R10'] = 1e3 * (fin_volt - self.ici_df.loc[stp_ind, 'volt']) / dchg_curr
        return self.ici_df

    def split_df_to_dicts(self):
        # Map step_mode to the respective dictionary
        step_mode_dict = {self.settings['rest_instructions']: self.rest_dict,
                          self.settings['chrg_indicator']: self.chrg_dict,
                          self.settings['dchg_indicator']: self.dchg_dict}
        # Iterate through each step_mode in ch_df
        for step_mode, group in self.ch_df.groupby('step_mode'):
            for step in group['step_nbr']:
                # Filter rows in df corresponding to the current step
                step_df = self.ici_df[self.ici_df[self.settings['step_counter']] == step]
                # Add the dataframe to the corresponding step_mode dictionary
                step_mode_dict[step_mode][step] = step_df

    def extract_ica_from_ici(self, col='volt'):
        v_flt = 5e-3
        if self.settings['mode_column'] in self.ici_df.columns:
            mode_column = self.settings['mode_column']
            mode_mapping = {
                'chrg' : {'first_index': self.ici_df[
                    self.ici_df[mode_column] == self.settings['charge_instructions']].first_valid_index(),
                                               'volt_mark': 0,
                                               'mAh_factor': 1},
                'dchg': {'first_index': self.ici_df[
                    self.ici_df[mode_column] == self.settings['discharge_instructions']].first_valid_index(),
                                                  'volt_mark': 5,
                                                  'mAh_factor': -1}
            }
        else:
            raise ValueError(f"Dataframe must contain the '{self.settings['mode_column']}' column")

        chrg_first_idx = mode_mapping['chrg']['first_index']
        dchg_first_idx = mode_mapping['dchg']['first_index']

        if chrg_first_idx < dchg_first_idx:
            df_chrg = self.ici_df.loc[:dchg_first_idx - 1]
            df_dchg = self.ici_df.loc[dchg_first_idx:]
        else:
            df_chrg = self.ici_df.loc[chrg_first_idx:]
            df_dchg = self.ici_df.loc[:chrg_first_idx - 1]

        # Process charge and discharge dataframes
        dfs = {}
        for mode in ['chrg', 'dchg']:
            voltages = (df_chrg if mode == 'chrg' else df_dchg)[col].values
            indices = (df_chrg if mode == 'chrg' else df_dchg).index.values
            idx_to_keep = self.filter_indices_positive(voltages, indices, mode, v_flt)

            ica_df = self.ici_df.loc[idx_to_keep, :].copy()

            # Compute the gradient for 'ica_raw' and apply Gaussian filtering
            ica_df['ica_raw'] = np.gradient(mode_mapping[mode]['mAh_factor'] * ica_df['mAh'], ica_df['volt'])
            ica_df['ica_gauss'] = gaussian_filter1d(ica_df['ica_raw'], sigma=1.3, mode='nearest')

            dfs[mode] = ica_df

        # Merge the charge and discharge dataframes
        ica_df = pd.concat(dfs.values(), axis=0, ignore_index=True)
        return ica_df

    def filter_indices_positive(self, voltages, indices, mode, v_flt):
        idx_to_retain = []
        volt_mark = 0 if mode == 'chrg' else 5

        for i in range(len(voltages)):
            if (mode == 'chrg' and voltages[i] > volt_mark + v_flt) or \
                    (mode == 'dchg' and voltages[i] < volt_mark - v_flt):
                volt_mark = voltages[i]
                idx_to_retain.append(indices[i])

        return np.array(idx_to_retain)

    def perform_ici_analysis(self):
        # Perform ICI analysis
        ica_df = self.extract_ica_from_ici(col='volt')
        proc_df = self.categorize_step()
        proc_df = self.find_ici_parameters(self.ici_df, proc_df)
        return proc_df, ica_df


if __name__ == '__main__':
    from check_current_os import get_base_path_batt_lab_data
    import os
    plt.style.use(['widthsixinches', 'ml_colors', 'noframe'])
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times New Roman",
        "text.latex.preamble": r'\usepackage{siunitx}'
    })
    BASE_PATH = get_base_path_batt_lab_data()
    ici_file = r"pulse_chrg_test\cycling_data_repaired\pickle_files_channel_240095_2_1_1000mHz_no_dchg_pulse\240095_2_1_ici_dump_rpt_1.pkl"
    neware_ici = os.path.join(BASE_PATH, ici_file)
    neware_ici_analysis = ICIAnalysis(neware_ici)
    pdfn, idfn = neware_ici_analysis.perform_ici_analysis()
    print('Neware case completed without issues')
    pec_ici_file = r"pulse_chrg_test\high_frequency_testing\pec_ici_example.pkl"
    pec_ici = os.path.join(BASE_PATH, pec_ici_file)
    pec_ici_analysis = ICIAnalysis(pec_ici)
    pdfp, idfp = pec_ici_analysis.perform_ici_analysis()
    print('PEC case completed without issues')
    plt.figure()
    for ch_case in ['dchg']:
        plt.scatter(pdfn[pdfn.ici_mode==ch_case].maxV, pdfn[pdfn.ici_mode==ch_case].k, label=f'Neware_{ch_case}')
        plt.scatter(pdfp[pdfp.ici_mode==ch_case].maxV, pdfp[pdfp.ici_mode==ch_case].k, label=f'PEC_{ch_case}')
    plt.legend()
    plt.xlabel('Voltage [V]')
    plt.ylabel(r'Diffusive resistance [$\unit{\milli\ohm\per\sqrt\second}$]')
