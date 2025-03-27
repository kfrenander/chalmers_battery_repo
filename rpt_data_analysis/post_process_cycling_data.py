import os
import re
import pandas as pd
import numpy as np
from scipy.integrate import trapezoid
from misc_classes.test_metadata_reader import MetadataReader
from scipy.optimize import curve_fit, root_scalar
from test_data_analysis.ici_analysis_class import ICIAnalysis


def initiate_test_names_pulse_charge():
    test_name_list = [
        '1000mHz Pulse Charge',
        '1000mHz Pulse Charge no pulse discharge',
        '500mHz Pulse Charge',
        '320mHz Pulse Charge',
        '100mHz Pulse Charge',
        '100mHz Pulse Charge no pulse discharge',
        '10mHz Pulse Charge',
        'Reference test constant current',
    ]
    return test_name_list


def extract_rpt_from_filepath(filepath):
    rpt_pattern = r'rpt_\d+'
    filename = os.path.split(filepath)[-1]
    return re.findall(rpt_pattern, filename)[0]


def extract_channel_from_file(filepath):
    try:
        fname = os.path.split(filepath)[-1]
        tester_channel = re.findall(r'\d{6}_\d_\d', fname)
        if tester_channel:
            return tester_channel[0]
        else:
            print(f'No channel ID found in string {fname}')
            return None
    except Exception as e:
        print('An error occurred:', e)
        return None


class CycleAgeingDataIndexer:

    def __init__(self):
        self.top_dir = None
        self.directory_file_dict = None
        self.test_naming = None
        self.ageing_data = None
        self.combined_data = None
        self.combined_data_arbitrary_replicates = None
        self.visual_profile = VisualProfileAllAgeingTests()

    def index_data(self):
        self.directory_file_dict = {}
        for root, dirs, files in os.walk(self.top_dir):
            if 'pickle' in root:
                f_list = []
                chnl_id = extract_channel_from_file(root)
                for file in files:
                    file_path = os.path.join(root, file)
                    f_list.append(file_path)
                self.directory_file_dict[chnl_id] = f_list
        return None

    # def initiate_channel_test_mapping(self):
    #     chnl_list = [f'240095_{k}_{i}' for k in range(2, 4) for i in range(1, 9)]
    #     test_name_list = [
    #         '1000mHz',
    #         '1000mHz',
    #         '1000mHz_no_pulse',
    #         '1000mHz_no_pulse',
    #         '500mHz',
    #         '500mHz',
    #         '320mHz',
    #         '320mHz',
    #         '100mHz',
    #         '100mHz',
    #         '100mHz_no_pulse',
    #         '100mHz_no_pulse',
    #         '10mHz',
    #         '10mHz',
    #         'CC reference',
    #         'CC reference'
    #     ]
    #     self.test_naming = dict(zip(chnl_list, test_name_list))
    #     return None

    def fill_ageing_data(self):
        self.ageing_data = {}
        for idx, file_list in self.directory_file_dict.items():
            try:
                self.ageing_data[idx] = CycleAgeingDataReader(idx, file_list)
            except Exception as e:
                print('An error occurred:', e)

    def generate_replicate_combined_data(self):
        test_meta_data = MetadataReader()
        TEST_NAMES = test_meta_data.excel_data.TEST_CONDITION.unique()
        self.combined_data = {}
        for tn in TEST_NAMES:
            tmp_list = []
            for ch_id, ag_data in self.ageing_data.items():
                if ag_data.meta_data.test_condition == tn:
                    tmp_list.append(ag_data.rpt_data)
            print(len(tmp_list), tn)
            if tmp_list:
                combined_df = pd.merge_asof(tmp_list[0], tmp_list[1],
                                            left_on='fce',
                                            right_on='fce',
                                            suffixes=('_cell1', '_cell2'))
                combined_df['avg_rel_cap'] = combined_df[['cap_relative_cell1', 'cap_relative_cell2']].mean(axis=1)
                combined_df['std_rel_cap'] = combined_df[['cap_relative_cell1', 'cap_relative_cell2']].std(axis=1)
                self.combined_data[tn] = combined_df
        return None

    def generate_arbitrary_replicates_combined_data(self):
        test_meta_data = MetadataReader()
        TEST_NAMES = test_meta_data.excel_data.TEST_CONDITION.unique()
        self.combined_data_arbitrary_replicates = {}
        for tn in TEST_NAMES:
            tmp_list = []
            for ch_id, ag_data in self.ageing_data.items():
                if ag_data.meta_data.test_condition == tn:
                    tmp_list.append(ag_data.rpt_data)
            if tmp_list:
                suffix_list = [f"_cell{i + 1}" for i in range(len(tmp_list))]

                # Start with the first dataframe in the list
                combined_df = tmp_list[0]

                # Merge the rest of the dataframes
                for i in range(1, len(tmp_list)):
                    combined_df = pd.merge_asof(
                        combined_df,
                        tmp_list[i],
                        left_on='fce',
                        right_on='fce',
                        suffixes=('', suffix_list[i])
                    )
                filtered_columns = [
                    col for col in combined_df.columns
                    if 'cap_relative' in col and 'sigma_cap_relative' not in col
                ]
                combined_df['avg_rel_cap'] = combined_df[filtered_columns].mean(axis=1)
                combined_df['std_rel_cap'] = combined_df[filtered_columns].std(axis=1)
                self.combined_data_arbitrary_replicates[tn] = combined_df
        return None

    def run(self, top_directory, project_key='pulse_charging'):
        self.top_dir = top_directory
        self.test_naming = initiate_test_names_pulse_charge()
        self.index_data()
        self.fill_ageing_data()
        self.generate_replicate_combined_data()
        self.generate_arbitrary_replicates_combined_data()


class CycleAgeingDataReader:

    def __init__(self, chnl_id=None, list_of_files=None):
        self.chnl_id = chnl_id
        self.TEST_NAME = None
        self.pkl_files = list_of_files
        self.rpt_data = None
        self.ica_data = None
        self.ici_data = None
        self.dyn_data = None
        self.meta_data = None
        self.average_temperature = None
        self.popt = None
        self.pcov = None
        self.read_rpt_summary()
        self.read_ica_data()
        self.read_ici_data()
        self.read_metadata()
        self.popt, self.pcov = self.fit_degradation_function()
        self.TEST_NAME = self.meta_data.test_condition
        self.visual_profile = VisualProfileUniqueTest(test_name=self.TEST_NAME)

    def read_rpt_summary(self):
        found_summary = False
        for pkl_file in self.pkl_files:
            if 'rpt_summary' in pkl_file:
                try:
                    self.rpt_data = pd.read_pickle(pkl_file)
                    if not self.rpt_data['date'].is_monotonic_increasing:
                        print(f'!!! FOUND NON-MONOTONIC TIME SERIES IN {self.chnl_id} FOR TEST {self.TEST_NAME}!!!')
                        self.rpt_data = self.rpt_data.sort_values(by='date').reset_index(drop=True)
                        rpt_idx_update = [f'rpt_{i}' for i in range(1, self.rpt_data.shape[0] + 1)]
                        self.rpt_data = self.rpt_data.set_axis(rpt_idx_update)
                    self.rpt_data['fce'] = np.arange(0, self.rpt_data.shape[0]*40, 40)

                    found_summary = True
                except FileNotFoundError:
                    print(f"File '{pkl_file}' not found.")
                except PermissionError:
                    print(f"No permission to access file '{pkl_file}'.")
                except IOError:
                    print(f"I/O error while reading file '{pkl_file}'.")
                except EOFError:
                    print(f"End of file reached unexpectedly while reading '{pkl_file}'.")
                except (ValueError, TypeError):
                    print(f"Error reading data from file '{pkl_file}'.")
                except MemoryError:
                    print(f"Not enough memory to read file '{pkl_file}'.")
                except Exception as e:
                    print(f"An error occurred while reading file '{pkl_file}': {e}")
        if not found_summary:
            print(f'No rpt summary file found for data from channel {self.chnl_id}.\nPlease check data.')

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
        x_data = self.rpt_data.fce.astype('float')
        y_data = self.rpt_data.cap_relative.astype('float')
        popt, pcov = curve_fit(self._q_function, x_data, y_data, p0=[1, 1000, 1])
        return popt, pcov

    def read_metadata(self):
        for file in self.pkl_files:
            if 'metadata' in file:
                try:
                    self.meta_data = MetadataReader(file_path=file)
                except Exception as e:
                    print(f'Error occured when reading metadata from {file}: {e}')

    def read_ica_data(self):
        self.ica_data = {}
        found_ica = False
        for pkl_file in self.pkl_files:
            if 'ica_dump' in pkl_file:
                found_ica = True
                rpt_idx = ''
                try:
                    rpt_idx = extract_rpt_from_filepath(pkl_file)
                except re.error:
                    print('Not able to resolve regex for identifying RPT index')
                except (TypeError, AttributeError, SyntaxError, ValueError, UnicodeError, IndexError) as e:
                    print(f'An error occurred when trying to find RPT index for {pkl_file}: {e}')
                except Exception as e:
                    print(f'Unexpected error occurred when trying to find RPT index for {pkl_file}: {e}')
                if rpt_idx:
                    try:
                        self.ica_data[rpt_idx] = pd.read_pickle(pkl_file)
                    except (PermissionError, IOError, EOFError, ValueError, TypeError, MemoryError) as e:
                        print(f"An error occurred while reading file '{pkl_file}': {e}")
                    except Exception as e:
                        print(f"An error occurred while reading file '{pkl_file}': {e}")
        if not found_ica:
            print(f'No ica files found for data from channel {self.chnl_id}.\nPlease check data.')

    def read_ici_data(self):
        self.ici_data = {}
        found_ici = False
        for pkl_file in self.pkl_files:
            if 'ici_dump' in pkl_file:
                found_ici = True
                rpt_idx = ''
                try:
                    rpt_idx = extract_rpt_from_filepath(pkl_file)
                except re.error:
                    print('Not able to resolve regex for identifying RPT index')
                except (TypeError, AttributeError, SyntaxError, ValueError, UnicodeError, IndexError) as e:
                    print(f'An error occurred when trying to find RPT index for {pkl_file}: {e}')
                except Exception as e:
                    print(f'Unexpected error occurred when trying to find RPT index for {pkl_file}: {e}')
                if rpt_idx:
                    try:
                        df = pd.read_pickle(pkl_file)
                        self.ici_data[rpt_idx] = ICIAnalysis(df)
                        tmp_output = self.ici_data[rpt_idx].perform_ici_analysis()
                    except (PermissionError, IOError, EOFError, ValueError, TypeError, MemoryError) as e:
                        print(f"An error occurred while reading file '{pkl_file}': {e}")
                    except Exception as e:
                        print(f"An error occurred while reading file '{pkl_file}': {e}")
        if not found_ici:
            print(f'No ici files found for data from channel {self.chnl_id}.\nPlease check data.')

    def read_dynamic_data(self):
        self.dyn_data = {}
        found_dynamic_data = False
        for pkl_file in self.pkl_files:
            if 'dyn_df' in pkl_file:
                found_dynamic_data = True
                try:
                    self.dyn_data = pd.read_pickle(pkl_file)
                except (PermissionError, IOError, EOFError, ValueError, TypeError, MemoryError) as e:
                    print(f"An error occurred while reading file '{pkl_file}': {e}")
                except Exception as e:
                    print(f"An error occurred while reading file '{pkl_file}': {e}")
        if not found_dynamic_data:
            print(f'No dynamic data files found for data from channel {self.chnl_id}.\nPlease check data.')

    def make_temperature_summary(self):
        if not isinstance(self.dyn_data, pd.DataFrame):
            self.read_dynamic_data()
        if 'temperature' in self.dyn_data.columns:
            self.average_temperature = (trapezoid(self.dyn_data.temperature, self.dyn_data.float_time)
                                        / self.dyn_data.float_time.max())
        elif 'aux_T_1' in self.dyn_data.columns:
            self.average_temperature = (trapezoid(self.dyn_data.aux_T_1, self.dyn_data.float_time)
                                        / self.dyn_data.float_time.max())
        else:
            print(f'No temperature data found in dynamic data for {self.chnl_id}')
        # TODO: Implement some kind of histogram for temperature visualisation

    def find_fce_at_given_q(self, q_target):
        """
        Find the time at which the fitted Q(t) function yields a given Q.
        Args:
            q_target (float): Target capacity
        Returns:
            fce (float): Cycles corresponding to the target Q
        """
        if self.popt is None:
            raise ValueError("You must fit the function before calling this method.")

        # Solve for t such that Q(fce) = q_target
        def equation(fce):
            return q_target - self._q_function(fce, *self.popt)

        result = root_scalar(equation, bracket=[1e-5, 1e5], method='brentq')
        if result.converged:
            return result.root
        else:
            raise RuntimeError("Failed to converge to a solution.")

    def _q_function(self, fce, q0, tau, beta):
        """The function to fit: Q(fce) = q0 * exp(-(fce/tau)^beta)"""
        fce = np.asarray(fce)
        return q0 * np.exp(-(fce / tau) ** beta)


class VisualProfileAllAgeingTests:

    def __init__(self):
        self.TEST_NAMES = initiate_test_names_pulse_charge()
        self.COLORS = self.initiate_color_dictionary()
        self.LINE_STYLES = self.initiate_line_styles()
        self.MARKERS = self.initiate_markers()

    def initiate_color_dictionary(self):
        color_list = [
            '#0072bd',
            '#d95319',
            '#edb80f',
            '#7e2f8e',
            '#779a30',
            '#4cbfe6',
            '#a3192f',
            '#000000'
        ]
        return dict(zip(self.TEST_NAMES, color_list))

    def initiate_line_styles(self):
        line_style_list = [
            (0, ()),
            (0, (10, 5)),
            (0, (1, 5)),
            (0, (10, 5, 1, 5)),
            (0, (10, 5, 1, 5, 1, 5)),
            (0, (20, 10)),
            (0, (5, 2)),
            (0, (15, 7, 5, 5, 2, 5))
        ]
        return dict(zip(self.TEST_NAMES, line_style_list))

    def initiate_markers(self):
        marker_list = [
            'o',
            's',
            'd',
            '^',
            'v',
            'x',
            '+',
            '*'
        ]
        return dict(zip(self.TEST_NAMES, marker_list))


class VisualProfileUniqueTest(VisualProfileAllAgeingTests):

    def __init__(self, test_name=None):
        super().__init__()
        self.COLOR = None
        self.MARKER = None
        self.LINE_STYLE = None
        self.TEST_NAME = test_name
        self.set_unique_color()
        self.set_unique_marker()

    def set_unique_color(self):
        try:
            self.COLOR = self.COLORS[self.TEST_NAME]
        except KeyError as e:
            print(f'Could not set unique color for {self.TEST_NAME}, yields: {e}')
        except Exception as e:
            print(f'Unexpected error occured: {e}')
        if not self.COLOR:
            print(f'Defaulting color to forestgreen (#228B22)')
            self.COLOR = '#228B22'

    def set_unique_marker(self):
        try:
            self.MARKER = self.MARKERS[self.TEST_NAME]
        except KeyError as e:
            print(f'Could not set unique marker for {self.TEST_NAME}, yields: {e}')
        except Exception as e:
            print(f'Unexpected error occured: {e}')
        if not self.MARKER:
            print(f'Defaulting marker to square (.)')
            self.MARKER = '.'

    def set_unique_line_style(self):
        try:
            self.LINE_STYLE = self.LINE_STYLES[self.TEST_NAME]
        except KeyError as e:
            print(f'Could not set unique line style for {self.TEST_NAME}, yields: {e}')
        except Exception as e:
            print(f'Unexpected error occured: {e}')
        if not self.LINE_STYLE:
            print(f'Defaulting line style to solid ')
            self.LINE_STYLE = 'solid'
