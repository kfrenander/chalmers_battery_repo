import pandas as pd
import re
from io import StringIO
import pytz
import numpy as np


class NewareData9000:

    def __init__(self):
        self.test_info = None
        self.step_info = None
        self.cyc_info = None
        self.dyn_data = None
        self.xl_file = None
        self.rpt_step_range = None

    def load_neware_data(self, filename):
        self.xl_file = pd.ExcelFile(filename)
        self.call_constructors()
        self.call_data_loaders()

    def call_constructors(self):
        self.test_info = TestInformation(self.xl_file.parse('Test'))
        self.step_info = StepInformation()
        self.cyc_info = CyclingInformation()
        self.dyn_data = DynamicNewareData()

    def call_data_loaders(self):
        self.test_info.parse_input()
        self.step_info.load_step_data(self.xl_file.parse('Step', skiprows=1))
        self.cyc_info.load_cycle_data(self.xl_file.parse('Cycle', skiprows=1))
        self.dyn_data.load_dynamic_data(self.xl_file.parse('Record', skiprows=1))

    def find_rpt_step_range(self):
        strt_idx = self.step_info.identify_rpt_start_index()
        stop_idx = self.step_info.identify_rpt_stop_index()
        self.rpt_step_range = range(strt_idx, stop_idx)

    def _extract_dynamic_rpt(self):
        if not self.rpt_step_range:
            self.find_rpt_step_range()
        return self.dyn_data.dyn_data[self.dyn_data.dyn_data.step_nbr.isin(self.rpt_step_range)]

    def _extract_step_rpt(self):
        if not self.rpt_step_range:
            self.find_rpt_step_range()
        return self.step_info.step_characteristics[self.step_info.step_characteristics.StepID.isin(self.rpt_step_range)]


class TestInformation:

    def __init__(self, input_df):
        self.data_as_string = input_df.to_string(index=False)
        self.channel = ""
        self.model = ""
        self.batch = ""
        self.test_name = ""
        self.operator = ""
        self.end_time = ""
        self.data_sn = ""
        self.specific_capacity = ""
        self.active_materials = ""
        self.range = ""
        self.begin_time = ""
        self.step_info = StepInformationFromTestSheet(self.data_as_string)

    def parse_input(self):
        input_data = self.data_as_string
        step_line = find_substring_line(input_data, 'Step ID')
        lines = input_data.split('\n')
        for line in lines[:step_line]:
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) > 1:
                for ln, part in enumerate(parts):
                    if 'Chl' in part:
                        self.channel = self.check_for_nan_string(parts, ln)
                    elif 'Model' in part:
                        self.model = self.check_for_nan_string(parts, ln)
                    elif 'Batch' in part:
                        self.batch = self.check_for_nan_string(parts, ln)
                    elif 'Operator' in part:
                        self.operator = self.check_for_nan_string(parts, ln)
                    elif 'Begin Time' in part:
                        self.begin_time = self.check_for_nan_string(parts, ln)
                    elif 'End Time' in part:
                        self.end_time = self.check_for_nan_string(parts, ln)
                    elif 'DataSN' in part:
                        self.data_sn = self.check_for_nan_string(parts, ln)
                    elif 'Specific Capacity' in part:
                        self.specific_capacity = self.check_for_nan_string(parts, ln)
                    elif 'Active Materials' in part:
                        self.active_materials = self.check_for_nan_string(parts, ln)
                    elif 'Test Name' in part:
                        self.test_name = self.check_for_nan_string(parts, ln)
                    elif 'Range' in part:
                        self.range = self.check_for_nan_string(parts, ln)

    @staticmethod
    def check_for_nan_string(lst_of_strings, idx):
        if 'NaN' in lst_of_strings[idx + 1]:
            split_ = re.split(r':', lst_of_strings[idx])
            return '.'.join(split_[1:])
        else:
            return lst_of_strings[idx + 1]


class StepInformationFromTestSheet:

    def __init__(self, input_string):
        self.input_string = input_string
        self.sub_string = self.extract_substring()
        self.step_df = None
        self.read_step_text()

    def extract_substring(self):
        line_to_start = find_substring_line(self.input_string, 'Step ID')
        lines = self.input_string.split('\n')  # Split the string into lines
        sub_string_list = []
        for i, line in enumerate(lines):
            if i >= line_to_start:
                sub_string_list.append(line)
        return '\n'.join(sub_string_list)

    def read_step_text(self):
        self.step_df = pd.read_fwf(StringIO(self.sub_string)).dropna(subset=['Step Name']).dropna(how='all', axis=1)


class CyclingInformation:

    def __init__(self):
        self.cycle_data = None

    def load_cycle_data(self, data):
        if isinstance(data, pd.DataFrame):
            self.cycle_data = data
        else:
            raise ValueError("Input data must be a pandas DataFrame")


class StepInformation:

    def __init__(self):
        self.step_characteristics = None
        self.column_mapping = {}
        self.unit_conversion = {}
        self._init_column_mapping()
        self._init_unit_conversion()

    def identify_rpt_start_index(self):
        tmp_df = self.step_characteristics
        condition1 = tmp_df['StepDuration(s)'] == 28
        condition2 = tmp_df['StepType'] == 'Rest'
        return tmp_df[condition1 & condition2].StepID.values[0]

    def identify_rpt_stop_index(self):
        tmp_df = self.step_characteristics
        condition1 = tmp_df['StepDuration(s)'] == 34
        condition2 = tmp_df['StepType'] == 'Rest'
        return tmp_df[condition1 & condition2].StepID.values[0]

    def load_step_data(self, data):
        if isinstance(data, pd.DataFrame):
            self.step_characteristics = data
            self.rename_step_columns()
            self.rescale_step_units()
            self.convert_duration_to_seconds()
        else:
            raise ValueError("Input data must be a pandas DataFrame")

    def rename_step_columns(self):
        if self.step_characteristics is None:
            raise ValueError("DataFrame is not loaded. Please load data first.")
        self.step_characteristics.columns = self.step_characteristics.columns.map(self._rename_column)

    def rescale_step_units(self):
        if self.step_characteristics is None:
            raise ValueError("DataFrame is not loaded. Please load data first.")
        self.step_characteristics.columns = self.step_characteristics.columns.map(self._rescale_unit)

    def convert_duration_to_seconds(self):
        if self.step_characteristics is None:
            raise ValueError("DataFrame is not loaded. Please load data first.")
        for col in self.step_characteristics.columns:
            if '(H:M:S.mS)' in col:
                name, unit = re.match(r'(.+)\((.+)\)', col).groups()
                self.step_characteristics[f'{name}(s)'] = self.step_characteristics[col].apply(self._parse_duration)

    def _parse_duration(self, duration_str):
        # Splitting the duration string into hours, minutes, seconds, and milliseconds
        hours, minutes, seconds_ms = duration_str.split(':')
        seconds, milliseconds, microseconds = seconds_ms.split('.')
        # Calculating total duration in seconds
        total_seconds = (int(hours) * 3600 + int(minutes) * 60 + int(seconds) +
                         int(milliseconds) / 1000 + int(microseconds) / 1e6)
        return total_seconds

    def _rename_column(self, column):
        if column in self.column_mapping:
            return self.column_mapping[column]
        else:
            raise ValueError(f"Column '{column}' not found in the mapping dictionary.")

    def _rescale_unit(self, column):
        unit_match = re.match(r'(.+)\((.+)\)', column)
        if unit_match:
            name, unit = unit_match.groups()
            try:
                unit_name, unit_scale = self.unit_conversion.get(unit, unit)
                self.step_characteristics[column] = self.step_characteristics[column].apply(lambda x: x * unit_scale)
            except TypeError:
                unit_name = self.unit_conversion.get(unit, unit)
            except ValueError:
                unit_name = self.unit_conversion.get(unit, unit)
            return f'{name}({unit_name})'
        else:
            return column

    def _init_column_mapping(self):
        old_columns_names = [
            'Barcode',
            'CycleID',
            'StepID',
            'StepType',
            'Status',
            'Step Time(H:M:S.mS)',
            'TestFlowTime(H:M:S.mS)',
            'SetVolt(mV)',
            'SetCurrent(mA)',
            'StartVolt(mV)',
            'EndVolt(mV)',
            'MaxVolt(mV)',
            'MinVolt(mV)',
            'ChgMidVolt(mV)',
            'DChgMidVolt(mV)',
            'StartCurrent(mA)',
            'EndCurrent(mA)',
            'MaxCurrent(mA)',
            'MinCurrent(mA)',
            'StatrTemp(¡æ)',
            'EndTemp(¡æ)',
            'DCIR(m¦¸)',
            'ACIR(m¦¸)',
            'Capacity(mAh)',
            'CmpCapacity(mAh)',
            'Energy(mWh)',
            'CmpEnergy(mWh)',
            'CapRetentionRate(%)',
            'EndStatus',
            'CycleCnt',
            'DCIR(m¦¸).1',
            'RtcTimer'
        ]
        new_columns_names = [
            'Barcode',
            'CycleID',
            'StepID',
            'StepType',
            'Status',
            'StepDuration(H:M:S.mS)',
            'TestFlowTime(H:M:S.mS)',
            'SetVolt(mV)',
            'SetCurrent(mA)',
            'StartVolt(mV)',
            'EndVolt(mV)',
            'MaxVolt(mV)',
            'MinVolt(mV)',
            'ChgMidVolt(mV)',
            'DChgMidVolt(mV)',
            'StartCurrent(mA)',
            'EndCurrent(mA)',
            'MaxCurrent(mA)',
            'MinCurrent(mA)',
            'StartTemp(degC)',
            'EndTemp(degC)',
            'DCIR(mohm)',
            'ACIR(mohm)',
            'Capacity(mAh)',
            'CmpCapacity(mAh)',
            'Energy(mWh)',
            'CmpEnergy(mWh)',
            'CapRetentionRate(%)',
            'EndStatus',
            'CycleCnt',
            'DCIR(mohm)_1',
            'AbsoluteTime'
        ]
        self.column_mapping = dict(zip(old_columns_names, new_columns_names))

    def _init_unit_conversion(self):
        unit_name = [
            'mV',
            'mA',
            'degC',
            'mohm',
            'mWh',
            'mAh'
        ]
        unit_conversion = [
            ('V', 0.001),
            ('A', 0.001),
            ('degC', 1),
            ('ohm', 0.001),
            ('Wh', 0.001),
            ('Ah', 0.001)
        ]
        self.unit_conversion = dict(zip(unit_name, unit_conversion))


class DynamicNewareData:

    def __init__(self):
        self.dyn_data = None
        self.dynamic_column_mapping = {}
        self._init_dynamic_column_mapping()

    def load_dynamic_data(self, df):
        if isinstance(df, pd.DataFrame):
            self.dyn_data = df
            self.rename_dynamic_columns()
            self.convert_to_posix_time()
            self.convert_relative_timestamps_to_float_seconds()
        else:
            raise ValueError("Input data must be a pandas DataFrame")

    def rename_dynamic_columns(self):
        if self.dyn_data is None:
            raise ValueError("DataFrame is not loaded. Please load data first.")
        self.dyn_data.columns = self.dyn_data.columns.map(self._rename_column)

    def convert_to_posix_time(self):
        if self.dyn_data is None:
            raise ValueError("DataFrame is not loaded. Please load data first.")
        # Convert string timestamps to datetime objects
        timestamp = pd.to_datetime(self.dyn_data['RtcTimer'])
        stockholm_tz = pytz.timezone('Europe/Stockholm')
        timestamp_sthlm = timestamp.dt.tz_localize(stockholm_tz)
        self.dyn_data['posix_timestamp'] = timestamp_sthlm.astype(np.int64) / 1e9

    def convert_relative_timestamps_to_float_seconds(self):
        if self.dyn_data is None:
            raise ValueError("DataFrame is not loaded. Please load data first.")
        cols_to_convert = ['timeSinceBeginningTest', 'timeSinceBeginningStepTime', 'relative_time']
        for col in cols_to_convert:
            if col in self.dyn_data.columns:
                self.dyn_data[f'{col}_seconds'] = self.dyn_data[col].apply(self._convert_relative_to_seconds)

    @staticmethod
    def _convert_relative_to_seconds(relative_time):
        # Split the string into days, hours, minutes, and seconds and convert to floats
        days, time = relative_time.split('d ')
        hours, minutes, seconds = map(float, time.split(':'))

        # Extract milliseconds
        if '.' in str(seconds):
            seconds, milliseconds = map(float, str(seconds).split('.'))
        else:
            milliseconds = 0

        # Calculate total duration in seconds
        total_seconds = ((float(days) * 24 * 60 * 60) + (hours * 60 * 60) + (minutes * 60) +
                         seconds + (milliseconds / 1000))
        return total_seconds

    def _rename_column(self, column):
        if column in self.dynamic_column_mapping:
            return self.dynamic_column_mapping[column]
        else:
            raise ValueError(f"Column '{column}' not found in the mapping dictionary.")

    def _init_dynamic_column_mapping(self):
        dynamic_columns_auto_names = [
            'serial_number',
            'device',
            'cycle_no',
            'record_no',
            'step_no',
            'step_type',
            'Current',
            'Voltage',
            'Capacity',
            'Energy',
            'MD',
            'ES',
            'timestamp',
            'timeSinceBeginningTest',
            'timeSinceBeginningStepTime',
            'timeDataPointWasTaken',
            'relative_time',
            'CycleCnt',
            'DCIR(m¦¸)',
            'RtcTimer',
            'AuxTemp1'
        ]
        dynamic_columns_new_names = [
            'serial_number',
            'device',
            'cyc_nbr',
            'record_nbr',
            'step_nbr',
            'step_type',
            'curr',
            'volt',
            'cap',
            'egy',
            'md',
            'end_state',
            'timestamp',
            'timeSinceBeginningTest',
            'timeSinceBeginningStepTime',
            'timeDataPointWasTaken',
            'relative_time',
            'CycleCnt',
            'DCIR(mohm)',
            'RtcTimer',
            'cell_temperature'
        ]
        self.dynamic_column_mapping = dict(zip(dynamic_columns_auto_names, dynamic_columns_new_names))


class RptData:

    def __init__(self, df_dyn=pd.DataFrame(), df_stp= pd.DataFrame):
        self.rpt_summary = None
        self.df = df_dyn
        self.sdf = df_stp
        self.cap_mean = None
        self.cap_std = None

    def extract_capacity_values(self):
        cap_rows = []
        # First filter all basic properties of the discharge step
        for idx, series in self.sdf.iterrows():
            if series['StepType'] == 'CCDChg':
                if series['StepDuration(s)'] > 7200:
                    if series['MinCurrent(A)'] < -1:
                        if self.sdf.loc[idx - 2, 'StepType'] == 'CCCVChg':
                            cap_rows.append(idx)
        self.cap_mean = self.sdf.loc[cap_rows, 'Capacity(Ah)'].abs().mean()
        self.cap_std = self.sdf.loc[cap_rows, 'Capacity(Ah)'].abs().std()

    def extract_dynamic_ica_data(self):
        cond1 = (self.sdf['SetCurrent(A)'] - 0.175) < 0.05
        cond2 = self.sdf['StepDuration(s)'] > 10*3600
        ica_steps = self.sdf[cond1 & cond2].StepID
        return self.df[self.df.step_nbr.isin(ica_steps)]



def find_substring_line(string, substring):
    lines = string.split('\n')  # Split the string into lines
    for i, line in enumerate(lines):
        if substring in line:
            return i  # Return the line number where the substring is found
    return None  # Return None if the substring is not found in any line


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Excel file with output from BTS 9000 tester.
    my_file = (r"\\sol.ita.chalmers.se\groups\batt_lab_data\pulse_chrg_test\high_frequency_testing"
               r"\TrialTest-3-101-14.xlsx")
    xl = pd.ExcelFile(my_file)

    test_inf = TestInformation(xl.parse('Test'))
    test_inf.parse_input()
    substring = test_inf.step_info.extract_substring()

    cyc_info = CyclingInformation()
    cyc_info.load_cycle_data(xl.parse('Cycle', skiprows=1))

    step_info = StepInformation()
    step_info.load_step_data(xl.parse('Step', skiprows=1))

    dyn_data = DynamicNewareData()
    dyn_data.load_dynamic_data(xl.parse('Record', skiprows=1))
    dyn_data.convert_to_posix_time()

    neware_data = NewareData9000()
    neware_data.load_neware_data(my_file)

    neware_data.find_rpt_step_range()

    rpt_data = RptData(neware_data._extract_dynamic_rpt(), neware_data._extract_step_rpt())
    rpt_data.extract_capacity_values()
    ica_df = rpt_data.extract_dynamic_ica_data()

    # fig = plt.figure()
    # plt.plot(neware_data.dyn_data.dyn_data.timeSinceBeginningTest_seconds,
    #          neware_data.dyn_data.dyn_data.cell_temperature)
    fig = plt.figure()
    plt.plot(neware_data.dyn_data.dyn_data.timeSinceBeginningTest_seconds,
             neware_data.dyn_data.dyn_data.volt)
