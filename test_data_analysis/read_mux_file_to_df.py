from datetime import datetime
import pandas as pd


def utc_unix_time_to_local_datetime(unix_time):
    return datetime.utcfromtimestamp(unix_time / 1e3)


def local_unix_time_to_local_datetime(unix_time):
    return datetime.fromtimestamp(unix_time / 1e3)


def convert_unix_time_columns_to_local_datetime(df, unix_time_columns, local_time):
    if isinstance(unix_time_columns, str):  # Check if a single column is provided
        unix_time_columns = [unix_time_columns]  # Convert single column to list
    for col in unix_time_columns:
        if local_time:
            df[col + '_Local_DateTime'] = df[col].apply(local_unix_time_to_local_datetime)
        else:
            df[col + '_Local_DateTime'] = df[col].apply(utc_unix_time_to_local_datetime)
    return df


def convert_mux_log_to_df(mux_log):
    df = pd.read_csv(mux_log, names=['time1', 'voltage1', 'time2', 'voltage2'])
    if mux_log.endswith('.csv'):
        local_time=True
    else:
        local_time=False
    df = convert_unix_time_columns_to_local_datetime(df, ['time1', 'time2'], local_time=local_time)
    for col in df.columns:
        if 'voltage' in col:
            df[col] = filter_sudden_jumps(df[col], 0.5)
    return df


def convert_individual_mux_log_to_df(file_name):
    df = pd.read_csv(file_name, names=['time1', 'voltage1'])
    if mux_log.endswith('.csv'):
        local_time = True
    else:
        local_time = False
    df = convert_unix_time_columns_to_local_datetime(df, ['time1'], local_time=local_time)
    for col in df.columns:
        if 'voltage' in col:
            df[col] = filter_sudden_jumps(df[col], 0.5)
    return df


def filter_sudden_jumps(series, threshold):
    filtered_indices = [True]  # Assume the first data point is not a jump
    for i in range(1, len(series)):
        if abs(series[i] - series[i-1]) > threshold:
            filtered_indices.append(False)  # Mark as False if it's a jump
        else:
            filtered_indices.append(True)  # Otherwise, keep it
    return series[filtered_indices]


def read_mux_log(file_path):
    csv_sample = pd.read_csv(file_path, nrows=3)
    num_columns = len(csv_sample.columns)
    if num_columns == 4:
        return convert_mux_log_to_df(file_path)
    elif num_columns == 2:
        return convert_individual_mux_log_to_df(file_path)


if __name__ == '__main__':
    mux_log = r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\HaliBatt\SiGr_materials\CtrlMsmt\2024     2    14    11    22    29"
    csv_log = r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\HaliBatt\SiGr_materials\ICI_msmt\voltage_data_2024-02-27_12_07_18_984.csv"
    csv_individual_log = r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\HaliBatt\SiGr_materials\ICI_msmt\voltage_data_device25_2024-03-06_14_27_08_747.csv"
    df = read_mux_log(mux_log)
    df.filter(like='voltage').plot()
    df_python = read_mux_log(csv_log)
    df_python.filter(like='voltage').plot()
    df_indvdl = read_mux_log(csv_individual_log)
