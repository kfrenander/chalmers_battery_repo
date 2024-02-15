from datetime import datetime
import pandas as pd


def unix_time_to_local_datetime(unix_time):
    return datetime.fromtimestamp(unix_time / 1e3)


def convert_unix_time_columns_to_local_datetime(df, unix_time_columns):
    if isinstance(unix_time_columns, str):  # Check if a single column is provided
        unix_time_columns = [unix_time_columns]  # Convert single column to list
    for col in unix_time_columns:
        df[col + '_Local_DateTime'] = df[col].apply(unix_time_to_local_datetime)
    return df


def convert_mux_log_to_df(mux_log):
    df = pd.read_csv(mux_log, names=['time1', 'voltage1', 'time2', 'voltage2'])
    df = convert_unix_time_columns_to_local_datetime(df, ['time1', 'time2'])
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


def filter_sudden_jumps_in_dataframe(df, threshold):
    filtered_df = df.apply(lambda x: filter_sudden_jumps(x, threshold), axis=0)
    return filtered_df


if __name__ == '__main__':
    mux_log = r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\HaliBatt\SiGr_materials\CtrlMsmt\2024     2    14    11    22    29"
    df = convert_mux_log_to_df(mux_log)
    df.filter(like='voltage').plot()
