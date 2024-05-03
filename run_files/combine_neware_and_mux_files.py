import pandas as pd
from scipy.interpolate import interp1d
from test_data_analysis.read_mux_file_to_df import convert_mux_log_to_df
from test_data_analysis.read_neware_file import read_neware_80_xls


def merge_data_with_time_stamp(target_df, source_df, target_timestamp_col, source_timestamp_col, columns_to_import):
    merged_df = pd.merge_asof(target_df.sort_values(target_timestamp_col),
                              source_df.sort_values(source_timestamp_col),
                              left_on=target_timestamp_col,
                              right_on=source_timestamp_col,
                              direction='nearest')
    for col in columns_to_import:
        target_df[col] = merged_df[col]
    return target_df


def find_extreme_mean_column(dataframe, find_max=True):
    """
    Find the column with the highest or lowest mean value in a DataFrame.

    Parameters:
    - dataframe: pd.DataFrame
        The input DataFrame.
    - find_max: bool, optional (default=True)
        If True, find the column with the highest mean value.
        If False, find the column with the lowest mean value.

    Returns:
    - str
        The name of the column with the highest or lowest mean value.
    """
    # Calculate mean values for each column
    column_means = dataframe.mean()

    # Find the column with the highest or lowest mean value
    if find_max:
        extreme_mean_column = column_means.idxmax()
    else:
        extreme_mean_column = column_means.idxmin()

    return extreme_mean_column


def merge_neware_and_mux(fname_mux, fname_neware):
    mux_df = convert_mux_log_to_df(fname_mux)
    pe_col_name = find_extreme_mean_column(mux_df.filter(like='voltage'))
    ne_col_name = find_extreme_mean_column(mux_df.filter(like='voltage'), find_max=False)
    pe_volt_int = interp1d(mux_df['time1'], mux_df[pe_col_name], fill_value='extrapolate')
    ne_volt_int = interp1d(mux_df['time2'], mux_df[ne_col_name], fill_value='extrapolate')
    mux_df['derived_voltage'] = pe_volt_int(mux_df['time1']) - ne_volt_int(mux_df['time1'])
    ndf = read_neware_80_xls(fname_neware)
    df_merged = merge_data_with_time_stamp(ndf, mux_df,
                                           target_timestamp_col='abs_time',
                                           source_timestamp_col='time1_Local_DateTime',
                                           columns_to_import=['voltage1', 'voltage2', 'derived_voltage'])
    return df_merged


def merge_nidaq_and_neware(fname_nidaq, fname_neware, pe_col_name='', ne_col_name=''):
    nidaq_df = pd.read_csv(fname_nidaq)
    nidaq_df['nidaq_abs_time'] = pd.to_datetime(nidaq_df.time, unit='s', utc=True).dt.tz_convert('Europe/Stockholm')
    neware_df = read_neware_80_xls(fname_neware)
    neware_df['abs_time_tz'] = neware_df.abs_time.dt.tz_localize('Europe/Stockholm')
    import_cols = []
    for col in [pe_col_name, ne_col_name]:
        if col:
            import_cols.append(col)
    df_merged = merge_data_with_time_stamp(neware_df, nidaq_df,
                                           target_timestamp_col='abs_time_tz',
                                           source_timestamp_col='nidaq_abs_time',
                                           columns_to_import=import_cols)
    df_merged.rename(columns={pe_col_name: 'aux_pos', ne_col_name: 'aux_neg'}, inplace=True)
    return df_merged


def main():
    import matplotlib.pyplot as plt
    mux_log = (r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\HaliBatt\SiGr_materials\ICI_msmt\voltage_"
               r"data_2024-02-27_12_07_18_984.csv")
    fname_neware = (r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\HaliBatt\SiGr_materials\ICI"
                    r"_msmt\240072-1-1-2818575915.xlsx")
    fname_nidaq = (r"\\sol.ita.chalmers.se\groups\batt_lab_data\HaliBatt\SiGr_Materials\ICI_data"
                   r"\log_data_2024-04-15_10-01-23\output_data.csv")
    ici_neware_1 = (r"\\sol.ita.chalmers.se\groups\batt_lab_data\HaliBatt\SiGr_Materials\ICI_data"
                    r"\240072-1-1-2818575938.xlsx")

    # combined_df = merge_neware_and_mux(mux_log, fname_neware)
    merged_df = merge_nidaq_and_neware(fname_nidaq, ici_neware_1,
                                       pe_col_name='cDAQ1Mod2/ai0',
                                       ne_col_name='cDAQ1Mod2/ai1')
    visual_inspection = 0
    if visual_inspection:
        plt.plot(combined_df['abs_time'], combined_df['voltage1'], label='Neg electrode')
        plt.plot(combined_df['abs_time'], combined_df['voltage2'], label='Pos electrode')
        plt.plot(combined_df['abs_time'], combined_df['volt'], label='Full cell')
        plt.plot(combined_df['abs_time'], combined_df['derived_voltage'], label='Derived voltage')
        plt.legend()


if __name__ == '__main__':
    main()
