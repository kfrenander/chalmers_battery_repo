import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid


def read_neware_xls(file_path, calc_c_rate=False):
    xl_file = pd.ExcelFile(file_path)
    df = pd.DataFrame()
    for sheet in xl_file.sheet_names:
        if 'Detail_' in sheet:
            if df.empty:
                df = xl_file.parse(sheet)
            else:
                df = df.append(xl_file.parse(sheet), ignore_index=True)
    mA = [x for x in df.columns if '(mA)' in x]
    df.columns = ['Measurement', 'mode', 'step', 'arb_step1', 'arb_step2',
                  'curr', 'volt', 'cap', 'egy', 'rel_time', 'abs_time']
    df['step_time'] = pd.to_timedelta(df.rel_time)
    if isinstance(df['abs_time'][0], str):
        df.loc[:, 'abs_time'] = pd.to_datetime(df['abs_time'], format='%Y-%m-%d %H:%M:%S')
    df['float_step_time'] = df['step_time'].dt.seconds + df['step_time'].dt.microseconds / 1e6
    df['float_time'] = (df.abs_time - df.abs_time[0]).astype('timedelta64[s]')
    df['pwr'] = df.curr / 1000 * df.volt
    df['pwr_chrg'] = df.pwr.mask(df.pwr < 0, 0)
    df['pwr_dchg'] = df.pwr.mask(df.pwr > 0, 0)
    df['egy_tot'] = cumulative_trapezoid(df.pwr.abs() / (1000 * 3600), df.float_time, initial=0)
    df['egy_chrg'] = cumulative_trapezoid(df.pwr_chrg.abs() / (1000 * 3600), df.float_time, initial=0)
    df['egy_dchg'] = cumulative_trapezoid(df.pwr_dchg.abs() / (1000 * 3600), df.float_time, initial=0)
    if mA:
        df['mAh'] = cumulative_trapezoid(df.curr, df.float_time, initial=0) / 3600
        df['curr'] = df.curr / 1000
    else:
        df['mAh'] = cumulative_trapezoid(df.curr, df.float_time, initial=0) * 1000 / 3600
        df['cap'] = df['cap'] * 1000
    if calc_c_rate:
        if mA:
            c_rate = 1000 * df.groupby('arb_step2').curr.mean().abs() / df.groupby('arb_step2').cap.max().max()
        else:
            c_rate = df.groupby('arb_step2').curr.mean().abs() / df.groupby('arb_step2').cap.max().max()
        c_rate = 10**np.floor(np.log10(c_rate)) * round(c_rate / (10**np.floor(np.log10(c_rate))), 1)
        c_rate.fillna(0, inplace=True)
        for stp in c_rate.index:
            df.loc[df['arb_step2'] == stp, 'c_rate'] = c_rate.loc[stp]
        df.loc[:, 'c_rate'] = df.loc[:, 'c_rate'].fillna(method='ffill').fillna(0)
    return df


def _col_renamer(col_name):
    neware_native = ['DataPoint',
                     'Cycle Index',
                     'Step Index',
                     'Step Type',
                     'Time',
                     'Total Time',
                     'Current(μA)',
                     'Current(mA)',
                     'Voltage(V)',
                     'Capacity(mAh)',
                     'Spec. Cap.(mAh/g)',
                     'Chg. Cap.(mAh)',
                     'Chg. Spec. Cap.(mAh/g)',
                     'DChg. Cap.(mAh)',
                     'DChg. Spec. Cap.(mAh/g)',
                     'Energy(Wh)',
                     'Spec. Energy(mWh/g)',
                     'Chg. Energy(Wh)',
                     'Chg. Spec. Energy(mWh/g)',
                     'DChg. Energy(Wh)',
                     'DChg. Spec. Energy(mWh/g)',
                     'Date',
                     'Power(W)',
                     'dQ/dV(mAh/V)',
                     'dQm/dV(mAh/V.g)',
                     'Contact resistance(mΩ)',
                     'Module start-stop switch',
                     'V1',
                     'V2',
                     'Aux. ΔV',
                     'T1',
                     'T2',
                     'Aux. ΔT']
    local_names = ['measurement',
                   'arb_step2',
                   'arb_step1',
                   'mode',
                   'rel_time',
                   'total_time',
                   'curr',
                   'curr',
                   'volt',
                   'cap',
                   'spec_cap',
                   'chrg_cap',
                   'chrg_spec_cap',
                   'dchg_cap',
                   'dchg_spec_cap',
                   'egy',
                   'spec_egy',
                   'chrg_egy',
                   'chrg_spec_egy',
                   'dchg_egy',
                   'dchg_spec_egy',
                   'abs_time',
                   'power',
                   'ica',
                   'ica_spec',
                   'contact_resistance',
                   'module_strt_stop',
                   'aux_volt_1',
                   'aux_volt_2',
                   'aux_dv',
                   'aux_T_1',
                   'aux_T_2',
                   'aux_dT']
    rename_dct = dict(zip(neware_native, local_names))
    return rename_dct[col_name]


def _find_curr_unit(col):
    if '(μA)' in col:
        return 'micro'
    elif '(mA)' in col:
        return 'milli'
    elif '(A)' in col:
        return 'SI'


def _find_curr_col(df):
    for c in df.columns:
        if 'Current' in c:
            curr_col = c
    return curr_col


def _scale_current(df, i_unit):
    rdf = df.copy()
    if i_unit == 'micro':
        rdf.curr = rdf.curr / 1e6
    elif i_unit == 'milli':
        rdf.curr = rdf.curr / 1e3
    else:
        pass
    return rdf


def find_initial_data_row(df):
    # Find the row where the time-resolved data starts
    metadata_rows = 0
    for index, row in df.iterrows():
        # Here you can add conditions to identify the end of metadata
        # For example: If a specific column contains NaN values
        if pd.isna(row).any():
            metadata_rows += 1
        else:
            break
    return metadata_rows + 1


def find_aux_channels(xl):
    aux_channels = [sh_name for sh_name in xl.sheet_names if 'aux' in sh_name]
    return aux_channels


def parse_aux_channel(xl, sheet):
    df_tmp = xl.parse(sheet)
    FIRST_DATA = find_initial_data_row(df_tmp)
    aux_df = xl.parse(sheet, header=FIRST_DATA)
    # aux_df = aux_df.loc[:, (aux_df != 0).any(axis=0)]
    return aux_df


def read_neware_80_xls(fname, curr_unit='milli'):
    xl_ = pd.ExcelFile(fname)
    df = xl_.parse('record')
    aux_ch = find_aux_channels(xl_)
    aux_ch_df_dct = {k: parse_aux_channel(xl_, k) for k in aux_ch}
    for k, aux_df in aux_ch_df_dct.items():
        shared_cols = list(set(df.columns) & set(aux_df.columns))
        df = pd.merge(df, aux_df, on=shared_cols)
    c_col = _find_curr_col(df)
    i_unit = _find_curr_unit(c_col)
    df.columns = [_col_renamer(c) for c in df.columns]
    df = _scale_current(df, i_unit)
    df['step_time'] = pd.to_timedelta(df.rel_time).astype('int64') / 1e9
    df['abs_time'] = pd.to_datetime(df['abs_time'], format='%Y-%m-%d %H:%M:%S')
    df['float_time'] = pd.to_timedelta(df['total_time']).astype('int64') / 1e9
    df.loc[:, 'step_time_float'] = pd.to_timedelta(df.step_time).astype('timedelta64[ms]')
    df.loc[:, 'mAh'] = cumulative_trapezoid(df.curr, df.float_time / 3.6, initial=0)
    return df


if __name__ == '__main__':
    from check_current_os import get_base_path_batt_lab_data
    import os
    # my_files = [r"C:\Users\krifren\TestData\HalfCellData\AbVolvoData\240093-1-1-2818574078.xls",
    #             r"C:\Users\krifren\TestData\HalfCellData\AbVolvoData\240093-1-2-2818574077.xls",
    #             r"C:\Users\krifren\TestData\HalfCellData\AbVolvoData\240093-1-3-2818574078.xls"]
    # dfs = {
    #     'tesla_pos': read_neware_xls(my_files[0], calc_c_rate=True),
    #     'tesla_neg': read_neware_xls(my_files[1]),
    #     'green_neg': read_neware_xls(my_files[2])
    # }
    BASE_PATH_BATTLABDATA = get_base_path_batt_lab_data()
    test_file_v80 = os.path.join(BASE_PATH_BATTLABDATA, 'pulse_chrg_test/cycling_data_ici/240095-3-1-2818575237.xlsx')
    df = read_neware_80_xls(test_file_v80)
