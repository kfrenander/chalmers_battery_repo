import pandas as pd
import numpy as np
from scipy.integrate import cumtrapz


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
    df['egy_tot'] = cumtrapz(df.pwr.abs() / (1000 * 3600), df.float_time, initial=0)
    df['egy_chrg'] = cumtrapz(df.pwr_chrg.abs() / (1000 * 3600), df.float_time, initial=0)
    df['egy_dchg'] = cumtrapz(df.pwr_dchg.abs() / (1000 * 3600), df.float_time, initial=0)
    if mA:
        df['mAh'] = cumtrapz(df.curr, df.float_time, initial=0) / 3600
        df['curr'] = df.curr / 1000
    else:
        df['mAh'] = cumtrapz(df.curr, df.float_time, initial=0) * 1000 / 3600
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


if __name__ == '__main__':
    my_files = [r"C:\Users\krifren\TestData\HalfCellData\AbVolvoData\240093-1-1-2818574078.xls",
                r"C:\Users\krifren\TestData\HalfCellData\AbVolvoData\240093-1-2-2818574077.xls",
                r"C:\Users\krifren\TestData\HalfCellData\AbVolvoData\240093-1-3-2818574078.xls"]
    dfs = {
        'tesla_pos': read_neware_xls(my_files[0], calc_c_rate=True),
        'tesla_neg': read_neware_xls(my_files[1]),
        'green_neg': read_neware_xls(my_files[2])
    }
