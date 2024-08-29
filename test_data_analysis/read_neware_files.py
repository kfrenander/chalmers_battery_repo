import pandas as pd
import numpy as np
import sys
from scipy.integrate import cumtrapz
from test_data_analysis.neware_column_mapper import define_neware_renaming
import warnings


class NewareDataReader:
    def __init__(self, file_paths, calc_c_rate=False):
        self.file_paths = file_paths
        self.calc_c_rate = calc_c_rate
        self.xl_files = []
        self.ver_id = None
        self.col_rename_dict = None

        # Load the Excel files and identify the version
        self._load_files()
        self.ver_id = self._id_neware_version(self.xl_files[0])
        self.col_rename_dict = define_neware_renaming(self.ver_id)

    def _load_files(self):
        if isinstance(self.file_paths, str):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.xl_files = [pd.ExcelFile(self.file_paths, engine='openpyxl')]
        elif isinstance(self.file_paths, pd.ExcelFile):
            self.xl_files = [self.file_paths]
        elif isinstance(self.file_paths, list):
            if all([isinstance(fpath, pd.ExcelFile) for fpath in self.file_paths]):
                print('Function called with list of ExcelFile objects, no need to re-read.')
                self.xl_files = self.file_paths
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.xl_files = [pd.ExcelFile(fpath) for fpath in self.file_paths]

    def _id_neware_version(self, xl_file):
        v_80_indicator = 'record'
        v_76_indicator = 'Info'
        if v_80_indicator in xl_file.sheet_names:
            return 'v80'
        elif v_76_indicator in xl_file.sheet_names:
            return 'v76'
        else:
            print('Unknown Neware format, please check file/-s')
            sys.exit()

    def read_dynamic_data(self):
        if self.ver_id == 'v80':
            df_lst = [self._read_neware_v80_xls(xl_file) for xl_file in self.xl_files]
        elif self.ver_id == 'v76':
            df_lst = [self._read_neware_v76_xls(xl_file) for xl_file in self.xl_files]
        else:
            print('Unknown version format. Exiting...')
            return pd.DataFrame()

        df = pd.concat(df_lst, ignore_index=True)
        return self._calculate_cumulative_values(df)

    def read_statistics(self):
        if self.ver_id == 'v80':
            stat_df = self.xl_files[0].parse('step')
        elif self.ver_id == 'v76':
            stat_sheet = [sh_name for sh_name in self.xl_files[0].sheet_names if 'Statis_' in sh_name]
            stat_df = self.xl_files[0].parse(stat_sheet[0])
        stat_df = self._rename_df_columns(stat_df)
        stat_df['step_duration'] = pd.to_timedelta(stat_df['step_duration'])
        stat_df['step_duration_float'] = stat_df['step_duration'].apply(lambda x: x.total_seconds())
        return stat_df

    def read_cycle(self):
        if self.ver_id == 'v80':
            cycle_df = self.xl_files[0].parse('cycle')
        elif self.ver_id == 'v76':
            cycle_sheet = [sh_name for sh_name in self.xl_files[0].sheet_names if 'Cycle' in sh_name]
            cycle_df = self.xl_files[0].parse(cycle_sheet[0])
        cycle_df = self._rename_df_columns(cycle_df)
        return cycle_df

    def _read_neware_v80_xls(self, xl_file, curr_unit='milli'):
        df = xl_file.parse('record')
        aux_ch = self._find_aux_channels(xl_file)
        if aux_ch:
            aux_ch_df_dct = {k: self._parse_aux_channel(xl_file, k) for k in aux_ch}
            for k, aux_df in aux_ch_df_dct.items():
                shared_cols = list(set(df.columns) & set(aux_df.columns))
                df = pd.merge(df, aux_df, on=shared_cols)
        df['arb_step2'] = (df['Step Index'].diff() != 0).cumsum()
        df = self._process_dataframe(df)
        df = self._update_dataframe_times(df)
        df['float_time'] = pd.to_timedelta(df['total_time']).astype('int64') / 1e9
        return df

    def _read_neware_v76_xls(self, xl_file, curr_unit='milli'):
        main_sheet = [sh_name for sh_name in xl_file.sheet_names if 'Detail_' in sh_name]
        aux_sheets = [sh_name for sh_name in xl_file.sheet_names if 'DetailTemp' in sh_name or 'DetailVol' in sh_name]

        for sheet in main_sheet:
            df = xl_file.parse(sheet)

        df = self._process_dataframe(df)

        if aux_sheets:
            aux_df_lst = [xl_file.parse(sheet) for sheet in aux_sheets]
            for aux_df in aux_df_lst:
                aux_df = self._rename_df_columns(aux_df)
                shared_cols = list(set(df.columns) & set(aux_df.columns))
                df = pd.merge(df, aux_df, on=shared_cols)
        df = self._update_dataframe_times(df)

        if self.calc_c_rate:
            self._calculate_c_rate(df)

        return df

    def _process_dataframe(self, df):
        c_col = self._find_curr_col(df)
        i_unit = self._find_curr_unit(c_col)
        df = self._rename_df_columns(df)
        df = self._scale_current(df, i_unit)
        return df

    def _update_dataframe_times(self, df):
        df['step_time'] = pd.to_timedelta(df.rel_time).astype('timedelta64[ms]')
        df['abs_time'] = pd.to_datetime(df['abs_time'], format='%Y-%m-%d %H:%M:%S')
        df['float_step_time'] = df.step_time.apply(lambda x: x.total_seconds())
        posix_time = df['abs_time'].apply(lambda x: x.timestamp())
        df['float_time'] = posix_time - posix_time[0]
        return df

    def _find_curr_unit(self, col):
        if '(μA)' in col:
            return 'micro'
        elif '(mA)' in col:
            return 'milli'
        elif '(A)' in col:
            return 'SI'

    def _find_curr_col(self, df):
        for c in df.columns:
            if 'Current' in c:
                return c

    def _scale_current(self, df, i_unit):
        rdf = df.copy()
        if i_unit == 'micro':
            rdf.curr = rdf.curr / 1e6
        elif i_unit == 'milli':
            rdf.curr = rdf.curr / 1e3
        return rdf

    def _find_aux_channels(self, xl):
        return [sh_name for sh_name in xl.sheet_names if 'aux' in sh_name]

    def _parse_aux_channel(self, xl, sheet):
        df_tmp = xl.parse(sheet)
        first_data = self._find_initial_data_row(df_tmp)
        aux_df = xl.parse(sheet, header=first_data)
        return aux_df

    def _find_initial_data_row(self, df):
        metadata_rows = 0
        for index, row in df.iterrows():
            if pd.isna(row).any():
                metadata_rows += 1
            else:
                break
        return metadata_rows + 1

    def _rename_df_columns(self, df):
        rdf = df.copy()
        rdf.columns = [self.col_rename_dict[c] for c in rdf.columns]
        return rdf

    def _calculate_c_rate(self, df):
        i_unit = self._find_curr_unit(self._find_curr_col(df))
        if i_unit == 'milli':
            c_rate = 1000 * df.groupby('arb_step2').curr.mean().abs() / df.groupby('arb_step2').cap.max().max()
        else:
            c_rate = df.groupby('arb_step2').curr.mean().abs() / df.groupby('arb_step2').cap.max().max()

        c_rate = 10 ** np.floor(np.log10(c_rate)) * round(c_rate / (10 ** np.floor(np.log10(c_rate))), 1)
        c_rate.fillna(0, inplace=True)

        for stp in c_rate.index:
            df.loc[df['arb_step2'] == stp, 'c_rate'] = c_rate.loc[stp]

        df['c_rate'] = df['c_rate'].fillna(method='ffill').fillna(0)

    def _calculate_cumulative_values(self, df):
        posix_time = df['abs_time'].apply(lambda x: x.timestamp())
        df['float_time'] = posix_time - posix_time[0]
        df['mAh'] = cumtrapz(df.curr, df.float_time / 3.6, initial=0)
        df['pwr'] = df['volt'] * df['curr']
        df['pwr_chrg'] = df.pwr.mask(df.pwr < 0, 0)
        df['pwr_dchg'] = df.pwr.mask(df.pwr > 0, 0)
        df['egy_tot'] = cumtrapz(df.pwr.abs() / (1000 * 3600), df.float_time, initial=0)
        df['egy_chrg'] = cumtrapz(df.pwr_chrg.abs() / (1000 * 3600), df.float_time, initial=0)
        df['egy_dchg'] = cumtrapz(df.pwr_dchg.abs() / (1000 * 3600), df.float_time, initial=0)
        return df


if __name__ == '__main__':
    from check_current_os import get_base_path_batt_lab_data
    import time
    import os
    import datetime as dt

    # dfs = {
    #     'tesla_pos': read_neware_xls(my_files[0], calc_c_rate=True),
    #     'tesla_neg': read_neware_xls(my_files[1]),
    #     'green_neg': read_neware_xls(my_files[2])
    # }
    print(f'Data read in started at {dt.datetime.now():%Y-%m-%d__%H:%M.%S}.')
    BASE_PATH_BATTLABDATA = get_base_path_batt_lab_data()
    test_file_v80 = os.path.join(BASE_PATH_BATTLABDATA,
                                 'pulse_chrg_test/cycling_data_ici/debug_case/240095-3-1-2818575237-debug_mwe.xlsx')
    test_file_v76 = os.path.join(BASE_PATH_BATTLABDATA, "pulse_chrg_test/cycling_data/240095-3-7-2818575230.xlsx")
    test_files_combine = [
        "pulse_chrg_test/cycling_data/240095-3-1-2818575227.xlsx",
        "pulse_chrg_test/cycling_data/240095-3-1-2818575227_1..xlsx",
        "pulse_chrg_test/cycling_data/240095-3-1-2818575227_2..xlsx"
    ]
    test_files_combine = [os.path.join(BASE_PATH_BATTLABDATA, fname) for fname in test_files_combine]
    reader = NewareDataReader(test_file_v76)
    df_76 = reader.read_dynamic_data()
    stat_df = reader.read_statistics()
    cycle_df = reader.read_cycle()
    print(df_76.tail())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xl_reader = NewareDataReader(pd.ExcelFile(test_file_v76, engine='openpyxl'))
        df_76_fromxl = xl_reader.read_dynamic_data()
        print(df_76_fromxl.tail())
    tic = time.time()
    combine_reader = NewareDataReader(test_files_combine)
    df_comb = combine_reader.read_dynamic_data()
    toc = time.time()
    print(df_comb.tail())
    print(f'Time elapsed to read combined df is {toc - tic:.2f} seconds')
    tic = time.time()
    v80_reader = NewareDataReader(test_file_v80)
    df_80 = v80_reader.read_dynamic_data()
    stat_df_80 = v80_reader.read_statistics()
    cycle_df_80 = v80_reader.read_cycle()
    toc = time.time()
    print(df_80.tail())
    print(f'Time elapsed to read {test_file_v80} df is {toc - tic:.2f} seconds')