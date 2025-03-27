from test_data_analysis.read_neware_file import read_neware_xls
from test_data_analysis.capacity_test_analysis import find_cap_meas
from test_data_analysis.ica_analysis import find_ica_step, calc_ica_dva, large_span_ica_dva, simplified_ica_dva
from test_data_analysis.basic_plotting import dva_plot, ica_plot, cap_v_volt_multicolor
from scipy.signal import savgol_filter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import datetime as dt
from scipy.interpolate import interp1d


def find_step_characteristics(df):
    step_dict = {}
    for stp in df.arb_step2.unique():
        sub_df = df[df.arb_step2 == stp]
        avg_curr = sub_df.curr.mean()
        max_volt = sub_df.volt.max()
        min_volt = sub_df.volt.min()
        egy_tot = sub_df.egy_tot.max()
        egy_dchg = sub_df.egy_dchg.max()
        egy_chrg = sub_df.egy_chrg.max()
        dur = sub_df.float_time.max() - sub_df.float_time.min()
        step_dur = sub_df.step_time.max().total_seconds()
        stp_cap = sub_df.cap.abs().max()
        stp_mode = sub_df['mode'].mode().values[0]
        stp_date = sub_df['abs_time'].iloc[0]
        data_pts = {
            'stp_date': stp_date,
            'maxV': round(max_volt, 3),
            'minV': round(min_volt, 3),
            'cap': stp_cap,
            'curr': avg_curr,
            'duration': dur,
            'step_duration': step_dur,
            'step_nbr': stp,
            'step_mode': stp_mode,
            'egy_thrg': egy_tot,
            'egy_chrg': egy_chrg,
            'egy_dchg': egy_dchg
        }
        step_dict['{0}'.format(stp)] = data_pts
    df_out = pd.DataFrame(step_dict).T
    return df_out


def characterise_steps(df, step_counter='arb_step2', mode_indicator='mode'):
    gb = df.groupby(step_counter)
    attr = {k: [gb.get_group(k)['abs_time'].iloc[0],
                gb.get_group(k).volt.max(),
                gb.get_group(k).volt.min(),
                gb.get_group(k).curr.mean(),
                gb.get_group(k).egy_tot.max(),
                gb.get_group(k).egy_dchg.max(),
                gb.get_group(k).egy_chrg.max(),
                gb.get_group(k).cap.abs().max(),
                gb.get_group(k).step_time.max().total_seconds(),
                gb.get_group(k)[mode_indicator].mode().values[0],
                k]
            for k in gb.groups}
    df_out = pd.DataFrame.from_dict(attr, orient='index',
                                    columns=[
                                        'stp_date',
                                        'maxV',
                                        'minV',
                                        'curr',
                                        'egy_thrg',
                                        'egy_dchg',
                                        'egy_chrg',
                                        'cap',
                                        'step_duration',
                                        'step_mode',
                                        'step_nbr']
                                    )
    return df_out


def characterise_steps_agg(df, step_counter='arb_step2', mode_indicator='mode', orig_step_indicator='orig_step'):
    # Use groupby with aggregate functions
    df_out = df.groupby(step_counter).agg(
        stp_date=('abs_time', 'first'),
        maxV=('volt', 'max'),
        minV=('volt', 'min'),
        curr=('curr', 'mean'),
        egy_thrg=('egy_tot', 'max'),
        egy_dchg=('egy_dchg', 'max'),
        egy_chrg=('egy_chrg', 'max'),
        orig_step=(orig_step_indicator, 'first'),
        cap=('cap', lambda x: x.abs().max()),
        step_duration=('step_time', lambda x: x.max().total_seconds()),
        step_mode=(mode_indicator, lambda x: x.mode()[0]),
    ).reset_index()

    # Rename the step_counter column
    df_out.rename(columns={step_counter: 'step_nbr'}, inplace=True)

    return df_out.set_index('step_nbr', drop=False)


def characterise_cdaq_log(df, step_counter='step_nbr', mode_indicator='current_mode'):
    # Use groupby with aggregate functions
    df_out = df.groupby(step_counter).agg(
        maxV=('volt', 'max'),
        minV=('volt', 'min'),
        curr=('curr', 'median'),
        cap=('mAh', lambda x: (x - x.iloc[0]).abs().max()),
        step_duration=('float_step_time', lambda x: x.max()),
        step_mode=(mode_indicator, lambda x: x.mode().iloc[0]),
    ).reset_index()

    # Rename the step_counter column
    df_out.rename(columns={step_counter: 'step_nbr'}, inplace=True)

    return df_out.set_index('step_nbr', drop=False)


def extract_ica_data(df, step_list):
    ica_df = pd.DataFrame()
    for stp in step_list:
        if ica_df.empty:
            ica_df = df[df.step == stp]
        else:
            ica_df = ica_df.append(df[df.step == stp])
    return ica_df


def ica_filtering(df):
    mean_ica_volt_diff = (df.volt.diff() * np.sign(df.curr)).mean()
    for i in range(2):
        # For unknown reason filtering must be repeated twice to function fully
        ica_filter = df.volt.diff() * np.sign(df.curr) > mean_ica_volt_diff
        df = df[ica_filter]
    return df


def find_cell_name(file_name):
    chan_name = '{0}_{1}'.format(*[x.strip("-") for x in re.findall(r'-\d+', file_name) if len(x) < 3])
    # cell_inventory_file = r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\Aline_BAD\Cell_Inventory\Tesla2170CellsFromVCC201909_Updated_2020_02_24.xlsx"
    cell_inventory_file = r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\Aline_BAD\Cell_Inventory\Tesla2170CellsFromVCC201909_Updated_23_12_2019.xlsx"
    xl_file = pd.ExcelFile(cell_inventory_file)
    cell_inv = xl_file.parse(sheet_name='Sheet1')
    cell_inv['test_nbr'] = cell_inv['Test code'].dropna().map(lambda x: x.lstrip('#'))
    # test_mat = xl_file.parse(sheet_name='TestMatrix')
    return cell_inv[cell_inv['Channel'] == chan_name]['Bar Code Number'].values[0]


def find_nearest(arr, val):
    arr = np.asarray(arr)
    idx = (np.abs(arr - val)).argmin()
    return arr[idx]


def find_res(df: pd.DataFrame, char_df: pd.DataFrame):
    pulse_steps = char_df[(char_df.duration < 11) & (char_df.curr.abs() > 5)].step_nbr
    soc_arr = np.array([0, 10, 30, 50, 70, 90, 100])
    soc_lookup = interp1d([2.5, 3.1, 3.5, 3.7, 3.9, 4.1, 4.2], soc_arr)
    R0_dchg = {}
    R0_chrg = {}
    R10_dchg = {}
    R10_chrg = {}
    for stp in pulse_steps:
        ocv = df.loc[df[df.arb_step2 == stp - 1].last_valid_index(), 'volt']
        # print('Found ocv of {:.2f}'.format(ocv))
        lookup_soc = soc_lookup(ocv)
        round_soc = find_nearest(soc_arr, lookup_soc)
        # print('Found rounded soc of {} and unrounded soc of {:.2f}'.format(round_soc, lookup_soc))
        step_label = f'soc_{round(round_soc)}'
        curr = df[df.arb_step2 == stp].curr.mean()
        start_index_pulse = df[df.arb_step2 == stp].first_valid_index()
        fin_index_pulse = df[df.arb_step2 == stp].last_valid_index()
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


def rpt_analysis(xl_file: str, cell: dict, case: str = ''):
    """
    :param xl_file:     File path containing RPT measurement on standardised format
    :param cell:        Dictionary containing the voltage boundaries and expected c-rate.
    :param case:        String sepifying if specific case is consiered
    :return:
    """
    # ref = os.path.split(xl_file)[-1].split('.xl')[0]
    cell_name = find_cell_name(xl_file)
    op_dir = r'Z:\Provning\Neware\RPT_and_HPPC\rpt_analysis_{0}_{1}'.format(case, dt.datetime.now().__format__('%Y%m%d'))
    if not os.path.isdir(op_dir):
        os.makedirs(op_dir)
        os.makedirs(os.path.join(op_dir, 'dva_figs'))
        os.makedirs(os.path.join(op_dir, 'ica_figs'))
        os.makedirs(os.path.join(op_dir, 'hyst_figs'))
    df = read_neware_xls(xl_file)
    test_date = df.abs_time[0].date().strftime('%Y-%m-%d')
    ref = '{0}_rpt_{1}'.format(cell_name, test_date)
    char_df = find_step_characteristics(df)
    cap_dict, cap_df = find_cap_meas(df, cell)
    ica_step_list = find_ica_step(char_df, cell)
    ica_df = df[df.step.isin(ica_step_list)]
    if ica_df.float_time.diff().mean() < 10:
        ica_df = ica_filtering(ica_df)
        ica_df = large_span_ica_dva(ica_df)
        ica_df['ica_filt'] = savgol_filter(ica_df.ica, 69, 0)
        ica_df['dva_filt'] = savgol_filter(ica_df.dva, 71, 1)
    else:
        ica_df = simplified_ica_dva(ica_df)
        ica_df['ica_filt'] = savgol_filter(ica_df.ica, 13, 0)
        ica_df['dva_filt'] = savgol_filter(ica_df.dva, 9, 1)
    hyst_fig = cap_v_volt_multicolor(ica_df, name=cell_name)
    ica_fig = ica_plot(ica_df)
    ica_fig.get_axes()[0].set_xlim(2.9, 4.2)
    ica_fig.get_axes()[0].set_ylim(-15, 15)
    ica_fig.tight_layout()
    dva_fig = dva_plot(ica_df)
    dva_fig.get_axes()[0].set_ylim(-0.7, 0.7)
    dva_fig.tight_layout()
    ica_fig.savefig(os.path.join(op_dir, 'ica_figs', ref + 'ICA_test.png'), dpi=400)
    dva_fig.savefig(os.path.join(op_dir, 'dva_figs', ref + 'DVA_test.png'), dpi=400)
    hyst_fig.savefig(os.path.join(op_dir, 'hyst_figs', ref + 'ICA_hysteresis.png'), dpi=400)
    ica_dict = {
        '{0}_fig_ica'.format(ref): ica_fig,
        '{0}_fig_dva'.format(ref): dva_fig,
        '{0}_df'.format(ref): ica_df
    }
    plt.close('all')
    res_df = find_res(df, char_df)
    avg_cap = cap_df.cap.mean()
    ref_res = res_df.loc['soc_50', 'R10_dchg']
    summary_df = pd.DataFrame([[avg_cap, ref_res, test_date]], index=[cell_name],
                              columns=['capacity', 'resistance', 'date'])
    return summary_df, ica_dict


if __name__ == '__main__':
    test_dir = r"E:\Neware Data\RPT_and_HPPC\Initial_RPT"
    test_files = [r"Z:\Provning\Neware\RPT_and_HPPC\RPT_127.0.0.1_240119-4-5-128.xls",
                 r"Z:\Provning\Neware\RPT_and_HPPC\RPT_127.0.0.1_240119-4-6-129.xls"]
    tesla_cell = {
        'Umax': 4.18,
        'Umin': 2.55,
        'C_rate': -1.53
    }
    df_list = []
    op_df = pd.DataFrame()
    dir_files = os.listdir(test_dir)
    ica_dict = {}
    test_case = 'plt_up'
    for file in dir_files:
        # if '19-6-1' in file:
        #     print(file)
        #     df = read_neware_xls(os.path.join(test_dir, file))
        #     cap_v_volt_multicolor(df, name=find_cell_name(file))
        print('Test file {}'.format(file))
        df, dict = rpt_analysis(os.path.join(test_dir, file), tesla_cell, case=test_case)
        op_df = op_df.append(df)
        df_list.append(df)
        # ica_dict.update(dict)
    op_df.to_excel(
        os.path.join(r'Z:\Provning\Neware\RPT_and_HPPC\rpt_analysis_{0}_{1}'.format(test_case, dt.datetime.now().__format__('%Y%m%d')), 'rpt_summary.xlsx')
    )
