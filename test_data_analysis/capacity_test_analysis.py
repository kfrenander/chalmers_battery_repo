from PythonScripts.test_data_analysis.read_neware_file import read_neware_xls
import pandas as pd
from PythonScripts.test_data_analysis.ica_analysis import make_ica_dva_plots
from PythonScripts.test_data_analysis.basic_plotting import volt_curr_plot, cap_v_volt_multicolor
import os


def find_cap_meas(df: pd.DataFrame, cell: dict):
    i = 0
    cap_dict = {}
    op = pd.DataFrame(columns=['maxV', 'minV', 'cap', 'c_rate'])
    for stp in df.arb_step2.unique():
        partial_df = df[df.arb_step2 == stp]
        if partial_df.empty:
            continue
        avg_curr = partial_df.curr.mean()
        max_volt = partial_df.volt.max()
        min_volt = partial_df.volt.min()
        prev_volt = df[df.step == stp - 1].volt.mean()
        prev_mode = df[df.step == stp - 1].mode
        duration = partial_df.float_time.max() - partial_df.float_time.min()
        # print('For step {4}: \n curr is {0}, Umax is {1}, Umin is {2}, previous U is {5} and duration is {3}'.format(
        #     round(avg_curr, 2), round(max_volt, 2), round(min_volt, 2), round(duration), stp, round(prev_volt, 2)
        # ))
        data_pts = {
            'maxV': round(max_volt, 3),
            'minV': round(min_volt, 3),
            'cap': partial_df.cap.abs().max() / 1000,
            'c_rate': round(avg_curr * 1000 / df.cap.abs().max(), 2)
        }
        if abs(avg_curr - cell['C_rate']) < 0.05 and \
                abs(max_volt - cell['Umax'] < 0.02) and \
                abs(min_volt - cell['Umin'] < 0.1) and \
                prev_volt > 4.1 and \
                duration > 8500:
            i += 1
            print('Cap measurement {0} is step {1} with capacity {2}'.format(i, stp, data_pts['cap']))
            cap_dict['cap_meas_{0}'.format(i)] = partial_df
            op = op.append(pd.DataFrame(data=data_pts, index=['cap_meas_{0}'.format(i)]))
    op.loc['mean'] = op.mean()
    return cap_dict, op


if __name__ == '__main__':
    cell = {
        'Umax': 4.18,
        'Umin': 2.55,
        'C_rate': -1.53
    }
    cap_data_dir = r"E:\Neware Data\RPT_and_HPPC\Calendar_RPT"
    # cap_data_file = r"Z:\Provning\Neware\Capacity_test_two_repeats_127.0.0.1_240119-2-7-92.xls"
    # cap_data_file = r"Z:\Provning\Neware\Capacity_Test\capacity_test_x5_Cell41.xls"
    cap_data_file = r"E:\Neware Data\RPT_and_HPPC\Initial_RPT\127.0.0.1_240119-6-1-143.xls"
    df = read_neware_xls(cap_data_file)
    data_dict, op_df = find_cap_meas(df, cell)
    df_dict = {}
    op_dict = {}

    for root, dir, name in os.walk(cap_data_dir):
        for file in name:
            cell_id = file.split("_")[-1].split(".x")[0]
            df = read_neware_xls(os.path.join(root, file))
            df_dict[cell_id] = df
            # fig0 = volt_curr_plot(df)
            fig1 = cap_v_volt_multicolor(df, name=cell_id)
            # fig1.savefig(r"Z:\Provning\Neware\cap_test_analysis\cap_v_volt_{0}.png".format(cell_id), dpi=600)
            data_dict, op_df = find_cap_meas(df, cell)
            op_df['rel_cap'] = op_df.cap / op_df.cap[0]
            op_df.round(3)
            op_dict[cell_id] = op_df
            # op_df.to_excel(r"Z:\Provning\Neware\cap_test_analysis\{0}_rel_cap.xlsx".format(cell_id))
    # # fig0.savefig(r'Z:\Provning\Neware\cap_test_analysis\volt_and_curr.png', dpi=600)
    #
    # ica, dva, ica_dva_df = make_ica_dva_plots(df, name='Cby3')

