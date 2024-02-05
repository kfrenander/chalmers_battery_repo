import pandas as pd
import re
from backend_fix import fix_mpl_backend
fix_mpl_backend()


def find_data_init_row(file_string):
    with open(file_string) as readfile:
        for cnt, line in enumerate(readfile):
            if 'END RESULTS CHECK' in line:
                init_line = cnt
            else:
                readfile.close()
                break
    try:
        return init_line + 1
    except NameError:
        print(f'No properly structured data found for file {file_string}')
        return None
    except:
        print(f'Unknown generic error for file {file_string}')
        return None


def read_pec_csv(pec_file, num_of_cells=1, head_rows=13):
    test = pd.read_csv(pec_file, sep=',', skiprows=find_data_init_row(pec_file)).dropna(how='all', axis=1)
    col_names = [
        'test_idx', 'step', 'cyc', 'tot_time', 'step_time', 'cyc_chrg_time', 'cyc_dchg_time', 'abs_time',
        'volt', 'curr', 'chrg_cap', 'dchg_cap', 'chrg_egy', 'dchg_egy', 'reason_code', 'loop_var', 'T_amb', 'T_cell'
    ]
    df_dict = {}
    if num_of_cells > 1:
        test_indicators = ['.{}'.format(i) for i in range(1, num_of_cells)]
        for ind in test_indicators:
            test_num = int(re.findall(r'\d', ind)[0]) + 1
            cols = [x for x in test.columns if ind in x]
            rename_dict = dict(zip(cols, col_names))
            df_dict['Cell_{0}'.format(test_num)] = test[cols].dropna(how='all').rename(columns=rename_dict)
            test.drop(columns=cols, inplace=True)
        cols = [x for x in test.columns]
        df_dict['Cell_1'] = test.dropna(how='all').rename(columns=dict(zip(cols, col_names)))
    else:
        cols = [x for x in test.columns]
        df_dict['Cell_1'] = test.dropna(how='all').rename(columns=dict(zip(cols, col_names)))
    return df_dict


if __name__ == '__main__':
    test_file_multiple_cells = r'E:\\PEC_Data\\Temp_Rise_BDA_8seconds_3cells.csv'
    nbr_of_cells = 3
    dict1 = read_pec_csv(test_file_multiple_cells, nbr_of_cells, 37)
    test_file_one_cell = r"E:\PEC_Data\TeslaCellBdaProfileTemperature.csv"
    dict2 = read_pec_csv(test_file_one_cell, 3, 37)
    print(list(dict2))
    test_dir = r'Z:\Provning\PEC\BDA_tryout'
    import os
    dict3 = {}
    for f_ in os.listdir(test_dir):
        if f_.endswith('.csv'):
            name = re.search(r'\d+sec', f_).group()
            dict3[name] = read_pec_csv(os.path.join(test_dir, f_), 1, 35)
