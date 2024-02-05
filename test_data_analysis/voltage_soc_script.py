from test_data_analysis.rpt_analysis import find_cell_name
from test_data_analysis.read_neware_file import read_neware_xls
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('chalmers_KF')

my_files_current = [r"Z:\Provning\Neware\voltage_v_cap\Voltage_1C_and_Cby3_potentiostatic_current_127.0.0.1_240119-4-4-132.xls",
            r"Z:\Provning\Neware\voltage_v_cap\Voltage_1C_and_Cby3_potentiostatic_current_127.0.0.1_240119-4-1-132.xls",
            r"Z:\Provning\Neware\voltage_v_cap\Voltage_1C_and_Cby3_potentiostatic_current_127.0.0.1_240119-4-2-132.xls",
            r"Z:\Provning\Neware\voltage_v_cap\Voltage_1C_and_Cby3_potentiostatic_current_127.0.0.1_240119-4-3-132.xls"]
my_files_time = [r"Z:\Provning\Neware\voltage_v_cap\Voltage_1C_and_Cby3_potentiostatic_time_127.0.0.1_240119-4-7-131.xls",
                 r"Z:\Provning\Neware\voltage_v_cap\Voltage_1C_and_Cby3_potentiostatic_time_127.0.0.1_240119-4-8-131.xls",
                 r"Z:\Provning\Neware\voltage_v_cap\Voltage_1C_and_Cby3_potentiostatic_time_127.0.0.1_240119-4-5-132.xls",
                 r"Z:\Provning\Neware\voltage_v_cap\Voltage_1C_and_Cby3_potentiostatic_time_127.0.0.1_240119-4-6-132.xls"]
all_files = [my_files_current, my_files_time]

df_dict = {}
df_curr = {}
df_time = {}
for file in my_files_current:
    cell_name = find_cell_name(file)
    df_curr[cell_name] = read_neware_xls(file)

for file in my_files_time:
    cell_name = find_cell_name(file)
    df_time[cell_name] = read_neware_xls(file)

for my_files in all_files:
    for file in my_files:
        print('Trying for cell {0}'.format(file))
        cell_name = find_cell_name(file)
        df_dict[cell_name] = read_neware_xls(file)

soc_arr = np.arange(0.05, 1, 0.1)
plt.figure()
volt_dict = {}
volt_df = pd.DataFrame()
for key in df_dict:
    df = df_dict[key]
    step_list = []
    for stp in df.step.unique():
        avg_curr = df[df.step == stp].curr.mean()
        max_v = df[df.step == stp].volt.max()
        print(avg_curr, max_v)
        if abs(avg_curr + 4.6) < 0.1 and abs(max_v - 4.01) < 0.03:
            step_list.append(stp)
    if any(i < 5 for i in step_list):
        step_list.pop(0)
    i = 1
    # df_dict[key]['SOC'] = df_dict[key].cap / df_dict[key]
    sub_df = df_dict[key][df_dict[key].step.isin(step_list)]
    sub_df['soc'] = sub_df.cap / sub_df.cap.abs().max()
    v_list = {}
    for soc in soc_arr:
        v_temp = (sub_df[abs(sub_df.soc - soc) < 0.005].volt.mean())
        v_list['{0}'.format(round((1 - soc)*100))] = (v_temp)
        volt_dict['{0}_soc_{1}'.format(key, round(soc*100))] = v_temp
    temp_df = pd.DataFrame(v_list, index=[key])
    volt_df = volt_df.append(temp_df)
    for stp in sub_df.step.unique():
        label = '{0}_step{1}'.format(key, i)
        plt.plot(sub_df[sub_df.step == stp].cap / sub_df[sub_df.step == stp].cap.abs().max(),
                 sub_df[sub_df.step == stp].volt, label=label)
        i += 1
plt.xlabel('SOC [-]', fontsize=13)
plt.ylabel('Voltage [V]', fontsize=13)
plt.title('All cells with time')
plt.legend()
plt.tight_layout()

for xc in soc_arr:
    plt.axvline(xc)

fig, ax = plt.subplots(4, 1)
i = 0

for key in df_curr:
    df = df_curr[key]
    step_list = []
    for stp in df.step.unique():
        avg_curr = df[df.step == stp].curr.mean()
        max_v = df[df.step == stp].volt.max()
        if abs(avg_curr + 4.6) < 0.1 and abs(max_v - 4.01) < 0.03:
            step_list.append(stp)
    if any(i < 5 for i in step_list):
        step_list.pop(0)
    sub_df = df_dict[key][df_dict[key].step.isin(step_list)]
    ax[i].plot(sub_df.cap / sub_df.cap.abs().max(), sub_df.volt, label=key)
    ax[i].legend()
    i += 1
