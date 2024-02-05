import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PythonScripts.test_data_analysis.read_neware_file import read_neware_xls
from scipy import signal
plt.style.use('seaborn-bright')


def check_ici_step(df, step):
    stp_df = df[df.step == step]
    mean_curr = stp_df.curr.mean()
    first_curr = stp_df.loc[stp_df.first_valid_index(), 'curr']
    last_curr = stp_df.loc[stp_df.last_valid_index(), 'curr']
    duration = stp_df.float_time.max() - stp_df.float_time.min()
    if abs((first_curr + last_curr)/2 - mean_curr) < 1e-2 and abs(mean_curr) > 0 and abs(duration - 300) < 10:
        return True
    else:
        return False


def calc_step_res(df):
    dchg_curr = 1
    for st in df.step.unique():
        if st > 1:
            beg_ind = df[df.step == st].first_valid_index()
            stp_ind = df[df.step == st].last_valid_index()
            stp_curr = df[df.step == st]['curr'].mean()
            if stp_curr != 0:
                dchg_curr = stp_curr
                fin_volt = df.loc[stp_ind, 'volt']
            elif stp_curr == 0 and check_ici_step(df, int(st - 1)):
                df.loc[beg_ind, 'R0'] = 1e3 * (fin_volt - df.loc[beg_ind, 'volt']) / dchg_curr
                df.loc[stp_ind, 'R10'] = 1e3 * (fin_volt - df.loc[stp_ind, 'volt']) / dchg_curr
    return df


my_file = r"Z:\Provning\Neware\ICI_test_127.0.0.1_240119-2-8-100.xls"
df = read_neware_xls(my_file)

df = calc_step_res(df)

fig, ax1 = plt.subplots(1, 1)
ax1.plot(df.float_time, df.volt, label='voltage')
ax2 = ax1.twinx()
ax2.grid(False)
# ax2.set_ylim([0, 2e-3])
ax2.scatter(df.float_time, df.R0, marker='x', color='r', label='R0')
ax2.scatter(df.float_time, df.R10, marker='p', color='k', label='R10')

fig2, ax3 = plt.subplots(1, 1)
xaxis = (df.mAh - df.mAh.min()) / (df.mAh.max() - df.mAh.min())
ax3.plot(xaxis, df.volt, label='voltage')
ax4 = ax3.twinx()
ax4.grid(False)
# ax2.set_ylim([0, 2e-3])
ax4.scatter(xaxis, df.R0, marker='x', color='r', label='R0')
ax4.scatter(xaxis, df.R10, marker='p', color='k', label='R10')
ax4.legend()


test_df = df[df.curr < 0]
rem_df = test_df[test_df.volt.diff().abs() < 1e-3]
test_ica = np.gradient(test_df.mAh/1000, test_df.volt)
fig3, ax = plt.subplots(2, 1)
ax[0].plot(rem_df.mAh/1000, rem_df.volt)
ax[0].plot(test_df.mAh/1000, test_df.volt)
# ax[0].plot(test_df.volt, test_ica)
b, a = signal.butter(3, 0.01)
volt_filt_butt = signal.filtfilt(b, a, rem_df.volt)
ax[0].plot(rem_df.mAh/1000, volt_filt_butt)
# ax[1].plot(df.float_time, volt_filt_savgol)
filt_ica_butt = np.gradient(rem_df.mAh/1000, volt_filt_butt)
ax[1].plot(rem_df.volt, filt_ica_butt)
ax[1].set_ylim([0, 15])
