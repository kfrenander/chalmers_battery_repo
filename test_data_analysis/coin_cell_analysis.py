import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from test_data_analysis.basic_plotting import cap_v_volt_multicolor
from test_data_analysis.coin_cell_reader import coin_cell_data_reader
plt.style.use('ggplot')

data_file = "Z:\Provning\CoinCell\Trial20191111\Trial20191111_Run01_KF.bdat"
df = coin_cell_data_reader(data_file)
op_df = df[['time', 'volt', 'curr']]

cyc_list = df.Repeat.unique()
chrg_cap = []
dchg_cap = []

for cyc in cyc_list:
    chrg_cap.append(df[df.Repeat == cyc].cap.max())
    dchg_cap.append(abs(df[df.Repeat == cyc].cap.min()))

chrg_cap_arr = np.array(chrg_cap)
dchg_cap_arr = np.array(dchg_cap)
coul_eff = dchg_cap_arr / chrg_cap_arr
fig = plt.figure()
plt.plot(cyc_list, chrg_cap_arr*1000, '--*', color='red', label='Charge Cap [mAh]')
plt.plot(cyc_list, dchg_cap_arr*1000, '--*', color='black', label='Discharge cap [mAh]')
plt.grid(True)
plt.xlabel('Cycle number')
plt.ylabel('Capacity [mAh]')
plt.legend()

fig, ax = plt.subplots(1, 1)
ax.plot(cyc_list, coul_eff, '--.', color='red')
ax.set_xlabel('Cycle number')
ax.set_ylabel('Coulombic efficiency')
ax.set_title('Coulombic efficiency of cycling')

df_1c = df[df.Repeat > 2]
fig, ax = plt.subplots(1, 1)
ax.plot(df_1c.time, df_1c.volt, color='red')
ax2 = ax.twinx()
ax2.plot(df_1c.time, df_1c.curr*1000, color='blue')
ax2.set_ylabel('Current [mA]')
ax2.grid(False)
ax.set_ylabel('Voltage [V]')
ax.set_xlabel('Time')
ax.set_title('Voltage and Current v time')
plt.tight_layout()

df_1c.loc[:, 'mAh'] = cumulative_trapezoid(df_1c.curr, df_1c.time, initial=0) / 3.6
fig, ax = plt.subplots(1, 1)
ax.plot(df_1c.mAh, df_1c.volt)
fig2 = cap_v_volt_multicolor(df_1c, name='1C Cycling')
fig2.tight_layout()

plt.show()
