import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid


def coin_cell_data_reader(file_path):
    col_list = ['Mode', 'Alarm', 'Current_Range', 'Step', 'Repeat', 'time', 'step_time', 'curr',
                'volt', 'aux_volt', 'cap', 'cap_CV', 'energy', 'energy_CV', 'HFR1', 'HFR2']

    df = pd.read_csv(file_path, sep='\t', skiprows=11, names=col_list)

    # Defining unique step numbers for every charge or discharge step
    nbr_of_steps = df[df.step_time.diff() < 0].shape[0]
    my_range = np.arange(1, nbr_of_steps + 1, 1)
    df.loc[df.step_time.diff() < 0, 'step'] = my_range
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    return df


if __name__ == '__main__':
    from test_data_analysis.basic_plotting import cap_v_volt_multicolor
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    df = coin_cell_data_reader("Z:\Provning\CoinCell\Trial20191111\Trial20191111_Run01_KF.bdat")
    df_1c = df[df.Repeat > 2]
    fig, ax = plt.subplots(1, 1)
    ax.plot(df_1c.time, df_1c.volt, color='red')
    ax2 = ax.twinx()
    ax2.plot(df_1c.time, df_1c.curr * 1000, color='blue')
    ax2.set_ylabel('Current [mA]')
    ax2.grid(False)
    ax.set_ylabel('Voltage [V]')
    ax.set_xlabel('Time')
    ax.set_title('Voltage and Current v time')
    plt.tight_layout()

    df_1c['mAh'] = cumulative_trapezoid(df_1c.curr, df_1c.time, initial=0) / 3.6
    fig, ax = plt.subplots(1, 1)
    ax.plot(df_1c.mAh, df_1c.volt)
    fig2 = cap_v_volt_multicolor(df_1c, name='1C Cycling')
    fig2.tight_layout()
