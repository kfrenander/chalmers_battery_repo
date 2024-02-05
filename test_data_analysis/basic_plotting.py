import matplotlib.pyplot as plt
from matplotlib.collections import  LineCollection
import pandas as pd
import numpy as np
#plt.style.use('chalmers_kf')


def cap_v_volt_multicolor(df, name='', col='step'):
    fig4, axs = plt.subplots(1, 1)
    if df.curr.min() >= 0:
        x = df.mAh.max() - df.mAh.values
    else:
        x = df.mAh - df.mAh.min()
    y = df.volt.values
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(df[col].min(), df[col].max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(df[col][:-1])
    lc.set_linewidth(0.8)

    line = axs.add_collection(lc)
    # fig.colorbar(line, ax=axs)
    cbar = plt.colorbar(line)
    if col == 'step':
        cbar.set_label('Cycle number')
    else:
        cbar.set_label(col)
    xspan = x.max() - x.min()
    axs.set_xlim(x.min() - xspan / 50, x.max() + xspan / 50)
    axs.set_ylim(np.floor(y.min() - 0.1), y.max() + 0.1)
    axs.set_xlabel('Capacity [mAh]', fontsize=13)
    axs.set_ylabel('Voltage [V]', fontsize=13)
    axs.set_title('Capacity v Voltage over cycles for {0}'.format(name))
    axs.grid(True)
    plt.tight_layout()
    return fig4


def volt_curr_plot(df):
    fig1, ax = plt.subplots(1, 1)
    ax.plot(df.float_time / 3600, df.curr, linewidth=0.8, label='Current')
    ax.set_xlabel('Time [h]', fontsize=13)
    ax.set_ylabel('Current [A]', fontsize=13)
    # ax.grid(True)
    ax2 = ax.twinx()
    ax2.plot(df.float_time / 3600, df.volt, color='r', linewidth=0.8, label='Voltage')
    ax2.set_ylabel('Voltage [V]', fontsize=13)
    ax2.grid(False)
    # fig1.legend(fontsize=11)
    plt.tight_layout()
    return fig1


def ica_plot(df):
    fig, ax = plt.subplots(1, 1)
    ax.plot(df.volt, df.ica_filt, label='ICA')
    ax.set_xlabel('Voltage [V]', fontsize=13)
    ax.set_ylabel('dQdV', fontsize=13)
    ax.set_title('Incremental capacity analysis', fontsize=16)
    plt.tight_layout()
    return fig


def dva_plot(df):
    fig, ax = plt.subplots(1, 1)
    ax.plot((df.mAh - df.mAh.min()) / 1000, df.dva_filt, label='DVA')
    ax.set_xlabel('Capacity [Ah]', fontsize=13)
    ax.set_ylabel('dVdQ', fontsize=13)
    ax.set_title('Differential Voltage Analysis', fontsize=16)
    plt.tight_layout()
    return fig


def show_figure(fig):
    """
    Create a dummy figure and use its manager to display "fig"
    """
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    return None