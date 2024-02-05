class FullCellIci(object):

    def __init__(self, df):
        self.df = df

    def plot_full_profile(self, ax):
        ax.plot(self.df.float_time - self.df.float_time.iloc[0], self.df.volt)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Voltage [V]')
        ax.grid(color='grey', alpha=0.7)
        return ax


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.style.use('ml_colors')
    ici_file = r"\\sol.ita.chalmers.se\groups\batt_lab_data\smart_cell_JG\TestBatch2_autumn2023\pickle_files_Test2477\ici_from_rpt_1.pkl"
    ici_df = pd.read_pickle(ici_file)
    test_ = FullCellIci(ici_df)
    fig, ax = plt.subplots(1, 1)
    test_.plot_full_profile(ax)
