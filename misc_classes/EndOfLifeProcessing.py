import numpy as np
import pandas as pd
from scipy.stats import norm, bootstrap


class EndOfLifeProcessing(object):

    def __init__(self, d_set):
        self.dataset = d_set
        self.ref_value = self.id_ref_value()
        self.norm_dist_params = {k: norm.fit(arr) for k, arr in self.dataset.items()}
        self.df_oe_risk_raw = self.fill_oe_risk_df_raw()
        self.df_oe_risk_norm = self.fill_oe_risk_df_norm()

    def id_ref_value(self):
        return self.dataset['8_test'][0]

    def find_oe_risk_raw(self, arr, oe_lvl):
        nbr_oe = len(arr[arr > oe_lvl*self.ref_value])
        return nbr_oe / len(arr)

    def fill_oe_risk_df_raw(self, oe_lvls=np.array([1.02, 1.05, 1.1])):
        df = pd.DataFrame(index=oe_lvls, columns=self.dataset.keys())
        for oe in oe_lvls:
            for k in self.dataset.keys():
                rsk = self.find_oe_risk_raw(arr=self.dataset[k], oe_lvl=oe)
                # print(f'Risk for oe is {rsk} at oe={oe} and {k}')
                df.loc[oe, k] = rsk
        return df

    def fill_oe_risk_df_norm(self, oe_lvls=np.array([1.02, 1.05, 1.1])):
        df = pd.DataFrame(index=oe_lvls, columns=self.dataset.keys())
        for k, arr in self.dataset.items():
            loc, scale = self.norm_dist_params[k]
            for oe in oe_lvls:
                rsk = 1 - norm.cdf(oe * self.ref_value, loc=loc, scale=scale)
                # print(f'Risk for oe is {rsk} at oe={oe} and {k}')
                df.loc[oe, k] = rsk
        return df

    def find_btstrp_pp(self):
        btstrp_data = {k: (data, ) for k, data in self.dataset.items() if '8' not in k}
        btstrp_std = {k: bootstrap(dta, np.std) for k, dta in btstrp_data.items()}
        btrst_mean = {k: bootstrap(dta, np.mean) for k, dta in btstrp_data.items()}
        return btrst_mean, btstrp_std

    def _plot_norm_distribution(self, ax, n_cases='all', shade_area=0):
        w_case_loc, w_case_sig = self.norm_dist_params['2_test']
        t_rng = np.linspace(w_case_loc - 3*w_case_sig, w_case_loc + 3*w_case_sig, 1000)
        if n_cases == 'all':
            for k, vals in self.norm_dist_params.items():
                ax.plot(t_rng, norm.pdf(t_rng, loc=vals[0], scale=vals[1]), label=k)
        else:
            for k, vals in self.norm_dist_params.items():
                if k in n_cases:
                    ax.plot(t_rng, norm.pdf(t_rng, loc=vals[0], scale=vals[1]), label=k)
        self._plot_oe_line(ax)
        # ax.axvline(self.ref_value * 1.05, linestyle='dashed', color='forestgreen', label='Overestimation\nlimit - 5%')
        return ax

    def _plot_histogram(self, ax, n_cases='2_test', normal_bool=False, col_oe=0):
        arr = self.dataset[n_cases]
        ax.hist(arr, bins=15,
                edgecolor='black',
                rwidth=0.8,
                density=normal_bool,
                color='lightgray')
        if col_oe:
            for bar in ax.containers[0]:
                x = bar.get_x() + 0.5 * bar.get_width()
                if x > self.ref_value * 1.05:
                    bar.set_color('indianred')
                    bar.set_edgecolor('black')
            self._plot_oe_line(ax)
        return ax

    def _plot_oe_line(self, ax):
        ax.axvline(self.ref_value * 1.05, linestyle='dashed', color='forestgreen',
                   label='Overestimation\nlimit - 5%')
        return ax


if __name__ == '__main__':
    import pickle
    import matplotlib.pyplot as plt
    plt.style.use('ml_colors')
    test_case = "Test2_1"
    d_file = rf"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\processed_data\t_to_eol_{test_case}.pkl"
    with open(d_file, 'rb') as h:
        dataset = pickle.load(h)
    test_ = EndOfLifeProcessing(dataset)
    mu, sig = test_.find_btstrp_pp()
    test_plots = 1
    if test_plots:
        fig, ax = plt.subplots(1, 1)
        test_._plot_norm_distribution(ax)
        ax.set_xlabel('FCE to EOL [-]', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.legend()
        fig2, ax2 = plt.subplots(1, 1)
        ax2 = test_._plot_histogram(ax2, col_oe=1)
