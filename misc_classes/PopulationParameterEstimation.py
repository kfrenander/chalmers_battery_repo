import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt


class PopulationParameterEstimator(object):

    def __init__(self, df):

        self.df = df
        self.scale_factor = self.set_perc_scaling()
        self.cmb = self.extract_combinations()
        self.mu_, self.sig_, self.sig_mu, self.sig_sig = self.estimate_population_parameters()
        self.rse = self.estimate_rse()


    def extract_combinations(self):
        _, n = self.df.filter(like='cap').shape
        cols = self.df.filter(like='cap').columns
        return {n_cells: [k for k in combinations(cols, n_cells)] for n_cells in range(2, n + 1)}

    def convert_to_percent(self):
        if not self.scale_factor:
            self.set_perc_scaling()
        df = self.df.filter(like='cap') * self.scale_factor
        return df

    def estimate_rse(self):
        return 100*self.sig_sig.divide(self.df.loc['rpt_2':, 'Q_std']*100, axis=0)

    def estimate_population_parameters(self):
        m, n = self.df.filter(like='cap').shape
        col_names = [f'{i}_cells' for i in range(2, n + 1)]
        sig_mu_ = pd.DataFrame(index=self.df.index.copy(), columns=col_names)
        sig_sig_ = pd.DataFrame(index=self.df.index.copy(), columns=col_names)
        for i in range(2, m + 1):
            mu_ = {k: [self.scale_factor * np.mean(self.retrieve_samples(self.df, i, c)) for c in cmb_list]
                   for k, cmb_list in self.cmb.items()}
            sig_ = {k: [self.scale_factor * np.std(self.retrieve_samples(self.df, i, c), ddof=1) for c in cmb_list]
                    for k, cmb_list in self.cmb.items()}
            sig_mu = {k: np.std(vals, ddof=1) for k, vals in mu_.items()}
            sig_sig = {k: np.std(vals, ddof=1) for k, vals in sig_.items()}
            sig_mu_.iloc[i - 1, :] = [sig_mu_i for sig_mu_i in sig_mu.values()]
            sig_sig_.iloc[i - 1, :] = [sig_sig_i for sig_sig_i in sig_sig.values()]
        return mu_, sig_, sig_mu_.dropna(how='all').dropna(how='all', axis=1), sig_sig_.dropna(how='all').dropna(how='all', axis=1)

    def plot_var_mu(self, ax, case_to_plot='mu', sig_thresh=None, fig_title='', fmt='fce_num'):
        plt.style.use('kelly_colors')
        if case_to_plot == 'mu':
            for idx in self.sig_mu.index:
                ax.plot(self.sig_mu.columns, self.sig_mu.loc[idx, :], label=self._clean_rpt_name(idx, fmt=fmt))
            y_label = r'Variance $\sigma_{\mu}$'
        elif case_to_plot == 'sig':
            for idx in self.sig_sig.index:
                ax.plot(self.sig_sig.columns, self.sig_sig.loc[idx, :], label=self._clean_rpt_name(idx, fmt=fmt))
            y_label = r'Standard error $s_n$'
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xlabel('Number of replicates', fontsize=12)
        ax.grid(True)
        if sig_thresh:
            ax.axhline(sig_thresh, linestyle='dashed', color='black', label=r'Max $s_n$')
        ax.legend(ncols=2)
        return ax

    def set_perc_scaling(self):
        max_q = self.df.filter(like='cap').to_numpy().max()
        if max_q < 10:
            return 100
        else:
            return 1

    @staticmethod
    def _clean_rpt_name(rpt_str, fmt='rpt_num'):
        try:
            nm, nbr = rpt_str.split("_")
            if fmt == 'rpt_num':
                return f'{nm.upper()} {nbr}'
            elif fmt == 'fce_num':
                fce_ = (int(nbr) - 1) * 40
                return f'FCE {fce_:.0f}'
            else:
                print('Unknown format')
                return None
        except ValueError:
            print(f'\'{rpt_str}\' not unpackable, return original string')
            return rpt_str
        except:
            print('Unknown error, return None')
            return None

    @staticmethod
    def retrieve_samples(df, rpt_nbr, test_set):
        return df.loc[f'rpt_{rpt_nbr}', [*test_set]]


if __name__ == '__main__':
    import pandas as pd
    import os
    from check_current_os import get_base_path_batt_lab_data
    BASE_BATTLAB_PATH = get_base_path_batt_lab_data()
    df = pd.read_pickle(os.path.join(BASE_BATTLAB_PATH, "stat_test\processed_data\Test1_1.pkl"))
    test_case = PopulationParameterEstimator(df)
    dct_of_pkl = {
        "2-1": r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\processed_data\Test2_1.pkl",
        "1-1": r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\processed_data\Test1_1.pkl",
        "2-2": r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\processed_data\Test2_2.pkl",
        "1-2": r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\processed_data\Test1_2.pkl"
    }
    df_dict = {k: pd.read_pickle(f_) for k, f_ in dct_of_pkl.items()}
    ppe_dict = {k: PopulationParameterEstimator(df) for k, df in df_dict.items()}
