from PythonScripts.misc_classes.ExponentialDecayFitter import ExponentialDecayFitter as edf
import pickle
from PythonScripts.plot_scripts_for_papers.stat_article_plot import calculate_eol_stat
import os
import matplotlib.pyplot as plt


data_files = [r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\processed_data\Test1_2.pkl",
              r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\processed_data\Test2_1.pkl",
              r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\processed_data\Test2_2.pkl",
              r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\processed_data\Test1_1.pkl"
              ]
old_names = [f'Test{k}' for k in ['1_1', '1_2', '2_1', '2_2']]
new_names = [f'Test {j}' for j in ['A', 'B', 'C', 'D']]
rename_dict = dict(zip(old_names, new_names))
data_set = {}
for fname in data_files:
    with open(fname, 'rb') as h:
        test_key = rename_dict[os.path.split(fname)[-1].strip('.pkl')]
        data_set[test_key] = pickle.load(h)

one_test_fits = {}
for tk, df in data_set.items():
    one_test_fits[tk] = {k: edf(df.FCE, df[k]) for k in df.filter(like='cap').columns}

output_dir = r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\processed_data"
for k in one_test_fits.keys():
    op_file = os.path.join(output_dir, f'fit_one_replicate_{k.replace(" ", "_")}.pkl')
    with open(op_file, 'wb') as h:
        pickle.dump(one_test_fits[k], h)

eol_fits = calculate_eol_stat(data_set)
plt.style.use('ml_colors')

fit_case = eol_fits['Test B']['5_test'][0]
for tc in eol_fits.keys():
    for k, dct in eol_fits[tc].items():
        fig_dir = os.path.join(r"Z:\StatisticalTest\figures", f"{tc}", "data_fits")
        if not os.path.isdir(fig_dir):
            os.mkdir(fig_dir)
        for n, fit_case in dct.items():
            fig, ax = plt.subplots(1, 1)
            ax = fit_case.plot_fit(ax)
            ax.set_xlabel('FCE [-]', fontsize=15)
            ax.set_ylabel('Normalised capacity [-]', fontsize=15)
            param_vals = {k: fit_case.result.params[k].value for k in fit_case.result.params.keys()}
            param_str = '\n'.join([fr'{k}: {val:.2f}' for k, val in param_vals.items()])
            plt.text(0.75, 0.85, param_str,
                     fontsize=12,
                     bbox=dict(facecolor='wheat', alpha=0.5),
                     horizontalalignment='left',
                     verticalalignment='center', transform=ax.transAxes
                     )
            fig_name = os.path.join(fig_dir, f'{k}_fit_nbr_{n}.pdf')
            fig.savefig(fig_name, dpi=300)
            plt.close('all')
