import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from itertools import combinations
import os


def check_v_ref(tst_case, cl=0.01):
    ref_val = aov_ref[tst_case]
    p_fail = {}
    if ref_val < cl:
        for n, arr in aov_res[tst_case].items():
            p_fail[n] = np.sum(arr > cl) / np.size(arr)
    else:
        for n, arr in aov_res[tst_case].items():
            p_fail[n] = np.sum(arr < cl) / np.size(arr)
    return p_fail


plt.style.use('kelly_colors')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.rcParams['text.usetex'] = True
op_dir = r"Z:\StatisticalTest\figures_new_test_names"


file_list = [r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\processed_data\fit_one_replicate_Test_C.pkl",
             r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\processed_data\fit_one_replicate_Test_D.pkl",
             r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\processed_data\fit_one_replicate_Test_A.pkl",
             r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\processed_data\fit_one_replicate_Test_B.pkl"]
test_names = [f'Test {j}' for j in ['A', 'B', 'C', 'D']]
file_name_dct = dict(zip(test_names, file_list))
data_files = {}
for k, fname in file_name_dct.items():
    with open(fname, 'rb') as h:
        data_files[k] = pickle.load(h)

eol_dict = {}
for tst in data_files:
    eol_dict[tst] = {k: edf.eol_fit[0] for k, edf in data_files[tst].items()}
eol_df = pd.DataFrame.from_dict(eol_dict)

# Eg of how to call anova on scipy
# F, p = f_oneway(eol_df.loc['cap_1':'cap_3', 'Test A'], eol_df.loc['cap_1':'cap_3', 'Test D'])
cell_names = eol_df.index
cell_combination = {n: [k for k in combinations(cell_names, n)] for n in range(2, 8)}
test_combination = [k for k in combinations(test_names, 2)]
aov_res = {}
for tst_cmb in test_combination:
    tmp_res = {}
    for n, lst in cell_combination.items():
        tmp_res[n] = np.array([[f_oneway(eol_df.loc[[*c], tst_cmb[0]], eol_df.loc[[*d], tst_cmb[1]]).pvalue for c in lst] for d in lst])
    aov_res[f'{tst_cmb[0]}_{tst_cmb[1]}'.replace(" ", "_")] = tmp_res
aov_ref = {f'{tst[0]}_{tst[1]}'.replace(" ", "_"): f_oneway(eol_df.loc[:, tst[0]], eol_df.loc[:, tst[1]]).pvalue for tst in test_combination}
aov_df = pd.DataFrame.from_dict(aov_ref, orient='index', columns=['p_val'])

for cl in np.arange(0.01, 0.06, 0.01):
    p_fail_df = pd.DataFrame.from_dict({cs: check_v_ref(cs, cl=cl) for cs in aov_res.keys()})

    fig, ax = plt.subplots()
    p_fail_df.plot(ax=ax, color=colors[:6], linestyle='dashed')
    ax.set_xlabel('Number of Replicates', fontsize=16)
    ax.set_ylabel(r'$P_{false}$', fontsize=16)
    ax.set_title(f'Conf level={cl}')
    ax.grid(True)
    ax.legend(ncols=2)
    ax.set_ylim((0, 1))
    fig.savefig(os.path.join(op_dir, f'P_false_conf_level_{cl*100:.0f}.pdf'))
