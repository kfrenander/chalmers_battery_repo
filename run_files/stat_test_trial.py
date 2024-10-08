import pickle
import matplotlib.pyplot as plt
import os
from plot_scripts_for_papers.stat_article_plot import (extract_combinations, retrieve_samples,
                                                       make_comb_set, box_plot_on_data_dct, fail_prob_plot)
from scipy import stats
import numpy as np
import pandas as pd
from itertools import combinations
from check_current_os import get_correct_path


def plot_p_fail(p_fail_dct, lbl_case='fce'):
    # PLOT PROBABILITIES OF FALSE CONCLUSION WITH T-TEST
    x_width = 3.5
    aspect_rat = 3 / 4
    y_width = aspect_rat * x_width
    char_list = list(map(chr, range(ord('a'), ord('z') + 1)))
    n_rows, n_cols = 3, 2
    plt_num = 1
    p_fig = plt.figure(figsize=(n_cols * x_width, n_rows * y_width))
    for k_df, p_df in sort_dict(p_fail_dct).items():
        # p_fail_, pax = plt.subplots(1, 1)
        # pax = fail_prob_plot(pax, p_df)
        tmp_ax = p_fig.add_subplot(n_rows, n_cols, plt_num)
        tmp_ax = fail_prob_plot(tmp_ax, p_df)
        tmp_ax.set_ylabel('')
        tmp_ax.set_xlabel('')
        plt_lbl = f'({char_list[plt_num - 1]}) {fix_title(k_df)}'
        plt_num += 1
        # pax.set_title(fix_title(k), fontsize=11)
        # tmp_ax.set_title(fix_title(k), fontsize=11)
        if lbl_case == 'fce':
            nw_lbl = [_update_label(l.get_label()) for l in tmp_ax.lines]
            tmp_ax.legend([l for l in tmp_ax.lines], nw_lbl, fontsize=9)
        tmp_ax.set_ylim((0, 1))
        tmp_ax.text(0.5, -0.22, plt_lbl, transform=tmp_ax.transAxes, fontsize=13,
                    horizontalalignment='center')
        # p_fail_.savefig(os.path.join(fig_op_dir, f'{k[0]}_and_{k[1]}.png'), dpi=400)
    # sub_fig_p = set_font_sizes(sub_fig_p, 12)
    # sub_fig_p.suptitle('Real Data', fontsize=15)
    p_fig.supxlabel('Number of replicates used', fontsize=14)
    p_fig.supylabel('Probability of false prediction', fontsize=14)
    p_fig.tight_layout()
    return p_fig


def sort_dict(dct):
    my_keys = list(dct.keys())
    my_keys.sort()
    return {k_dct: dct[k_dct] for k_dct in my_keys}


def fix_title(k_):
    return f'{k_[0]} with {k_[1]}'


def _update_label(rp_str):
    nm, nbr = rp_str.split(" ")
    fce_ = (int(nbr) - 1) * 40
    return f'FCE {fce_:.0f}'


def pass_fail_stat_test(p_val, threshold=0.01):
    if p_val >= threshold:
        return 0
    else:
        return 1


def pass_fail_pt_estim(val_, ref):
    if val_ == ref:
        return 1
    else:
        return 0


def perform_full_stat_test(cap_dct, method='ttest'):
    list_of_test_combinations = [k_iter for k_iter in combinations(cap_dct.keys(), 2)]
    dct_of_likelihoods = {}
    dct_of_ref_probs = {}
    for cmb in list_of_test_combinations:
        dct_of_cell_combinations = make_comb_set({k: cap_dct[k] for k in cmb})
        ref_set_1 = cap_dct[cmb[0]].filter(like='cap').columns
        ref_set_2 = cap_dct[cmb[1]].filter(like='cap').columns
        nbr_of_rpts = min(np.array([df.filter(like='cap').shape[0] for k, df in cap_dct.items()]))
        nbr_of_cells = min(np.array([df.filter(like='cap').shape[1] for k, df in cap_dct.items()]))
        prob_df = pd.DataFrame(index=[f'rpt_{m}' for m in range(2, nbr_of_rpts+1)],
                               columns=[f'{n}_cells' for n in range(2, nbr_of_cells + 1)])
        ref_prob_df = pd.DataFrame(index=[f'rpt_{m}' for m in range(2, nbr_of_rpts + 1)],
                               columns=[f'{n}_cells' for n in range(2, nbr_of_cells + 1)])
        for m in range(2, nbr_of_rpts + 1):
            for n, c_cmb in dct_of_cell_combinations.items():
                d_set_ref1 = retrieve_samples(cap_dct[cmb[0]], m, ref_set_1)
                d_set_ref2 = retrieve_samples(cap_dct[cmb[1]], m, ref_set_2)
                ref_prob = perform_stat_test(d_set_ref1, d_set_ref2, method=method)
                prob_set = [perform_stat_test(retrieve_samples(cap_dct[cmb[0]], m, c1[0]),
                                              retrieve_samples(cap_dct[cmb[1]], m, c1[1]), method=method) for c1 in c_cmb]
                anova_res_set = [pass_fail_stat_test(p) for p in prob_set]
                stat_test_res = np.array([pass_fail_pt_estim(t_res, pass_fail_stat_test(ref_prob)) for t_res in anova_res_set])
                prob_df.loc[f'rpt_{m}', f'{n}_cells'] = 1 - sum(stat_test_res)/len(stat_test_res)
                ref_prob_df.loc[f'rpt_{m}', f'{n}_cells'] = ref_prob
        prob_df.dropna(inplace=True)
        dct_of_likelihoods[cmb] = prob_df
        dct_of_ref_probs[cmb] = ref_prob_df
    return dct_of_likelihoods, dct_of_ref_probs


def perform_stat_test(dset1, dset2, method='ttest'):
    if method == 'ttest':
        # Method chosen is dependent t-test for paired samples
        _, p = stats.ttest_rel(dset1, dset2)
    elif method == 'anova':
        # Method chosen is oneway ANOVA, appropriate for comparing three or more datasets
        _, p = stats.f_oneway(dset1, dset2)
    elif method == 'ttest_ind':
        # Method chosen is Student's t-test
        _, p = stats.ttest_ind(dset1, dset2)
    else:
        print('Not recognised method, return \'None\'')
        p = None
    return p


def calc_group_var(data_set):
    n_set = data_set.shape[0]
    return np.sum((data_set - data_set.mean()) ** 2) / (n_set - 1)


def calc_t_stat(d1, d2):
    d = d1 - d2
    n_data = d.shape[0]
    dm = d.mean()
    sd = np.sqrt(calc_group_var(d))
    t_stat = dm / (sd / np.sqrt(n_data))
    return t_stat


if __name__ == '__main__':
    import time
    from check_current_os import get_base_path_batt_lab_data
    plt.style.use('kelly_colors')
    start = time.time()
    SCALE = 1.5
    CASE_NAME = 'reduce'
    data_files = {
        "Test D": f"stat_test/synthetic_data_base_dir/chng_noise_data_sc{SCALE}_updated_method/Test_D_{CASE_NAME}_noise.pkl",
        "Test A": f"stat_test/synthetic_data_base_dir/chng_noise_data_sc{SCALE}_updated_method/Test_A_{CASE_NAME}_noise.pkl",
        "Test B": f"stat_test/synthetic_data_base_dir/chng_noise_data_sc{SCALE}_updated_method/Test_B_{CASE_NAME}_noise.pkl",
        "Test C": f"stat_test/synthetic_data_base_dir/chng_noise_data_sc{SCALE}_updated_method/Test_C_{CASE_NAME}_noise.pkl"
    }
    data_files = sort_dict(data_files)
    BASE_BATT_LAB_DIR = get_base_path_batt_lab_data()
    BASE_OUTPUT_DIR = r"Z:\StatisticalTest\figures_from_synthetic_data"
    data_files = {k: os.path.join(BASE_BATT_LAB_DIR, nm) for k, nm in data_files.items()}
    savefig = 0

    dta = {}
    for k, val in data_files.items():
        val = get_correct_path(val)
        with open(val, 'rb') as f:
            dta[k] = pickle.load(f)

    cmb_dct = extract_combinations(dta['Test A'])
    ex_data1 = retrieve_samples(dta['Test A'], rpt_nbr=4, test_set=cmb_dct[3][0])
    ex_data2 = retrieve_samples(dta['Test B'], rpt_nbr=4, test_set=cmb_dct[3][0])
    p_anova = perform_stat_test(ex_data1, ex_data2, method='anova')
    p_ttest = perform_stat_test(ex_data1, ex_data2, method='ttest')
    print('Data read in complete...')
    print('Starting statistical testing')
    p_fail_ttest_rel, ref_prob_ttest_rel = perform_full_stat_test(dta, method='ttest')
    print('ttest_rel done...')
    # p_fail_ttest_ind, ref_prob_ttest_ind = perform_full_stat_test(dta, method='ttest_ind')
    print('ttest_ind done...')
    # p_fail_anova, ref_prob_anova = perform_full_stat_test(dta, method='anova')
    print('ANOVA done...')

    # CHECK NORMALITY OF DATA WITH QQ-PLOT ON EACH RPT
    x_w = 3.25
    y_w = 3.25
    for cs, df in dta.items():
        n = df.shape[0]
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*x_w, nrows*y_w))
        fig = plt.figure(figsize=(ncols*x_w, nrows*y_w))
        for k, rp in enumerate(df.filter(like='cap').index):
            plt_ax = fig.add_subplot(nrows, ncols, k + 1)
            data = df.filter(like='cap').loc[rp, :]
            stats.probplot(data, dist='norm', plot=plt, rvalue=True)
            # plt_ax.set_ylabel(plt_ax.get_ylabel(), fontsize=12)
            # plt_ax.set_xlabel(plt_ax.get_xlabel(), fontsize=12)
            plt_ax.set_xlabel('')
            plt_ax.set_ylabel('')
            plt_ax.set_title(rp, fontsize=12)
            print(stats.kstest(data, stats.norm.cdf).pvalue)
        fig.supxlabel('Theoretical Quantiles', fontsize=16)
        fig.supylabel('Ordered values', fontsize=16)
        fig.tight_layout()
        if savefig:
            fig.savefig(os.path.join(BASE_OUTPUT_DIR, cs, f'qq_plot_{cs}.png'), dpi=400)

    # CHECK HOMOGENOUS VARIANCE BETWEEN TESTS
    rpt_comp = {f'rpt_{k}': {nm: dta[nm].filter(like='cap').loc[f'rpt_{k}', :] for nm in dta.keys()}
                    for k in range(2, 8)}
    b_fig = box_plot_on_data_dct(rpt_comp)


    # PLOT PROBABILITIES OF FALSE CONCLUSIONS FOR TWO TYPES OF T-TEST
    p_fig_ttest_rel = plot_p_fail(p_fail_ttest_rel)
    p_fig_ttest_rel.savefig(os.path.join(BASE_OUTPUT_DIR, f'fail_prob_ttest_rel_fce_{CASE_NAME}_noise.png'), dpi=400)
    p_fig_ttest_rel.savefig(os.path.join(BASE_OUTPUT_DIR, f'fail_prob_ttest_rel_fce_{CASE_NAME}_noise.pdf'))

    # p_fig_ttest_ind = plot_p_fail(p_fail_ttest_ind)
    # p_fig_ttest_ind.savefig(os.path.join(BASE_OUTPUT_DIR, f'fail_prob_ttest_ind_fce.png'), dpi=400)
    # p_fig_ttest_ind.savefig(os.path.join(BASE_OUTPUT_DIR, f'fail_prob_ttest_ind_fce.pdf'))
    #
    # p_fig_anova = plot_p_fail(p_fail_ttest_ind)
    # p_fig_anova.savefig(os.path.join(BASE_OUTPUT_DIR, f'fail_prob_anova_fce.png'), dpi=400)
    # p_fig_anova.savefig(os.path.join(BASE_OUTPUT_DIR, f'fail_prob_anova_fce.pdf'))

    end = time.time()
    print(f'Total time elapsed is {(end - start)/60:.2f} min')
