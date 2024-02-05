import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from rpt_data_analysis.ReadRptClass import OrganiseRpts
from misc_classes.PopulationParameterEstimation import PopulationParameterEstimator as ppe
from misc_classes.ExponentialDecayFitter import ExponentialDecayFitter as edf
from misc_classes.EndOfLifeProcessing import EndOfLifeProcessing as eolpp
from itertools import combinations
from scipy.stats import f_oneway, norm, zscore, ttest_rel
import os
from backend_fix import fix_mpl_backend
fix_mpl_backend()
my_style = 'kelly_colors'
plt.style.use(my_style)
lbl_font = {'weight': 'normal',
            'size': 14}


def set_font_sizes(fig, fnt_sz):
    for k in fig.axes:
        tmp_xlab = k.get_xlabel()
        k.set_xlabel(tmp_xlab, fontsize=fnt_sz)
        tmp_ylab = k.get_ylabel()
        k.set_ylabel(tmp_ylab, fontsize=fnt_sz)
        tmp_ttl = k.get_title()
        k.set_title(tmp_ttl, fontsize=fnt_sz)
        lgd = k.get_legend()
        [txt.set_fontsize(fnt_sz*2/3) for txt in lgd.texts]
        tmp_xticks = k.get_xticks()
        k.set_xticklabels([f'{xt:.0f}' for xt in tmp_xticks], fontsize=fnt_sz * 3 / 4)
        tmp_yticks = k.get_yticks()
        k.set_yticklabels([f'{yt:.1f}' for yt in tmp_yticks], fontsize=fnt_sz * 3 / 4)
    return fig


def make_data_set(inp_dct):
    ## Create overall data for each test
    full_data_dct = {}
    for t_name, c_dct in inp_dct.items():
        tmp_df = pd.DataFrame()
        col_names = []
        for c_id, df in c_dct.items():
            ch_nbr = re.findall(r"\d", c_id)[-1:]
            col_names.append(f'cap_{ch_nbr[0]}')
            if tmp_df.empty:
                tmp_df = df.cap_relative
            else:
                tmp_df = pd.concat([tmp_df, df.cap_relative], axis=1)
        tmp_df.columns = col_names
        tmp_df['FCE'] = [40 * (int(idx.split("_")[1]) - 1) for idx in tmp_df.index]
        tmp_df['Q_mean'] = tmp_df.filter(like='cap').mean(axis=1)
        tmp_df['Q_std'] = tmp_df.filter(like='cap').std(axis=1)
        full_data_dct[t_name] = tmp_df
    return full_data_dct


def flatten_list(nested_list):
    flat_list = []
    for sublist in nested_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def extract_combinations(df):
    _, n = df.filter(like='cap').shape
    cols = df.filter(like='cap').columns
    return {n_cells: [k for k in combinations(cols, n_cells)] for n_cells in range(2, n + 1)}


def make_comb_set(inp_dct, symmetry_data=False):
    nbr_of_tests = min(np.array([df.filter(like='cap').shape[1] for k, df in inp_dct.items()]))
    if symmetry_data:
        test_name_list = set(flatten_list([df.filter(like='cap').columns.to_list() for k, df in inp_dct.items()]))
        comb_dct = {f'{n}': [(k, k) for k in combinations(test_name_list, n)] for n in range(2, nbr_of_tests + 1)}
    else:
        cell_names = {k: df.filter(like='cap').columns.to_list() for k, df in inp_dct.items()}
        n_max = find_nbr_of_cells(inp_dct)
        comb_dct = {}
        for n in range(2, n_max + 1):
            t_names = [k for k in cell_names.keys()]
            comb_dct[f'{n}'] = [(k1, k2) for k1 in combinations(cell_names[t_names[0]], n)
                                for k2 in combinations(cell_names[t_names[1]], n)]
    return comb_dct


def clean_data(df, z_lim=2, cln_md='mean'):
    zdf = abs(zscore(df.filter(like='cap'), axis=1))
    if cln_md == 'mean':
        flt_df = df.filter(like='cap').loc[:, zdf.mean() < z_lim]
    elif cln_md == 'any_ol':
        zdf = zdf.fillna(0)
        cols_to_keep = zdf[zdf < z_lim].dropna(how='any', axis=1).columns
        flt_df = df.loc[:, cols_to_keep]
    flt_df.loc[:, 'Q_mean'] = flt_df.filter(like='cap').mean(axis=1)
    flt_df.loc[:, 'Q_std'] = flt_df.filter(like='cap').std(axis=1)
    flt_df.loc[:, 'FCE'] = df['FCE']
    return flt_df


def estimate_population_parameters(df):
    cmb = extract_combinations(df)
    m, n = df.filter(like='cap').shape
    sig_mu_ = pd.DataFrame(index=df.index, columns=[f'{i}_cells' for i in range(2, n + 1)])
    sig_sig_ = pd.DataFrame(index=df.index, columns=[f'{i}_cells' for i in range(2, n + 1)])
    for i in range(2, m + 1):
        mu_ = {k: [np.mean(retrieve_samples(df, i, c)) for c in cmb_list] for k, cmb_list in cmb.items()}
        sig_ = {k: [np.std(retrieve_samples(df, i, c), ddof=1) for c in cmb_list] for k, cmb_list in cmb.items()}
        sig_mu = {k: np.std(vals, ddof=1) for k, vals in mu_.items()}
        sig_sig = {k: np.std(vals, ddof=1) for k, vals in sig_.items()}
        sig_mu_.iloc[i - 1, :] = [sig_mu_i for sig_mu_i in sig_mu.values()]
        sig_sig_.iloc[i - 1, :] = [sig_sig_i for sig_sig_i in sig_sig.values()]
    return sig_mu_.dropna(how='all').dropna(how='all', axis=1), sig_sig_.dropna(how='all').dropna(how='all', axis=1)


def find_nbr_of_rpts(inp_dct):
    return min(np.array([df.filter(like='cap').shape[0] for k, df in inp_dct.items()]))


def find_nbr_of_cells(inp_dct):
    return min(np.array([df.filter(like='cap').shape[1] for k, df in inp_dct.items()]))


def perform_f_oneway(dset1, dset2):
    F_stat, p = f_oneway(dset1, dset2)
    return p


def perform_ttest(dset1, dset2):
    _, p = ttest_rel(dset1, dset2)
    return p


def retrieve_samples(df, rpt_nbr, test_set):
    return df.loc[f'rpt_{rpt_nbr}', [*test_set]]


def pass_fail_anova(val, threshold=0.01):
    if val >= threshold:
        return 0
    else:
        return 1


def pass_fail_test(val, exp):
    if val == exp:
        return 1
    else:
        return 0


def perform_full_anova(cap_dct):
    list_of_test_combinations = [k for k in combinations(cap_dct.keys(), 2)]
    dct_of_likelihoods = {}
    dct_of_ref_probs = {}
    for cmb in list_of_test_combinations:
        dct_of_cell_combinations = make_comb_set({k: cap_dct[k] for k in cmb})
        ref_set_1 = cap_dct[cmb[0]].filter(like='cap').columns
        ref_set_2 = cap_dct[cmb[1]].filter(like='cap').columns
        nbr_of_rpts = find_nbr_of_rpts({k: cap_dct[k] for k in cmb})
        nbr_of_cells = find_nbr_of_cells({k: cap_dct[k] for k in cmb})
        prob_df = pd.DataFrame(index=[f'rpt_{m}' for m in range(2, nbr_of_rpts+1)],
                               columns=[f'{n}_cells' for n in range(2, nbr_of_cells + 1)])
        ref_prob_df = pd.DataFrame(index=[f'rpt_{m}' for m in range(2, nbr_of_rpts + 1)],
                               columns=[f'{n}_cells' for n in range(2, nbr_of_cells + 1)])
        for m in range(2, nbr_of_rpts + 1):
            for n, c_cmb in dct_of_cell_combinations.items():
                d_set_ref1 = retrieve_samples(cap_dct[cmb[0]], m, ref_set_1)
                d_set_ref2 = retrieve_samples(cap_dct[cmb[1]], m, ref_set_2)
                ref_prob = perform_f_oneway(d_set_ref1, d_set_ref2)
                prob_set = [perform_f_oneway(retrieve_samples(cap_dct[cmb[0]], m, c1[0]),
                                            retrieve_samples(cap_dct[cmb[1]], m, c1[1])) for c1 in c_cmb]
                anova_res_set = [pass_fail_anova(p) for p in prob_set]
                stat_test_res = np.array([pass_fail_test(t_res, pass_fail_anova(ref_prob)) for t_res in anova_res_set])
                # print(f'Amount of failed is {1 - sum(stat_test_res)/len(stat_test_res):.2f} for comb {cmb} '
                #       f'with {n} cells and {m} datapoints')
                prob_df.loc[f'rpt_{m}', f'{n}_cells'] = 1 - sum(stat_test_res)/len(stat_test_res)
                ref_prob_df.loc[f'rpt_{m}', f'{n}_cells'] = ref_prob
        prob_df.dropna(inplace=True)
        dct_of_likelihoods[cmb] = prob_df
        dct_of_ref_probs[cmb] = ref_prob_df
    return dct_of_likelihoods, dct_of_ref_probs


def gen_synth_data(df, outlier=False, n_fault=1):
    m, n = df.filter(like='cap').shape
    rp = [f'rpt_{i}' for i in range(1, m + 1)]
    if outlier == 'Random':
        synth_data = {rpt: generate_data_w_outliers(n=n, mu=df.loc[rpt, 'Q_mean'], sig=df.loc[rpt, 'Q_std'])
                      for rpt in rp}
    elif outlier == 'Cell_Fault':
        synth_data = {rpt: generate_data_w_cell_fault(n=n, mu=df.loc[rpt, 'Q_mean'],
                                                      sig=df.loc[rpt, 'Q_std'], rpt_str=rpt, n_fault=n_fault)
                      for rpt in rp}
    elif outlier == 'Offset':
        synth_data = {rpt: generate_data_w_random_offset(n=n, mu=df.loc[rpt, 'Q_mean'], sig=df.loc[rpt, 'Q_std'],
                                                         rpt_str=rpt, n_offset=n_fault)
                      for rpt in rp}
    else:
        synth_data = {rpt: np.random.default_rng(fixed_seed).normal(loc=df.loc[rpt, 'Q_mean'],
                                                                    scale=df.loc[rpt, 'Q_std'],
                                                                    size=n)
                      for rpt in rp}
    synth_df = pd.DataFrame.from_dict(synth_data, orient='index', columns=[f'cap_{i}' for i in range(1, n + 1)])
    return synth_df


def generate_data_w_random_offset(n, mu, sig, rpt_str, n_offset=1):
    n1 = n - n_offset
    if rpt_str == 'rpt_1':
        mean_scale = 1
    else:
        mean_scale = 0.95
    ds1 = np.random.default_rng(fixed_seed).normal(loc=mu, scale=sig, size=n1)
    ds2 = np.random.default_rng(fixed_seed).normal(loc=mu * mean_scale, scale=sig, size=n_offset)
    ds = np.concatenate((ds1, ds2))
    # np.random.shuffle(ds)
    return ds


def generate_data_w_cell_fault(n, mu, sig, rpt_str, n_fault=1):
    fault_offset = 0.01
    rpt_nbr = int(rpt_str.split('_')[1])
    n1 = n - n_fault
    ds1 = np.random.default_rng(fixed_seed).normal(loc=mu, scale=sig, size=n1)
    ds_fault = np.random.default_rng(fixed_seed).normal(loc=mu - (rpt_nbr - 1)*fault_offset, scale=sig, size=n_fault)
    ds = np.concatenate((ds1, ds_fault))
    return ds


def generate_data_w_outliers(n, mu, sig):
    n1 = int(np.floor(0.75*n))
    n2 = n - n1
    ds1 = np.random.default_rng(fixed_seed).normal(loc=mu, scale=sig, size=n1)
    ds2 = np.random.default_rng(fixed_seed).normal(loc=mu, scale=sig * 10, size=n2)
    ds = np.concatenate((ds1, ds2))
    np.random.shuffle(ds)
    return ds


def plot_and_fit_norm(d_set, ax, xloc=0.1, yloc=0.85, print_fit=True):
    loc, scale = norm.fit(d_set)
    x_range = np.linspace(d_set.min() - scale, d_set.max() + scale, 1000)
    ax.hist(d_set, density=True, bins='auto', histtype='bar',
            alpha=0.5, edgecolor='black')
    ax.plot(x_range, norm.pdf(x_range, loc=loc, scale=scale), linewidth=2)
    if print_fit:
        ax.text(xloc, yloc, fr'$\mu={loc:.2f}$' + '\n' + fr'$\sigma={scale:.2f}$',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=lbl_font,
                bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 6})
    return ax


def plot_data_w_errorbar(df, ax):
    mu_Q = df.filter(like='cap').mean(axis=1)
    sigma_Q = df.filter(like='cap').std(axis=1)
    FCE = [40 * (i) for i in range(df.shape[0])]
    ax.errorbar(x=FCE, y=mu_Q * 100, yerr=sigma_Q * 100, elinewidth=1.5,
                   marker='s',
                   capsize=6)
    ax.fill_between(FCE, (mu_Q - sigma_Q) * 100, (mu_Q + sigma_Q) * 100, alpha=0.2)
    ax.legend(loc='upper right')
    ax.set_xlabel('Number of Full Cycle Equivalents [FCE]', fontdict=lbl_font)
    ax.set_ylabel('Percentage of Capacity Retained [%]', fontdict=lbl_font)
    ax.set_ylim(70, 102)
    ax.grid(color='gray', alpha=0.7)
    return ax


def plot_all_cells(df, ax, plot_mean=False):
    plt.style.use('kelly_colors')
    FCE = [40 * (i) for i in range(df.shape[0])]
    ax.plot(FCE, df.filter(like='cap'), marker='.', linestyle='dashed')
    ax.legend(df.columns, ncol=2, prop={'size': 6})
    if plot_mean:
        mu_q = df.filter(like='cap').mean(axis=1)
        ax.plot(FCE, mu_q, linestyle='solid', marker='^', label=r'$\mu_{Q}$')
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax = fix_tick_dist(ax)
    return ax


def visualise_data(data_set, ax, print_fit=True, print_label=True):
    # fig_norm, ax = plt.subplots(1, 1)
    if isinstance(data_set, dict):
        yloc = 0.85
        for k, d_set in data_set.items():
            ax = plot_and_fit_norm(d_set, ax, yloc=yloc, print_fit=print_fit)
            yloc += -0.2
        lgd = [k for df, k in data_set.items()]
    else:
        ax = plot_and_fit_norm(data_set, ax, print_fit=print_fit)
    ax.grid(alpha=.2)
    if print_label:
        ax.set_xlabel('Relative capacity', fontdict=lbl_font)
        ax.set_ylabel('Normalised occurrence', fontdict=lbl_font)
    return ax


def clean_for_plot(rpt_name):
    if isinstance(rpt_name, str):
        return rpt_name.upper().replace('_', ' ')
    elif isinstance(rpt_name, dict):
        return [r.upper().replace('_', ' ') for r in rpt_name.keys()]


def fix_title(k):
    return f'Test {k[0]} with Test {k[1]}'


def export_data_for_r(rpt_data, exp_col='cap_relative'):
    op_dir = r"Z:\StatisticalTest\processed_data"
    for n in range(1, 8):
        rpt_ = f'rpt_{n}'
        export_df = pd.DataFrame()
        for k, rpt in rpt_data.summary_dict.items():
            u_name = k
            t_name = rpt.test_name
            test_idx = k.split("_")[-1]
            cap = rpt.data.loc[rpt_, exp_col]
            data_names = ['unique_name', 'Rpt_cap', 'test_name', 'test_idx']
            exp_data = [u_name, cap, t_name, test_idx]
            if export_df.empty:
                export_df = pd.DataFrame(exp_data, index=['unique_name', 'Rpt_cap', 'test_name', 'test_idx'],
                                         columns=[1])
                export_df = export_df.T
            else:
                tmp_df = pd.DataFrame(exp_data, index=['unique_name', 'Rpt_cap', 'test_name', 'test_idx'], columns=[1])
                export_df = pd.concat([export_df, tmp_df.T], ignore_index=True)
        export_df.to_excel(os.path.join(op_dir, f"{exp_col}_{rpt_}.xlsx"))
    return None


def fix_tick_dist(ax, n=5):
    min_tick = min(ax.get_ylim())
    if min_tick > 0.9:
        min_pt = 0.9
    elif min_tick > 0.8:
        min_pt = 0.8
    elif min_tick > 0.7:
        min_pt = 0.7
    elif min_tick > 0.6:
        min_pt = 0.6
    else:
        min_pt = 0.5
    ax.set_yticks(np.linspace(min_pt, 1, n))
    return ax


def _output_fig(fig_obj, f_name):
    fig_op_dir = r"Z:\StatisticalTest\figures_new_test_names"
    for file_type in ['.pdf', '.png']:
        fig_obj.savefig(os.path.join(fig_op_dir, f'{f_name}{file_type}'), dpi=400)
    return None


def calculate_eol_stat(cap_dict):
    eol_dct = {}
    for k, df in cap_dict.items():
        combs = extract_combinations(df)
        comb_dct = {}
        N, _ = df.shape
        for n, tpls in combs.items():
            comb_dct[f'{n}_test'] = {}
            x = np.tile(np.arange(0, N*50, 50), n)
            for m, tpl in enumerate(tpls):
                y = df.loc[:, tpl].values.flatten(order='F')
                comb_dct[f'{n}_test'][m] = edf(x, y)
        eol_dct[k] = comb_dct
    return eol_dct


def fail_prob_plot(ax, df, font_sz=9):
    x = [int(col.split("_")[0]) for col in df.columns]
    for rp in df.index:
        ax.plot(x, df.loc[rp, :], linestyle='dashed', marker='*', label=clean_for_plot(rp))
    ax.set_xlabel('Number of cells included', fontsize=font_sz)
    ax.set_ylabel('Probability of false prediction', fontsize=font_sz)
    ax.grid(color='gray', alpha=0.6)
    ax.set_xticks(x)
    ax.legend(ncols=2, fontsize=font_sz-2)
    return ax


def make_all_cell_subfig(dct, n_rows, n_cols, fig_title=''):
    x_width = 4
    ar = 3/4
    y_width = ar * x_width
    fig = plt.figure(figsize=(x_width*n_cols, y_width*n_rows))
    plt_num = 1
    for k, df in dct.items():
        ax = fig.add_subplot(n_rows, n_cols, plt_num)
        ax = plot_all_cells(df, ax)
        plt_num += 1
    fig.supxlabel('Full cycle equivalents')
    fig.supylabel('Normalised Capacity retention')
    fig.suptitle(fig_title, fontsize=14)
    return fig


def box_plot_on_data_dct(data_dct,
                         n_rows=2,
                         aspect_rat=0.75,
                         x_width=6,
                         f_name=''):
    y_width = aspect_rat * x_width
    n_cols = int(len(data_dct)/n_rows)
    fig_box = plt.figure(figsize=(n_cols * x_width, n_rows * y_width))
    plt_num = 1
    for k, d_ in data_dct.items():
        ax = fig_box.add_subplot(n_rows, n_cols, plt_num)
        tmp = [s for k, s in d_.items()]
        ax.boxplot(tmp)
        ax.set_xticklabels(d_.keys())
        ax.set_title(clean_for_plot(k))
        ax = fix_tick_dist(ax, 5)
        plt_num += 1
    fig_box.supylabel('Normalised Capacity', fontsize=20)
    fig_box.supxlabel('Test set', fontsize=20)
    fig_box.axes[5].set_ylim(top=1)
    for n in range(n_rows):
        for i in range(3*n, 3*n + 2):
            fig_box.axes[i].sharey(fig_box.axes[3*n + 2])
    if f_name:
        _output_fig(fig_box, f_name)
    return fig_box


def sort_dict(dct):
    my_keys = list(dct.keys())
    my_keys.sort()
    return {k: dct[k] for k in my_keys}


def main():
    data_set_location = r"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\cycling_data"
    fig_op_dir = r"Z:\StatisticalTest\figures_new_test_names"
    global fixed_seed
    fixed_seed = 274589933423713474590742301910344420568
    if not os.path.isdir(fig_op_dir):
        os.mkdir(fig_op_dir)
    my_data = OrganiseRpts(data_set_location, proj='stat')
    aspect_rat = 3 / 4
    x_width = 6
    y_width = x_width * aspect_rat

    test_names = set([f'Test{nm_list[0]}_{nm_list[1]}' for nm_list in combinations(['1', '2', '1', '2'], 2)])
    name_keys = dict(zip(['Test1_1', 'Test1_2', 'Test2_1', 'Test2_2'], ['A', 'B', 'C', 'D']))
    test_subsets = {name_keys[nm]: {k: val.data for k, val in my_data.summary_dict.items() if nm in k} for nm in test_names}
    cap_data = make_data_set(test_subsets)
    cap_data = sort_dict(cap_data)
    ppe_dict = {k: ppe(df) for k, df in cap_data.items()}
    outlier_types = ['Random', 'Cell_Fault', 'Offset', 'Default']
    synth_cap_dct = {otlr: {k: gen_synth_data(df, outlier=otlr) for k, df in cap_data.items()} for otlr in outlier_types}
    # p_fail_outliers = {otlr: perform_full_anova(synth_data) for otlr, synth_data in synth_cap_dct.items()}
    # synth_cap_data = {k: gen_synth_data(df) for k, df in cap_data.items()}
    p_failure_dct, ref_prob_dct = perform_full_anova(cap_dct=cap_data)

    fce_fig, fax = plt.subplots(1, 1)
    test_comp = {tst: {k: df.filter(like='cap').loc[k, :] for k in df.index
                       if int(k.split("_")[1]) % 2 == 0} for tst, df in cap_data.items()}
    rpt_comp = {f'rpt_{k}': {nm: cap_data[nm].filter(like='cap').loc[f'rpt_{k}', :] for nm in cap_data.keys()}
                for k in range(2, 8)}
    close_value_sets = {
        f'rpt_{k}': {nm: cap_data[nm].filter(like='cap').loc[f'rpt_{k}', :] for nm in cap_data.keys() if ('C' in nm) or ('B' in nm)}
        for k in range(2, 11)}
    extreme_set = {
        f'rpt_{k}': {nm: cap_data[nm].filter(like='cap').loc[f'rpt_{k}', :] for nm in cap_data.keys() if ('A' in nm) or ('D' in nm)}
        for k in range(2, 8)}
    visualise_data(test_comp['A'], fax)
    visualise_data(rpt_comp['rpt_3'], fax)

    n_rows, n_cols = 4, 1
    fig_t = plt.figure(figsize=(n_cols * x_width, n_rows * y_width))
    plt_num = 1
    for k, d_ in test_comp.items():
        ax = fig_t.add_subplot(n_rows, n_cols, plt_num)
        ax = visualise_data(d_, ax, print_fit=False, print_label=False)
        plt_num += 1
    [fig_t.get_axes()[n].legend(clean_for_plot(test_comp[dct])) for n, dct in enumerate(test_comp)]
    [fig_t.get_axes()[n].set_title(k, fontsize=9) for n, k in enumerate(test_comp)]
    fig_t.supxlabel('Normalised Capacity', fontdict=lbl_font)
    fig_t.supylabel('Normalised Occurrence', fontdict=lbl_font)
    fig_t.subplots_adjust(left=0.15)
    fig_t.savefig(os.path.join(fig_op_dir, f'cap_progression_test_portrait_{my_style}.png'), dpi=400)

    n_rows, n_cols = 3, 2
    fig_rpt = plt.figure(figsize=(n_cols * x_width, n_rows * y_width))
    plt_num = 1
    for k, d_ in rpt_comp.items():
        ax = fig_rpt.add_subplot(n_rows, n_cols, plt_num)
        ax = visualise_data(d_, ax, print_fit=False, print_label=False)
        plt_num += 1
    [fig_rpt.get_axes()[n].legend(rpt_comp[dct].keys()) for n, dct in enumerate(rpt_comp)]
    [fig_rpt.get_axes()[n].set_title(clean_for_plot(k)) for n, k in enumerate(rpt_comp)]
    fig_rpt.supxlabel('Normalised Capacity', fontsize=20)
    fig_rpt.supylabel('Normalised Occurrence', fontsize=20)
    _output_fig(fig_rpt, f'cap_progression_rpt_{my_style}_portrait')

    fig_box = box_plot_on_data_dct(rpt_comp, f_name='box_plot_cap_decay_landscape')
    cls_data_box = box_plot_on_data_dct(close_value_sets, n_rows=2, x_width=4)
    cls_data_box.tight_layout()
    xtrm_data_box = box_plot_on_data_dct(extreme_set, n_rows=2, x_width=4.5)
    xtrm_data_box.tight_layout()
    _output_fig(cls_data_box, 'box_plot_close_data')
    _output_fig(xtrm_data_box, 'box_plot_xtrm_data')

    fig, qe_ax = plt.subplots(1, 1, figsize=(x_width, aspect_rat*x_width))
    for key, df in cap_data.items():
        qe_ax.errorbar(x=df.FCE, y=df.Q_mean * 100, yerr=df.Q_std * 100, elinewidth=1.5,
                       marker='s',
                       capsize=6,
                       label=f'{key}')
        qe_ax.fill_between(df.FCE, df.Q_mean * 100 - df.Q_std * 100, df.Q_mean * 100 + df.Q_std * 100, alpha=0.2)
    qe_ax.legend(loc='upper right')
    qe_ax.set_xlabel('Number of Full Cycle Equivalents [FCE]', fontdict=lbl_font)
    qe_ax.set_ylabel('Percentage of Capacity Retained [%]', fontdict=lbl_font)
    qe_ax.set_ylim(70, 102)
    qe_ax.grid(color='gray', alpha=0.7)
    _output_fig(fig, 'capacity_decay')

    x_width = 4
    y_width = aspect_rat * x_width
    n_cols = 2
    n_rows = 2
    full_data_sub_plot = plt.figure(figsize=(n_cols * x_width, n_rows * y_width))
    plt_num = 1
    plot_labels = ['(a) Test A', '(b) Test B', '(c) Test C', '(d) Test D']
    fig_labels = dict(zip(cap_data.keys(), plot_labels))
    for k, df in cap_data.items():
        tmp_ax = full_data_sub_plot.add_subplot(n_rows, n_cols, plt_num)
        tmp_ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        tmp_ax = plot_all_cells(df, tmp_ax, plot_mean=True)
        _output_fig(fig, f'full_test_data_{k}')
        tmp_ax.text(0.5, -0.165, fig_labels[k], transform=tmp_ax.transAxes, fontsize=10, horizontalalignment='center')
        plt_num += 1
    full_data_sub_plot.subplots_adjust(bottom=0.12)
    full_data_sub_plot.subplots_adjust(top=0.95)
    full_data_sub_plot.subplots_adjust(left=0.1)
    full_data_sub_plot.subplots_adjust(hspace=0.25)
    full_data_sub_plot.supxlabel('Full Cycle Equivalents')
    full_data_sub_plot.supylabel('Normalised Capacity Retention')
    _output_fig(full_data_sub_plot, 'full_test_data_labeled')

    # Output unique figures
    x_width = 6
    y_width = aspect_rat * x_width
    for k, df in cap_data.items():
        fig, tmp_ax = plt.subplots(1, 1, figsize=(x_width, y_width))
        tmp_ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        tmp_ax = plot_all_cells(df, tmp_ax, plot_mean=True)
        tmp_ax.set_xlabel('Full Cycle Equivalents', fontdict=lbl_font)
        tmp_ax.set_ylabel('Normalised Capacity Retention', fontdict=lbl_font)
        _output_fig(fig, f'full_test_data_{k}')


    sig_fig, saxs = plt.subplots(ncols=n_cols, nrows=n_rows,
                                 figsize=(n_cols * x_width, n_rows * y_width))
    for z in zip(ppe_dict.values(), saxs.ravel()):
        z[0].plot_var_mu(z[1], case_to_plot='sig')

    x_width = 3.5
    y_width = aspect_rat * x_width
    char_list = list(map(chr, range(ord('a'), ord('z') + 1)))
    for key in p_failure_dct.keys():
        n_rows, n_cols = 3, 2
        plt_num = 1

        sub_fig_p = plt.figure(figsize=(n_cols * x_width, n_rows * y_width))
        for k, p_df in p_failure_dct.items():
            # p_fail_, pax = plt.subplots(1, 1)
            # pax = fail_prob_plot(pax, p_df)
            tmp_ax = sub_fig_p.add_subplot(n_rows, n_cols, plt_num)
            tmp_ax = fail_prob_plot(tmp_ax, p_df)
            tmp_ax.set_ylabel('')
            tmp_ax.set_xlabel('')
            plt_lbl = f'({char_list[plt_num - 1]}) {fix_title(k)}'
            plt_num += 1
            # pax.set_title(fix_title(k), fontsize=11)
            # tmp_ax.set_title(fix_title(k), fontsize=11)

            tmp_ax.set_ylim((0, 1))
            tmp_ax.text(0.5, -0.2, plt_lbl, transform=tmp_ax.transAxes, fontsize=13,
                        horizontalalignment='center')
            # p_fail_.savefig(os.path.join(fig_op_dir, f'{k[0]}_and_{k[1]}.png'), dpi=400)
        sub_fig_p = set_font_sizes(sub_fig_p, 12)
        # sub_fig_p.suptitle('Real Data', fontsize=15)
        sub_fig_p.supxlabel('Number of replicates used', fontsize=14)
        sub_fig_p.supylabel('Probability of false prediction', fontsize=14)
        sub_fig_p.tight_layout()
    _output_fig(sub_fig_p, 'p_fail_subplot_landscape_updated_names')

    plt.close('all')
    return None


if __name__ == '__main__':
    main()
