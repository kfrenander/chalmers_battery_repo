import numpy as np
from matplotlib.ticker import FormatStrFormatter
from misc_classes.EndOfLifeProcessing import EndOfLifeProcessing
import matplotlib.pyplot as plt
import os
import pickle
from pd2ppt import df_to_powerpoint, df_to_table
from pptx import Presentation
from scipy.stats import norm
from matplotlib import font_manager
fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],
    'weight' : 'normal', 'size' : 12}


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
test_cases = ["Test1_1", "Test1_2", "Test2_1", "Test2_2"]
test_names = ["Test A", "Test B", "Test C", "Test D"]
name_dict = dict(zip(test_cases, test_names))
for tc in test_cases:
    d_file = rf"\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\processed_data\t_to_eol_{tc}.pkl"
    with open(d_file, 'rb') as h:
        dataset = pickle.load(h)
    eol_pp = EndOfLifeProcessing(dataset)

    # OUTPUT FIGURES FOR HISTOGRAMS AND NORMAL FITS FOR ALL EOL ESTIMATIONS
    update_figs = 0
    shade_area = 1
    fig_dir = os.path.join(r"Z:\StatisticalTest\figures_new_test_names", f"{name_dict[tc]}")
    if update_figs:
        print('Updating figures')
        if not os.path.isdir(fig_dir):
            os.mkdir(fig_dir)
        for k in eol_pp.dataset.keys():
            fig, ax = plt.subplots(1, 1)
            fig_name = os.path.join(fig_dir, f'norm_and_histogram_{k}.pdf')
            ax = eol_pp._plot_histogram(ax, normal_bool=True, n_cases=k, col_oe=1)
            ax = eol_pp._plot_norm_distribution(ax, n_cases=k)
            ax.set_xticks(ax.get_xticks(), labels=[f'{x:.0f}' for x in ax.get_xticks()])
            ax.set_yticks(ax.get_yticks(), labels=[f'{y:.3f}' for y in ax.get_yticks()])
            ax.text(0.05, 0.9, f'{name_dict[tc]}: {k.split("_")[0]} cells',
                    fontsize=12,
                    horizontalalignment='left',
                    verticalalignment='center',
                    transform=ax.transAxes)
            ax.set_xlabel('FCE to EOL [-]', fontsize=14)
            ax.set_ylabel('Frequency', fontsize=14)
            ax.lines[0].set_label(None)
            ax.legend(loc='upper right')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            if shade_area:
                mu, sig = eol_pp.norm_dist_params[k]
                x = np.linspace(mu - 3*sig, mu + 3*sig, 1000)
                y = norm.pdf(x, loc=mu, scale=sig)
                rsk = 1 - norm.cdf(1.05 * eol_pp.ref_value, loc=mu, scale=sig)
                ax.text(0.7, 0.5, f'$P(OE)={100*rsk:.2f}\%$',
                        fontsize=12,
                        horizontalalignment='left',
                        verticalalignment='center',
                        transform=ax.transAxes)
                ax.fill_between(x[x > 1.05*eol_pp.ref_value], y[x > 1.05*eol_pp.ref_value], 0,
                                color='orange', alpha=0.5)
                fig_name = os.path.join(fig_dir, f'norm_and_histogram_{k}_with_shade_with_cdf.pdf')
            fig.savefig(fig_name, dpi=300)
        plt.close('all')

    # OUTPUT RISK BASED ON NORMAL DISTRIBUTION TO TABLE IN POWERPOINT
    output_ppt_table = 0
    if output_ppt_table:
        df = eol_pp.df_oe_risk_norm
        df.fillna(0, inplace=True)
        df.columns = [c.replace('_test', '') for c in df.columns]
        df = df * 100
        df.insert(0, 'OE', df.index)
        m, n = df.shape
        col_format = ['.2' for k in range(n)]
        df_to_powerpoint(os.path.join(fig_dir, f'risk_table_{tc}.pptx'),
                         df.iloc[:, :-1],
                         col_formatters=col_format,
                         width=19,
                         height=6,
                         left=3,
                         top=3,
                         name='risk_table')
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        pr_shp = df_to_table(slide, df, col_formatters=col_format)
