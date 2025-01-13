from test_data_analysis.pec_smartcell_data_handler import PecSmartCellDataHandler
from check_current_os import get_base_path_batt_lab_data
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(['widthsixinches', 'ml_colors'])
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times New Roman",
        "text.latex.preamble": r'\usepackage{siunitx}'
    })


def extract_conditions(test_condition):
    pattern = r'(?P<Duty>\d+(\.\d+)?)duty (?P<C_rate>\d+(\.\d+)?)C (?P<Frequency>\d+(\.\d+)?)Hz'
    match = re.search(pattern, test_condition)
    if match:
        return match.groupdict()
    else:
        return {'Duty': 100, 'C_rate': 1, 'Frequency': 0}


BASE_PATH = get_base_path_batt_lab_data()
data_dir = os.path.join(BASE_PATH, 'pulse_chrg_test/high_frequency_testing/PEC_export')
handler = PecSmartCellDataHandler(data_dir)

handler.merge_test_condition_data()
mean_temps = handler.calculate_mean_temperature()

cap_90_dict = {}
for cell, pscd in handler.merged_pscd.items():
    pscd.fit_degradation_function()
    condition_dict = extract_conditions(pscd.formatted_metadata['TEST_CONDITION'])
    cap_90_dict[cell] = [pscd.find_fce_at_given_q(0.9), mean_temps.loc[cell, 'Mean Temperature'], *condition_dict.values()]
cap_90_df = pd.DataFrame.from_dict(cap_90_dict, orient='index')
cap_90_df.columns = ['fce_at_q', 'mean_temp', *condition_dict.keys()]
cap_90_df['Duty'] = cap_90_df['Duty'].astype(float)
cap_90_df['C_rate'] = cap_90_df['C_rate'].astype(float)
cap_90_df['Frequency'] = cap_90_df['Frequency'].astype(float)
ax_f = cap_90_df.plot.scatter(x='Frequency', y='fce_at_q', c='Duty')
ax_T = cap_90_df.plot.scatter(x='mean_temp', y='fce_at_q', c='Duty')
cap_90_corr = cap_90_df.drop('C_rate', axis=1).corr()

cap_90_ici = {cell: pscd.filter_ici_on_cap([0.9]) for cell, pscd in handler.merged_pscd.items()}

fig, ax = plt.subplots(1, 1)
all_lines = []
for cell, pscd in handler.merged_pscd.items():
    pscd.style['label'].replace('%', '\%')
    line, = ax.plot(pscd.rpt_obj.rpt_summary.fce, pscd.rpt_obj.rpt_summary.cap_normalised, **pscd.style)
    all_lines.append((line, pscd.style['label']))
unique_lines = {label: line for line, label in all_lines}
ax.legend(unique_lines.values(), unique_lines.keys())
ax.set_xlabel('Full Cycle Equivalents')
ax.set_ylabel('Capacity Retention')

fig2, ax_err = plt.subplots(1, 1)
fig3, ax_avg = plt.subplots(1, 1)
for cond, dct in handler.merged_condition_data.items():
    dct['style']['label'].replace('%', '\%')
    ax_err.errorbar(dct['merged_df'].fce, dct['merged_df'].mean_capacity, yerr=dct['merged_df'].sigma_capacity,
                    capsize=2, **dct['style'])
    ax_avg.plot(dct['merged_df'].fce, dct['merged_df'].mean_capacity, **dct['style'])
ax_avg.legend()
ax_err.legend()
