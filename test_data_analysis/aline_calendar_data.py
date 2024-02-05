from rpt_data_analysis.ReadRptClass import OrganiseRpts
from backend_fix import fix_mpl_backend


fix_mpl_backend()
cal_data_dir = r"\\sol.ita.chalmers.se\groups\batt_lab_data\ALINE_data\CalendarData"
data_set = OrganiseRpts(cal_data_dir, proj='aline')

all_tests = [k.split("_")[0] for k in data_set.summary_dict]

cal_fig = data_set.plot_rpt_data(all_tests, x_mode='time')
cal_fig.gca().set_ylim(0.9, 1.03)

ica_fig = data_set.plot_ica(all_tests, rpt_num=['rpt_1'])
