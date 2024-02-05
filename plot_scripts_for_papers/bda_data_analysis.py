from rpt_data_analysis.ReadRptClass import OrganiseRpts

bda_comp = r"\\sol.ita.chalmers.se\groups\batt_lab_data\20210816"
bda_orig = r"\\sol.ita.chalmers.se\groups\batt_lab_data\20200923_pkl"
bda_comp_data = OrganiseRpts(bda_comp, proj='bda_comp')
bda_orig_data = OrganiseRpts(bda_orig)
