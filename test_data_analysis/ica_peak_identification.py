from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from test_data_analysis.tesla_half_cell import data_reader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from test_data_analysis.ica_analysis import gaussianfilterint
from scipy.signal import find_peaks


def find_nearest_cap(df, volt):
    return df.iloc[(df['volt'] - volt).abs().argsort()[:1]]['cap'].iloc[0]


pe_file = r"Z:\Provning\Halvcellsdata\20200910-AJS-PH0S06-Tes-C10-BB5.txt"
ne_file = r"Z:\Provning\Halvcellsdata\20200910-AJS-NH0S05-Tes-C10-BB2.txt"
ref_data = r"\\sol.ita.chalmers.se\groups\batt_lab_data\Ica_files\2_8\2_8_ica_dump_rpt_1.pkl"

# Read in files for each electrode test
df_pe = data_reader(pe_file)
df_ne = data_reader(ne_file)
df_ref = pd.read_pickle(ref_data)

# Find the appropriate step limits
part_df_pe = df_pe[df_pe['step'] == 7]
part_df_ne = df_ne[df_ne['step'] == 7]

# Negative sign in integral to compensate discharge current
part_df_ne.loc[:, 'cap_cell'] = cumtrapz(-part_df_ne['curr'], part_df_ne['time'], initial=0) / 3.6
part_df_pe.loc[:, 'cap_cell'] = cumtrapz(part_df_pe['curr'], part_df_pe['time'], initial=0) / 3.6
part_df_ne.loc[:, 'sol'] = part_df_ne['cap'] / part_df_ne.cap.max()
part_df_pe.loc[:, 'sol'] = 1 - part_df_pe['cap'] / part_df_pe.cap.max()

r_hc = 1.5 / 2
A_neg_test = np.pi * (r_hc) ** 2  # cm^2
A_pos_test = np.pi * (r_hc) ** 2  # cm^2

x_pe = part_df_pe.loc[:, 'cap_cell'] / A_pos_test  # mAh / cm^2
y_pe = part_df_pe.loc[:, 'pot']
x_ne = part_df_ne.loc[:, 'cap_cell'] / A_neg_test  # mAh / cm^2
y_ne = part_df_ne.loc[:, 'pot']

df_ch = df_ref[df_ref.curr > 0]
x_ref = df_ch.cap / 1000
y_ref = df_ch.volt

fit_fun_pe = lambda x, A_pe, s_pe: interp1d(A_pe * x_pe + s_pe, y_pe, fill_value='extrapolate')(x)
fit_fun_ne = lambda x, A_ne, s_ne: interp1d(A_ne * x_ne + s_ne, y_ne, fill_value='extrapolate')(x)
fit_fun_cell = lambda x, A_pe, s_pe, A_ne, s_ne: fit_fun_pe(x, A_pe, s_pe) - fit_fun_ne(x, A_ne, s_ne)

p_init = [1.2, -0.003, 1.1, -0.005]
lo_b = (0.5, -0.25, 0.5, -0.25)
hi_b = (1.5, 0, 1.5, 0)

popt, pcov = curve_fit(fit_fun_cell, x_ref, y_ref, p0=p_init, bounds=(lo_b, hi_b))

ica = np.gradient(x_ref, y_ref)
ica_f = gaussianfilterint(x_ref, ica)
ica_peaks, _ = find_peaks(ica_f)
plt.figure()
plt.plot(y_ref, ica_f, linestyle='dashed', color='black')
plt.scatter(y_ref.array[ica_peaks], ica_f[ica_peaks], color='black', s=40)

plt.figure()
plt.plot(x_ref, y_ref)
plt.plot(x_ref, fit_fun_pe(x_ref, *popt[:2]))
plt.plot(x_ref, fit_fun_ne(x_ref, *popt[2:]))
[plt.axvline(find_nearest_cap(df_ch, v) / 1000, linestyle='dashed', color='blue') for v in y_ref.array[ica_peaks]]
