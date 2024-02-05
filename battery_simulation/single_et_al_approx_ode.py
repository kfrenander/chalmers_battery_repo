import numpy as np
import pandas as pd
import scipy.interpolate
from scipy.integrate import odeint
import scipy.optimize as scopt
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.constants
import re
import os
import datetime as dt
import pandas as pd
from PythonScripts.backend_fix import fix_mpl_backend
plt.rcParams['axes.grid'] = True
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['axes.titlesize'] = 18
#plt.style.use('chalmers_kf')
# mpl.rcParams['savefig.dpi'] = 800


def dQdt_li_diff_fun(Q, t, c_li_max, D_li_sei, A_el, V, U, T, mu_li_sei):
    F = scipy.constants.value('Faraday constant')
    R = scipy.constants.value('molar gas constant')
    pre_factor = A_el**2 * F**2 * D_li_sei * c_li_max / (V * Q) * np.exp(-mu_li_sei/(R*T))
    dQdt = pre_factor * np.exp(-(F*U)/(R*T))
    return dQdt


def dQdt_li_diff_lumped(Q, t, D_li_sei, A_el, V, c_li0, U, T):
    F = scipy.constants.value('Faraday constant')
    R = scipy.constants.value('molar gas constant')
    dQdt = A_el**2 * F**2 * D_li_sei * c_li0 / (V * Q) * np.exp(-F * U / (R * T))
    return dQdt


def dQdt_solv_diff_fun(Q, t, c_solv, D_ec_sei, A_el, V):
    F = scipy.constants.value('Faraday constant')
    dQdt = F**2 * A_el**2 * D_ec_sei * c_solv / (V * Q)
    return dQdt


def find_neg_pot(test_key):
    soc_pattern = r'soc\d+'
    neg_max = 0.9
    neg_min = 0.063
    soc_lvl = int(re.search(r'\d+', re.search(soc_pattern, test_key).group(0)).group(0))  / 100
    neg_soc = (neg_max - neg_min) * soc_lvl + neg_min
    return u_look_up(neg_soc)


def fit_c_li0_fun(t, c_li0, U, T):
    # Set up all constants that are not being tuned.
    F = scipy.constants.value('Faraday constant')
    r_part = 12.5e-6  # m
    eps_s_neg = 0.525  # -
    A_spec = 3 * eps_s_neg / r_part  # m2 / m3
    L_neg = 105e-6  # m
    D_li_sei = 1.6e-14

    # Derived constants
    A_cell = 0.029 * n_sheets  # m2
    V_neg_el = L_neg * A_cell
    A_surf = V_neg_el * A_spec
    Q0 = A_surf * F * L0 / V_sei
    A_el = A_spec * V_neg_el
    return odeint(dQdt_li_diff_lumped, Q0, t, args=(D_li_sei, A_el, V_sei, c_li0, U, T))[:, 0]


# fix matplotlib backend
fix_mpl_backend()

# setup for output
op_dir = r"Z:\SEI_models\single_et_al_new_figures"
td = dt.datetime.today()

if not os.path.exists(op_dir):
    os.makedirs(op_dir)

# Fitting directory
fit_dir = r"Z:\SEI_models\LG_tuning_data"
cal_data = {}
for root, dir, files in os.walk(fit_dir):
    for file in files:
        cal_data[file.strip('.xlsx')] = pd.read_excel(os.path.join(root, file), engine='openpyxl')

# Set up physical parameters of model
F = scipy.constants.value('Faraday constant')
R = scipy.constants.value('molar gas constant')
NA = scipy.constants.value('Avogadro constant')
ev_J_conv = scipy.constants.physical_constants['electron volt-joule relationship'][0]
eq_pot_file = r"C:\battery-model\PythonScripts\battery_simulation\c6_eq_pot.txt"
u_look_up = scipy.interpolate.interp1d(np.loadtxt(eq_pot_file)[:, 0], np.loadtxt(eq_pot_file)[:, 1])

L0 = 3e-9
M_sei = 0.5 * 0.16 + 0.5 * 0.07  # kg / mol
rho_sei = 1.6e4  # kg / m3
V_sei = M_sei / rho_sei  # m3 / mol

r_part = 12.5e-6  # m
eps_s_neg = 0.525  # -
A_spec = 3 * eps_s_neg / r_part  # m2 / m3
L_neg = 105e-6  # m
n_sheets = 2 * 28  # Due to double sided coating
A_cell = 0.029 * n_sheets  # m2
V_neg_el = L_neg * A_cell
A_surf = V_neg_el * A_spec
Q0 = A_surf * F * L0 / V_sei

print('Initial capacity loss to form SEI layer of {:.2f}nm was {:.1f}Ah'.format(L0*1e9, Q0 / 3600))
L_ah = 3600 * V_sei / (F * A_surf)
print('Thickness of SEI layer when 1Ah is lost is {:.2f}nm'.format(L_ah*1e9))
t = np.linspace(0, 8760*3600*10, 10000)
D_li_sei = 5e-14  # m2/s
c_li_max = 5e9 / NA * 1e6  # Atoms / cm3 * mol / atoms * cm3 / m3
mu_li_sei = 1.121 * ev_J_conv * NA  # J / mol
sol0 = 0.5
U0 = u_look_up(sol0)  # V vs Li/Li+, taken from interpolation data.
T0 = 298  # K
A_el = A_spec * V_neg_el
pre_factor = A_el**2 * F**2 * D_li_sei * c_li_max * np.exp(-mu_li_sei/(R*T0))/(V_sei * Q0)
Q = odeint(dQdt_li_diff_fun, Q0, t, args=(c_li_max, D_li_sei, A_el, V_sei, U0, T0, mu_li_sei))

# Other parameterisation of Li diffusion limited model
c_li0 = 15e-3  # mol / m3
D_li_sei_alt = 1.6e-14
Q_li_diff = odeint(dQdt_li_diff_lumped, Q0, t, args=(D_li_sei_alt, A_el, V_sei, c_li0, U0, T0))

# Setup solution for solvent diffusion limited growth
c_solv = 8e3  # mol / m3 - taken from Kupper et al 2018
D_ec = 2.1e-18  # From thin air
Q_solv_diff = odeint(dQdt_solv_diff_fun, Q0, t, args=(c_solv, D_ec, A_el, V_sei))

# UGLY FIX OF ERRONEOUS DATA POINTS
# Error in measurement at soc40 temp 35degc cell1
err_meas = [cal_data[k] for k in cal_data if 'soc40' in k and '35' in k and 'cell1' in k][0]
err_meas.drop(err_meas.index[0], inplace=True)
err_meas.loc[:, 'Days'] = (err_meas['RPT_date'] -err_meas['RPT_date'].iloc[0]).dt.days
err_meas.loc[:, 'Capacity_%'] = (err_meas['Capacity_Ah'] / err_meas['Capacity_Ah'].iloc[0]) * 100

# Error in measurement at soc70 temp 25degc cell2
err_meas = [cal_data[k] for k in cal_data if 'soc70' in k and '25' in k and 'cell2' in k][0]
err_meas.drop(err_meas.index[0], inplace=True)
err_meas.loc[:, 'Days'] = (err_meas['RPT_date'] -err_meas['RPT_date'].iloc[0]).dt.days
err_meas.loc[:, 'Capacity_%'] = (err_meas['Capacity_Ah'] / err_meas['Capacity_Ah'].iloc[0]) * 100

# Test fitting the solution using only c_li0
T_pattern = r'\d+degc'
c_li0_dict = {}
for test_name in cal_data:
    #if 'torage_soc70__25degc_cell1_1' in test_name:
    u_neg = find_neg_pot(test_name)
    soc_lvl = re.search(r'soc\d+', test_name).group()
    cell_id = re.search(r'cell\d', test_name).group()
    T_degc = int(re.search(r'\d+', re.search(T_pattern, test_name).group(0)).group(0))
    T = scipy.constants.convert_temperature(T_degc, 'Celsius', 'Kelvin')
    fit_t = cal_data[test_name].Days*24*3600
    fit_t_refined = (np.linspace(fit_t.min(), fit_t.max(), 1000))
    fit_q = (cal_data[test_name].Capacity_Ah.iloc[0] - cal_data[test_name].Capacity_Ah) * 3600 + Q0
    popt, pcov = scopt.curve_fit(lambda t, c_li0: fit_c_li0_fun(t, c_li0, u_neg, T), fit_t, fit_q)
    c_li0_dict['Soc_{}_T_{}_{}'.format(soc_lvl, T_degc, cell_id)] = [popt[0], soc_lvl, T_degc, u_neg]
    cli0_fig = plt.figure(figsize=(14, 11))
    plt.plot(fit_t_refined / (8760 * 3600), fit_c_li0_fun(fit_t_refined, popt, u_neg, T) / 3600,
             label='Fitted function, cli0={:.2e}'.format(popt[0]))
    plt.scatter(fit_t / (8760 * 3600), fit_q / 3600, s=200, color='red', label='Raw data')
    plt.title('Fitted sei at U={:.3f} and T={:.0f}'.format(u_neg, T))
    plt.legend()
    plt.xlabel('Time [years]')
    plt.ylabel('Capacity loss [Ah]')
    plt.ylim([0, 8])
    cli0_fig.savefig(os.path.join(op_dir, 'Soc_{}_T_{}_{}.png'.format(soc_lvl, T_degc, cell_id)), dpi=cli0_fig.dpi)
    plt.close(cli0_fig)
df_cli0 = pd.DataFrame.from_dict(c_li0_dict, orient='index', columns=['c_li0', 'soc', 'T', 'U_neg'])
df_cli0['U_neg'] = df_cli0['U_neg'].astype('float')
df_cli0.loc[:, '1/T'] = 1 / (df_cli0['T'] + 273.15)
df_cli0.loc[:, 'ln(cli0)'] = np.log(df_cli0['c_li0'])

df_cli0.plot.scatter('1/T', 'ln(cli0)', c='U_neg', colormap='jet', figsize=(14, 8), s=200)
grps = df_cli0.groupby(by='soc')
lin_fits = {}
T_arr = np.linspace(1 / (df_cli0['T'].max() + 283.15), 1 / (df_cli0['T'].min() + 263.15), 1000)
for g in grps.groups:
    tmp_df = grps.get_group(g)
    my_fit = np.polyfit(tmp_df['1/T'], tmp_df['ln(cli0)'], deg=1)
    lin_fits[g] = my_fit
    plt.plot(T_arr, np.poly1d(my_fit)(T_arr), label=g)  # , color=plt.cm.jet(tmp_df.U_neg.mean())
plt.xlim([0.0031, 0.0034])
plt.legend(prop={'size': 16})
plt.xlabel(r'$\frac{1}{T}$')
plt.ylabel(r'$\ln(c_{li,0})$')
plt.savefig(os.path.join(op_dir, 'Arrhenius_plot_w_fit_hi_res.png'), dpi=800)



corr_fig = plt.figure(figsize=(14, 9))
plt.matshow(df_cli0.corr(), fignum=corr_fig.number)
plt.xticks(range(df_cli0.select_dtypes(['number']).shape[1]),
           df_cli0.select_dtypes(['number']).columns,
           fontsize=14, rotation=45)
plt.yticks(range(df_cli0.select_dtypes(['number']).shape[1]),
           df_cli0.select_dtypes(['number']).columns,
           fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
corr_fig.savefig(os.path.join(op_dir, 'cli0_correlation_figure.png'), dpi=corr_fig.dpi)

U_mesh, T_mesh = np.meshgrid(df_cli0['U_neg'], df_cli0['T'])
z_pos = np.zeros(U_mesh.shape).flatten()
dz = df_cli0['c_li0'].values.ravel()

gb = df_cli0.groupby(['soc', 'T'])
for g in gb.groups:
    k = gb.get_group(g)
    grp_fig = plt.figure(figsize=(16, 9))
    for idx in k.index:
        cl_id = re.search(r'cell\d', idx)
        plt.plot(t / (3600 * 8760),
                 fit_c_li0_fun(t, k.loc[idx, 'c_li0'], k.loc[idx, 'U_neg'], k.loc[idx, 'T'] + 273.15) / 3600,
                 label='{}'.format(cl_id))
        plt.title('T_{}_{}'.format(k.loc[idx, 'T'], k.loc[idx, 'soc']))
        plt.xlabel('Time [year]')
        plt.ylabel('Capacity lost to SEI [Ah]')
        plt.savefig(os.path.join(op_dir, 'T_{}_{}.png'.format(k.loc[idx, 'T'], k.loc[idx, 'soc'])), dpi=grp_fig.dpi)

L_sei = Q * V_sei / (A_spec * F * V_neg_el)

sol_dict = {}

U_rng = u_look_up([1, 0.5, 0])
soc_rng = [0.99, 0.5, 0.01]
u = u_look_up(soc_rng)
T_rng = scipy.constants.convert_temperature([0, 25, 60], 'Celsius', 'Kelvin')

c_li_max_init = 5e9 / NA * 1e6
D_li_sei_init = 5e-12
A_el_init = A_spec * V_neg_el
# mu_li_sei_init = 1.121 * ev_J_conv * NA  # J / mol  Taken from Shi paper (might be wrong)
mu_li_sei_init = 10
init_dict = {
    'c_li_max': c_li_max_init,
    'D_li_sei': D_li_sei_init,
    'A_el': A_el_init,
    'mu_li_sei': mu_li_sei_init
}
rng_dict = {
    'c_li_max': [-1, 2],
    'D_li_sei': [-2, 1],
    'A_el': [0, 1],
    'mu_li_sei': [-1, 3]
}
sc_dict = {}

for key in init_dict:
    sc_dict[key] = np.float_power(10, np.linspace(rng_dict[key][0], rng_dict[key][1], 4))
rng = np.arange(-2, 4, 2)
scaling = np.float_power(10, rng)

for key in sc_dict:
    for scaling in sc_dict[key]:
        inv_par = scaling * init_dict[key]
        c_li_max = c_li_max_init
        D_li_sei = D_li_sei_init
        A_el = A_el_init
        mu_li_sei = mu_li_sei_init
        if 'c_li' in key:
            c_li_max = scaling * init_dict[key]
        elif 'D_li' in key:
            D_li_sei = scaling * init_dict[key]
        elif 'A_el' in key:
            A_el = scaling * init_dict[key]
        elif 'mu_li' in key:
            mu_li_sei = scaling * init_dict[key]
        # Loop over voltage parameter
        for soc in soc_rng:
            U = u_look_up(soc)
            sol_dict['{}_{:.2e}_sol_{:.2f}'.format(key, inv_par, soc)] = \
                odeint(dQdt_li_diff_fun, Q0, t, args=(c_li_max, D_li_sei, A_el, V_sei, U, T0, mu_li_sei))
            L_sei = sol_dict['{}_{:.2e}_sol_{:.2f}'.format(key, inv_par, soc)] * V_sei / (A_spec * F * V_neg_el)
        for T in T_rng:
            sol_dict['{}_{:.2e}_T_{:.2f}'.format(key, inv_par, T)] = \
                odeint(dQdt_li_diff_fun, Q0, t, args=(c_li_max, D_li_sei, A_el, V_sei, U0, T, mu_li_sei))
        sol_dict['{}_{:.2e}_base'.format(key, inv_par)] = \
            odeint(dQdt_li_diff_fun, Q0, t, args=(c_li_max, D_li_sei, A_el, V_sei, U0, T0, mu_li_sei))


# fig, ax = plt.subplots(3, 2, figsize=(16, 9), sharey=True, sharex=True)
# mng = plt.get_current_fig_manager()
# mng.window.showMaximized()
for in_key in init_dict:
    fig, ax = plt.subplots(3, 2, figsize=(16, 9), sharey='col', sharex=True)
    for key in sol_dict:

        if in_key in key:
            print(key)
            u05 = re.search('sol_0.99', key)
            u4 = re.search('sol_0.5', key)
            u8 = re.search('sol_0.01', key)
            T0C = re.search('T_273', key)
            T25 = re.search('T_298', key)
            T45 = re.search('T_333', key)
            if u05:
                ax[0, 0].plot(t / (8760 * 3600), sol_dict[key] / 3600, label=key.split('_sol')[0])
            elif u4:
                ax[1, 0].plot(t / (8760 * 3600), sol_dict[key] / 3600, label=key.split('_sol')[0])
            elif u8:
                ax[2, 0].plot(t / (8760 * 3600), sol_dict[key] / 3600, label=key.split('_sol')[0])
            elif T0C:
                ax[0, 1].plot(t / (8760 * 3600), sol_dict[key] / 3600, label=key.split('_sol')[0])
            elif T25:
                ax[1, 1].plot(t / (8760 * 3600), sol_dict[key] / 3600, label=key.split('_T')[0])
            elif T45:
                ax[2, 1].plot(t / (8760 * 3600), sol_dict[key] / 3600, label=key.split('_T')[0])
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[2, 0].set_xlabel('Time [years]')
    ax[2, 1].set_xlabel('Time [years]')
    for i, item in enumerate(U_rng):
        ax[i, 0].set_title('SOL={}, T=298K'.format(soc_rng[i]))
        ax[i, 1].set_title('SOL=0.5, T={:.2f}K'.format(T_rng[i]))

    fig.text(0.06, 0.5, 'Capacity lost to SEI growth [Ah]', va='center', rotation='vertical', fontsize=20)
    fig.savefig(os.path.join(op_dir, '{}_{:%Y-%m-%d}.png'.format(in_key, td)), dpi=fig.dpi)

for param in init_dict:
    fig = plt.figure(figsize=(16, 9))
    for key in sol_dict:
        if 'base' in key and param in key:
            plt.plot(t / (8760*3600), sol_dict[key] / 3600, label=key.split('_base')[0])
    plt.title('{} at T={}K, SOL={:.2f}'.format(param, T0, sol0))
    plt.xlabel('Time [years]')
    plt.ylabel('Capacity lost to SEI growth [Ah]')
    # plt.ylim([0.008, 0.012])
    plt.legend()
    plt.savefig(os.path.join(op_dir, '{}_base_{:%Y-%m-%d}.png'.format(param, td)), dpi=fig.dpi)
plt.close('all')
