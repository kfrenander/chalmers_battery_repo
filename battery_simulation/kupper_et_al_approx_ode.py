import numpy as np
import scipy.interpolate
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.constants
plt.rcParams['axes.grid'] = True
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 16


def dQdt_fun(Q, t, A_spec, A_surf, V_sei, k_0_sei, E_a, T, eta, c_solv):
    F = scipy.constants.value('Faraday constant')
    R = scipy.constants.value('molar gas constant')
    k_f = k_0_sei * np.exp(-E_a / (R*T)) * np.exp(-0.5*F*eta /(R*T))
    r0 = k_f * c_solv**2
    dQdt = A_spec * A_surf * F / (Q * V_sei) * r0
    return dQdt


F = scipy.constants.value('Faraday constant')
eq_pot_file = r"C:\battery-model\PythonScripts\battery_simulation\c6_eq_pot.txt"
u_look_up = scipy.interpolate.interp1d(np.loadtxt(eq_pot_file)[:, 0], np.loadtxt(eq_pot_file)[:, 1])

# Cell parameters
r_part = 12.5e-6  # m
eps_neg = 0.525  # -
A_spec = 3 * eps_neg / r_part  # m2 / m3
L_neg = 105e-6  # m
A_cell = 0.029 * 28  # m2
V_neg_el = L_neg * A_cell
A_surf = V_neg_el * A_spec

# SEI material properties and initial values
M_sei = 0.5 * 0.16 + 0.5 * 0.07  # kg / mol
rho_sei = 1.6e4  # kg / m3
V_sei = M_sei / rho_sei  # m3 / mol
L0 = 5e-9  # m, initial thickness sei
Q0 = A_surf * F * L0 / V_sei

# Reaction quantities, all values from Kupper et al 2018
E_a = 55.5e3  # J / mol Activation energy
# k0 = 1.02e-19  # m^4 / (kmol * s)  Pre-exponential factor
c_solv = 8  # kmol / m^3  Concentration of EC

# Model variables
T0 = scipy.constants.convert_temperature(45, 'Celsius', 'Kelvin')
eta = 0.1

# Solve ODE
k0 = 2.7e-8  # Assuming the factor from Kupper et al is not k0 but kc
t = np.linspace(0, 8760*3600*8, 10000)
# (Q, t, A_spec, A_surf, V_sei, k_0_sei, E_a, T, eta, c_solv)
Q = odeint(dQdt_fun, Q0, t, args=(A_spec, A_surf, V_sei, k0, E_a, T0, eta, c_solv))

plt.plot(t / (3600*8760), Q / 3600)
