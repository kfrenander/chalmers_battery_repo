import pandas as pd
import numpy as np
import lmfit
import matplotlib.pyplot as plt
import os
import re
from matplotlib import cm
import matplotlib as mpl
from scipy.signal import find_peaks
from PythonScripts.rpt_data_analysis.ReadRptClass import look_up_fce
from matplotlib.offsetbox import AnchoredText
from scipy.interpolate import interp1d
import scipy.constants
from PythonScripts.backend_fix import fix_mpl_backend
from scipy.signal import argrelextrema
import schemdraw
import schemdraw.elements as elm
fix_mpl_backend()
plt.rcParams['figure.figsize'] = 8, 6
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['lines.linewidth'] = 1.7
plt.rcParams['xtick.labelsize'] = 'x-large'
plt.rcParams['ytick.labelsize'] = 'x-large'
lbl_font = {'weight': 'normal',
            'size': 20}
anch_font = {'size': 14}
plt.rc('legend', fontsize=14)
mark_size = 5
cap_size = 6
plt.rc('font', **{"family": 'sans-serif', 'sans-serif': 'Helvetica'})
plt.style.use('kelly_colors')


def fce_converter(rpt_str):
    rpt_num = int(re.search(r'\d+', rpt_str).group())
    return (rpt_num - 1) * 50


def read_gamry_metadata(file_string):
    with open(file_string) as readfile:
        text = readfile.read()
    return text.split('ZCURVE')[0]


def z_w(sigma, w):
    return (1 - 1j) * sigma * w ** (-1 / 2)


def z_flw(sigma, w, D, delta):
    return z_w(sigma, w) * np.tanh(delta * (1j * w / D) ** 1/2)


def z_non_ideal_warburg(w, D, delta):
    return np.tanh(delta * (1j * w / D) ** 1 / 2)


def Zc(C, w):
    return -1j / (w * C)


def z_cpe(C, a, w):
    return 1 / ((1j * C)**a*w)


def z_l(l, w):
    return 1j*w*l


def Zr(R):
    return R


def Zpar(a, b):
    return 1 / (1 / a + 1 / b)


def z_randles(omega, sigma, c_dl, r_ct, r_0):
    return Zr(r_0) + Zpar(Zr(r_ct) + z_w(sigma, omega), Zc(c_dl, omega))


def z_randles_cpe(omega, sigma, c_dl, alpha_cdl, r_ct, r_0):
    return Zr(r_0) + Zpar(Zr(r_ct) + z_w(sigma, omega), z_cpe(c_dl, alpha_cdl, omega))


def z_randles_2rc(omega, sigma, c_dl1, r_ct1, c_dl2, r_ct2, r_0):
    z_ran_2rc = (Zr(r_0) +
                 Zpar(Zr(r_ct1), Zc(c_dl1, omega)) +
                 Zpar(Zr(r_ct2) + z_w(sigma, omega), Zc(c_dl2, omega)))
    return z_ran_2rc


def z_randles_2rc_cpe(omega, sigma, c_dl1, alpha_cdl1, r_ct1, c_dl2, alpha_cdl2, r_ct2, r_0):
    z_ran_2rc_cpe = (Zr(r_0) +
                     Zpar(Zr(r_ct1), z_cpe(c_dl1, alpha_cdl1, omega)) +
                     Zpar(Zr(r_ct2) + z_w(sigma, omega), z_cpe(c_dl2, alpha_cdl2, omega)))
    return z_ran_2rc_cpe


def z_randles_extended(omega, sigma, c_dl1, r_ct1, c_dl2, r_ct2, r_aux, c_aux, r_0):
    z_ran_ext = (Zr(r_0) +
                 Zpar(Zr(r_ct1), Zc(c_dl1, omega)) +
                 Zpar(Zr(r_ct2), Zc(c_dl2, omega)) +
                 Zpar(Zr(r_aux) + z_w(sigma, omega), Zc(c_aux, omega)))
    return z_ran_ext


def z_randles_double(omega, sigma1, c_dl1, r_ct1, sigma2, c_dl2, r_ct2, r_0):
    z_ran_double = (Zr(r_0) +
                    Zpar(Zr(r_ct1) + z_w(sigma1, omega), Zc(c_dl1, omega)) +
                    Zpar(Zr(r_ct2) + z_w(sigma2, omega), Zc(c_dl2, omega)))
    return z_ran_double


def z_randles_inductive(omega, sigma, l, r_ct1, c_dl2, r_ct2, r_aux, c_aux, r_0):
    z_ran_ext = (Zr(r_0) +
                 Zpar(Zr(r_ct1), z_l(l, omega)) +
                 Zpar(Zr(r_ct2), Zc(c_dl2, omega)) +
                 Zpar(Zr(r_aux) + z_w(sigma, omega), Zc(c_aux, omega)))
    return z_ran_ext


def z_rc(R0, Rct, Cdl, omega):
    return Zr(R0) + Zpar(Zr(Rct), Zc(Cdl, omega))


def draw_circuit(mode='std'):
    if mode=='std':
        with schemdraw.Drawing() as d:
            d.config(unit=2)
            d += elm.Line().left().length(8)
            d += elm.Line().up().length(0.5)
            d += elm.SourceV().label('$U_{OCV}$')
            d += elm.Line().length(0.5)
            d += elm.Resistor().right().label('$R_0$')
            d += elm.Dot()
            d.push()
            d += elm.Line().up().length(1)
            d += elm.Line().right().length(1)
            d += elm.Capacitor().right().label('$C_{dl}$')
            d += elm.Line().length(1)
            d += elm.Line().down().length(1)
            d.pop()
            d += elm.Line().down().length(1)
            d += elm.Resistor().right().label('$R_{ct}$', loc='bottom')
            d += elm.Resistor().label('$R_W$', loc='bottom')
            d += elm.Line().up().length(1)
            d += elm.Dot()
            d += elm.Line().right().length(2)
    elif mode == '2rc_cpe':
        with schemdraw.Drawing() as d:
            d.config(unit=2)
            d += elm.Line().left().length(10)
            d += elm.Line().up().length(0.5)
            d += elm.SourceV().label('$U_{OCV}$')
            d += elm.Line().length(0.5)
            d += elm.Resistor().right().label('$R_0$')
            d += elm.Dot()
            d.push()
            d += elm.Line().up().length(1)
            d += elm.Capacitor().right().label('$C_{sei}$')
            d += elm.Line().down().length(1)
            d.pop()
            d += elm.Line().down().length(1)
            d += elm.Resistor().right().label('$R_{sei}$', loc='bottom')
            d += elm.Line().up().length(1)
            d += elm.Dot()
            d += elm.Line().right().length(1)
            d += elm.Dot()
            d.push()
            d += elm.Line().up().length(1)
            d += elm.Line().right().length(1)
            d += elm.Capacitor().right().label('$C_{dl}$')
            d += elm.Line().length(1)
            d += elm.Line().down().length(1)
            d.pop()
            d += elm.Line().down().length(1)
            d += elm.Resistor().right().label('$R_{ct}$', loc='bottom')
            d += elm.Resistor().label('$R_W$', loc='bottom')
            d += elm.Line().up().length(1)
            d += elm.Dot()
            d += elm.Line().right().length(1)
    return d


def calc_complex_diff(a, b):
    re_diff = sum(np.sqrt((a.real - b.real)**2))
    im_diff = sum(np.sqrt((a.imag - b.imag)**2))
    return re_diff + im_diff


def set_cycle_number(test_nbr):
    t_interp = np.array([1, 2, 3, 4])
    c_interp = np.array([0, 300, 600, 900])
    return np.interp(test_nbr, t_interp, c_interp)


def find_test_number(f_name):
    t = re.search(r'TEST\d+', f_name).group()
    return int(re.search(r'\d+', t).group())


def find_cell_id(dir_name):
    c_id = re.search(r'Cell_\d+', dir_name).group()
    return c_id


def ceil_to_sig_nbr(x):
    pwr = -np.floor(np.log10(x))
    return np.ceil(x*10**pwr) / 10**pwr


def set_lim_for_plot(fig_obj):
    try:
        fig_lines = fig_obj.gca().lines
        x_min = min([min(l.get_xdata()) for l in fig_lines])
        x_max = max([max(l.get_xdata()) for l in fig_lines])
        y_min = min([min(l.get_ydata()) for l in fig_lines])
        y_max = max([max(l.get_ydata()) for l in fig_lines])
        ax_len = ceil_to_sig_nbr(max(x_max - x_min, y_max - y_min))
        offset = ax_len / 40
        return (x_min - offset, x_min + ax_len), (y_min - offset, y_min + ax_len)
    except AttributeError:
        print('Input to function should be matplotlib figure with lines')
        return None


def kelly_contrast_colors(i):
    kelly_colors_hex = [
        '#E68FAC',
        '#222222',
        '#F3C300',
        '#875692',
        '#F38400',
        '#A1CAF1',
        '#BE0032',
        '#C2B280',
        '#848482',
        '#008856'
    ]
    return kelly_colors_hex[i]


class GamryDataSet(object):

    def __init__(self, file_string):
        self.data_file = file_string
        self.cls_cmap = cm.Paired
        self.data_init_row = self.find_data_init_row(file_string)
        self.dataset = self.read_gamry_file(file_string)
        self.char_freq = self.find_char_freq()
        self.metadata = read_gamry_metadata(file_string)
        self.fit_minimizer = lmfit.minimizer.MinimizerResult()
        self.fitted_data = dict()
        self.test_nbr = find_test_number(file_string)
        self.fce_nbr = set_cycle_number(self.test_nbr)
        self.diff_slope = self.calc_diff_slope()
        self.lin_fit_diff = self.find_lin_fit()
        self.test_ref = f'{int(self.fce_nbr)}_FCE'

    def __repr__(self):
        return f'{self.fce_nbr}'

    def objective(self, params):
        w = self.dataset['freq_hz'].values*2*np.pi
        if 'r_aux' in params:
            r_ct1 = params['r_ct1'].value
            r_ct2 = params['r_ct2'].value
            c_dl1 = params['c_dl1'].value
            c_dl2 = params['c_dl2'].value
            r_aux = params['r_aux'].value
            c_aux = params['c_aux'].value
            sig = params['sigma'].value
            r_0 = params['r_0'].value
            z_full = z_randles_extended(w, sig, c_dl1, r_ct1, c_dl2, r_ct2, r_aux, c_aux, r_0)
        elif 'r_ct2' in params:
            r_ct1 = params['r_ct1'].value
            r_ct2 = params['r_ct2'].value
            c_dl1 = params['c_dl1'].value
            c_dl2 = params['c_dl2'].value
            sig = params['sigma'].value
            r_0 = params['r_0'].value
            z_full = z_randles_2rc(w, sig, c_dl1, r_ct1, c_dl2, r_ct2, r_0)
        else:
            r_ct = params['r_ct'].value
            c_dl = params['c_dl'].value
            sig = params['sigma'].value
            r_0 = params['r_0'].value
            z_full = z_randles(w, sig, c_dl, r_ct, r_0)
        res_im = self.dataset['z_imag_ohm'] - z_full.imag
        res_re = self.dataset['z_real_ohm'] - z_full.real
        return np.append(res_re, res_im)

    def plot_fitted_randles(self):
        w = 2 * np.pi * self.dataset['freq_hz'].values
        if 'r_aux' in self.fit_minimizer.params:
            randles_fitted = z_randles_extended(w,
                                                self.fit_minimizer.params['sigma'],
                                                self.fit_minimizer.params['c_dl1'],
                                                self.fit_minimizer.params['r_ct1'],
                                                self.fit_minimizer.params['c_dl2'],
                                                self.fit_minimizer.params['r_ct2'],
                                                self.fit_minimizer.params['r_aux'],
                                                self.fit_minimizer.params['c_aux'],
                                                self.fit_minimizer.params['r_0']
                                                )
            plot_label = 'Extended Randles'
        elif 'r_ct2' in self.fit_minimizer.params:
            randles_fitted = z_randles_2rc(w,
                                           self.fit_minimizer.params['sigma'],
                                           self.fit_minimizer.params['c_dl1'],
                                           self.fit_minimizer.params['r_ct1'],
                                           self.fit_minimizer.params['c_dl2'],
                                           self.fit_minimizer.params['r_ct2'],
                                           self.fit_minimizer.params['r_0']
                                           )
            plot_label = '2 RC Randles'
        else:
            randles_fitted = z_randles(w,
                                       self.fit_minimizer.params['sigma'],
                                       self.fit_minimizer.params['c_dl'],
                                       self.fit_minimizer.params['r_ct'],
                                       self.fit_minimizer.params['r_0']
                                       )
            plot_label = 'Standard Randles'
        plt.plot(randles_fitted.real, -randles_fitted.imag, label=f'Fitted with {plot_label}')
        plt.xlabel('Re(Z) [Ohm]')
        plt.ylabel('-Im(Z) [Ohm]')
        return None

    def alternative_fit(self, mode='std'):
        exp_data = (self.dataset['z_real_ohm'] + self.dataset['z_imag_ohm']*1j).values
        if mode == 'std':
            fit_name = 'Standard Randles'
            fmodel = lmfit.Model(z_randles)
            params = fmodel.make_params(sigma=2e-3, c_dl=5, r_ct=3e-2, r_0=3e-2)
        elif mode == '2rc':
            fit_name = '2 RC Randles'
            fmodel = lmfit.Model(z_randles_2rc)
            params = fmodel.make_params(sigma=2e-3, c_dl1=0.05, r_ct1=2e-2, c_dl2=5, r_ct2=2.5e-2, r_0=1e-2)
        elif mode == 'ind':
            fit_name = 'Randles w inductor'
            fmodel = lmfit.Model(z_randles_inductive)
            params = fmodel.make_params(sigma=2e-3, l=1e-8, r_ct1=2e-2, c_dl2=5,
                                        r_ct2=2.5e-2, r_0=1e-2, r_aux=6e-2, c_aux=50)
        elif mode == 'd_war':
            fit_name = 'Double warburg'
            fmodel = lmfit.Model(z_randles_double)
            params = fmodel.make_params(sigma1=4e-2, sigma2=4e-3, c_dl1=2e-3, r_ct1=2e-2,
                                        c_dl2=5, r_ct2=2.5e-2, r_0=1e-3)
        elif mode == 'std_cpe':
            fit_name = 'Randles w CPE'
            fmodel = lmfit.Model(z_randles_cpe)
            params = fmodel.make_params(sigma=2e-3, c_dl=5, alpha_cdl=0.5, r_ct=3e-2, r_0=3e-2)
        elif mode == '2rc_cpe':
            fit_name = '2 RC cpe Randles'
            fmodel = lmfit.Model(z_randles_2rc_cpe)
            params = fmodel.make_params(sigma=2e-3, c_dl1=0.05, alpha_cdl1=0.5, r_ct1=2e-2,
                                        c_dl2=5, alpha_cdl2=0.5, r_ct2=2.5e-2, r_0=1e-2)
        else:
            fit_name = 'Extended Randles'
            fmodel = lmfit.Model(z_randles_extended)
            params = fmodel.make_params(sigma=2e-3, c_dl1=0.05, r_ct1=2e-2, c_dl2=5,
                                        r_ct2=2.5e-2, r_0=1e-2, r_aux=6e-2, c_aux=50)
        for p in params:
            params[p].min = 0
            if 'alpha' in p:
                params[p].max = 1
        result = fmodel.fit(exp_data, params, omega=self.dataset['freq_hz'].values * 2 * np.pi)
        self.fitted_data[fit_name] = result
        return result

    def fit_randles_circuit(self, mode='std'):
        params = lmfit.Parameters()
        if mode == 'std':
            params.add('r_ct', value=3e-2, min=1e-3, max=0.5)
            params.add('c_dl', value=5)
            params.add('sigma', value=2e-3)
            params.add('r_0', value=5e-2, min=1e-3, max=0.5)
        elif mode == '2rc':
            params.add('r_ct1', value=2e-2, min=1e-3, max=0.5)
            params.add('c_dl1', value=0.05, min=1e-5, max=1e4)
            params.add('r_ct2', value=2.5e-2, min=1e-3, max=0.5)
            params.add('c_dl2', value=5, min=1e-5, max=1e4)
            params.add('sigma', value=2e-3, min=1e-4)
            params.add('r_0', value=5e-2, min=1e-4, max=0.5)
        elif mode == 'ext':
            params.add('r_ct1', value=2e-2, min=1e-3, max=0.5)
            params.add('c_dl1', value=0.05, min=1e-5, max=1e4)
            params.add('r_ct2', value=2.5e-2, min=1e-3, max=0.5)
            params.add('c_dl2', value=5, min=1e-5, max=1e4)
            params.add('r_aux', value=6e-2)
            params.add('c_aux', value=50)
            params.add('sigma', value=2e-3, min=1e-4)
            params.add('r_0', value=5e-2, min=1e-4, max=0.5)
        self.fit_minimizer = lmfit.minimize(self.objective, params, method='leastsq')
        return self.fit_minimizer.params

    def read_gamry_file(self, file_string):
        col_names_hyb = ['pt', 'time', 'freq_hz', 'z_real_ohm', 'z_imag_ohm', 'z_sig_ohm', 'z_mod_ohm', 'z_phs_ohm',
                         'i_dc_amp', 'v_dc_volt', 'ie_range', 'i_mod_amp', 'v_mod_volt']
        col_names_galv = ['pt', 'time', 'freq_hz', 'z_real_ohm', 'z_imag_ohm', 'z_sig_ohm',
                          'z_mod_ohm', 'z_phs_ohm', 'i_dc_amp', 'v_dc_volt', 'ie_range']
        df = pd.read_csv(file_string, sep='\t', skiprows=self.data_init_row + 2,
                         header=None, decimal=',', encoding='ansi')
        df = df.dropna(how='all', axis=1)
        if df.shape[1] == 13:
            df.columns = col_names_hyb
        elif df.shape[1] == 11:
            df.columns = col_names_galv
        df = df.set_index('pt', drop=False)
        df = df[df['z_imag_ohm'] < 1e-3]
        return df

    def set_linestyle(self):
        if self.fce_nbr < 100:
            return 'solid'
        elif self.fce_nbr < 400:
            return (0, (1, 1))
        elif self.fce_nbr < 700:
            return 'dashdot'
        elif self.fce_nbr < 1000:
            return (0, (5, 1))

    def set_scattermarker(self, f):
        if f < 0.05:
            return u"$\u2605$"  # Star
        elif f < 0.5:
            return u"$\u25A0$"  # Filled black square
        elif f < 50:
            return u"$\u2BC5$"  # Upwards triangle
        elif f < 1e4:
            return u"$\u2B1F$"  # Black pentagon

    def plot_nyquist(self, c='', ext_label='', mark_char_f=True):
        if not c:
            c = kelly_contrast_colors(int(self.fce_nbr / 100))
        if not ext_label:
            ext_label = f'EIS raw data {self.test_ref}'
        if not self.fce_nbr:
            self.fce_nbr = set_cycle_number(find_test_number(self.data_file))
        plt.plot(self.dataset['z_real_ohm'], -self.dataset['z_imag_ohm'],
                 linestyle=self.set_linestyle(), color=c, label=ext_label)
        plt.xlabel(r'Re(Z) / $\Omega$')
        plt.ylabel('-Im(Z) / $\Omega$')
        freq_pts = [1000, 10, 0.1, 0.01]
        for pt in freq_pts:
            df_sort = self.dataset.iloc[(self.dataset['freq_hz'] - pt).abs().argsort()[:1]]
            plt.scatter(df_sort['z_real_ohm'], -df_sort['z_imag_ohm'],
                        marker=self.set_scattermarker(pt), s=80, c=c)
        textstr = u"$\u2605$ 0.01Hz \n" \
                  "$\u25A0$ 0.1Hz \n" \
                  "$\u2BC5$ 10Hz \n" \
                  "$\u2B1F$ 1000Hz"
        anch_text = AnchoredText(textstr, loc=4, prop=anch_font)
        plt.gca().add_artist(anch_text)
        plt.legend(loc=2)
        return None

    def find_char_freq(self):
        min_pts = argrelextrema(self.dataset.z_imag_ohm.values, np.greater)
        f_char = self.dataset.loc[min_pts[0][-1], 'freq_hz']
        return f_char

    def plot_re_z_v_sigma(self, c='', ext_label=''):
        if not c:
            c = self.cls_cmap(self.fce_nbr / 900)
        if not ext_label:
            ext_label = f'Diffusion style plot for {self.test_ref}'
        plt.plot(np.sqrt(1 / (self.dataset['freq_hz'] * 2 * np.pi)),
                 self.dataset['z_real_ohm'],
                 color=c,
                 label=ext_label,
                 linestyle=self.set_linestyle())
        plt.xlabel(r'$\omega^{-1/2} [\sqrt{s}]$')
        plt.ylabel('Re(Z)')
        plt.legend(loc=2)
        return None

    def plot_linear_fit_re_z(self, cell_id=''):
        if not self.lin_fit_diff.any():
            self.find_lin_fit()
        df = self.find_linear_diff_slope_pts()
        plt.scatter(df.z_real_ohm, df.t_data,
                    marker='*',
                    label=f'Raw data {cell_id}, fce {self.fce_nbr}')
        plt.plot(df.z_real_ohm,
                 np.poly1d(self.lin_fit_diff)(df.z_real_ohm),
                 alpha=0.75,
                 color='red',
                 linestyle='dashdot')
        return None

    def find_linear_diff_slope_pts(self):
        self.dataset.loc[:, 't_data'] = np.sqrt(1 / (self.dataset['freq_hz'] * 2 * np.pi))
        drdt = np.gradient(self.dataset['z_real_ohm'], self.dataset.loc[:, 't_data'] )
        self.dataset.loc[:, 'dRe'] = drdt
        tol_val = drdt.mean() / 100
        fin_slope = drdt[-1]
        relevant_pts = self.dataset[abs(drdt - fin_slope) < tol_val]
        return relevant_pts

    def calc_diff_slope(self):
        df = self.find_linear_diff_slope_pts()
        return df['dRe'].mean()

    def find_lin_fit(self):
        df = self.find_linear_diff_slope_pts()
        return np.polyfit(df.z_real_ohm, df.t_data, deg=1)

    def calc_diffusivity(self, ocv):
        # Define the needed physical constants that are to be used to calculate diffusion
        surf_area = 3*0.5/5e-6*(0.925*0.065*160e-6 + 0.85*0.0635*140e-6)
        molar_mass = 95.88e-3  #kg/mol
        dens = 2100 #
        molar_vol = molar_mass / dens
        F = scipy.constants.value('Faraday constant')
        ocv_near50 = ocv[:, abs(ocv[0, :] - 0.5) < 0.01]
        ocv_grad = np.gradient(ocv_near50[1], ocv_near50[0]).mean()
        D = 0.5 * (ocv_grad * molar_vol / (surf_area * F))**2 * self.diff_slope**(-2)
        return D

    def find_axis_lim(self):
        df = self.dataset
        re_diff = df.z_real_ohm.max() - df.z_real_ohm.min()
        im_diff = df.z_imag_ohm.max() - df.z_real_ohm.min()
        ax_len = ceil_to_sig_nbr(max(re_diff, im_diff))
        offset = ax_len / 30
        x_min = df.z_real_ohm.min() - offset
        x_max = x_min + ax_len + offset
        y_min = -df.z_imag_ohm.max() - offset
        y_max = y_min + ax_len + offset
        return (x_min, x_max), (y_min, y_max)

    @staticmethod
    def find_data_init_row(file_string):
        with open(file_string) as readfile:
            for cnt, line in enumerate(readfile):
                if 'Zreal' in line:
                    init_line = cnt
        try:
            return init_line
        except NameError:
            print(f'No properly structured data found for file {file_string}')
            return None
        except:
            print(f'Unknown generic error for file {file_string}')
            return None


class GroupGamryDatasets(object):

    def __init__(self, data_dir):
        self.top_dir = data_dir
        self.data_dict = {}
        self.fit_dict = {}
        self.cell_id = find_cell_id(self.top_dir)
        self.f_list = self.find_data_files()
        self.plot_col = self.find_cell_key_col()
        self.fce_arr = np.array([set_cycle_number(find_test_number(n)) for n in self.f_list])
        self.fill_data_dict()
        self.ocv_dict = self.find_ocv()
        self.s_arr = self.fill_set_diff_slope()
        self.fit_name = ''

    def __repr__(self):
        return "{}_fce{}".format(self.cell_id, *self.data_dict)

    def set_fit_type(self, mode):
        if mode == 'std':
            fit_name = 'Standard Randles'
        elif mode == '2rc':
            fit_name = '2 RC Randles'
        elif mode == 'ind':
            fit_name = 'Randles w inductor'
        elif mode == 'd_war':
            fit_name = 'Double warburg'
        else:
            fit_name = 'Extended Randles'
        return fit_name

    def find_circ_elem_type(self, inp_str):
        if 'c_' in inp_str:
            return 'Capacitor'
        elif 'r_' in inp_str:
            return 'Resistor'
        elif 'sig' in inp_str:
            return 'Warburg'
        else:
            return 'Inductor'

    def find_diffusivity(self):
        self.find_ocv()
        diff_dict = {k: self.data_dict[k].calc_diffusivity(self.ocv_dict[k]) for k in self.ocv_dict}
        return diff_dict

    def find_data_files(self):
        tmp_list = []
        for r, d, f in os.walk(self.top_dir):
            for n in f:
                if n.endswith('.DTA'):
                    tmp_list.append(n)
        return tmp_list

    def find_cell_key_col(self):
        if '661' in self.cell_id:
            return 'maroon'
        elif '707' in self.cell_id:
            return 'forestgreen'
        elif '714' in self.cell_id:
            return 'orange'
        elif '717' in self.cell_id:
            return 'mediumblue'
        else:
            return None

    def find_ica_dir(self):
        if '661' in self.cell_id:
            return r"\\sol.ita.chalmers.se\groups\batt_lab_data\20200923\pickle_files_channel_1_1"
        elif '707' in self.cell_id:
            return r"\\sol.ita.chalmers.se\groups\batt_lab_data\20200923\pickle_files_channel_2_1"
        elif '714' in self.cell_id:
            return r"\\sol.ita.chalmers.se\groups\batt_lab_data\20200923\pickle_files_channel_2_5"
        elif '717' in self.cell_id:
            return r"\\sol.ita.chalmers.se\groups\batt_lab_data\20200923\pickle_files_channel_2_7"
        else:
            return None

    def find_ocv(self):
        pkl_dir = self.find_ica_dir()
        ica = {fce_converter(re.search(r'rpt_\d+', f).group()):
                   pd.read_pickle(os.path.join(pkl_dir, f)) for f in os.listdir(pkl_dir) if 'ica' in f}
        ocv = {f"{k:.0f}": self.calc_ocv_curve(ica[k]) for k in ica if k in self.fce_arr}
        return ocv

    def calc_ocv_curve(self, df):
        dchg = df[df['mode'].str.contains('DChg')]
        chrg = df[df['mode'].str.contains('_Chg')]
        soc_dchg = 1 - dchg['cap'] / dchg['cap'].max()
        soc_chrg = chrg['cap'] / chrg['cap'].max()
        dchg_volt = interp1d(soc_dchg, dchg['volt'])
        chrg_volt = interp1d(soc_chrg, chrg['volt'], fill_value='extrapolate')
        soc_pts = np.linspace(0, 1, 200)
        return np.array([soc_pts, np.mean([chrg_volt(soc_pts), dchg_volt(soc_pts)], axis=0)])

    def fill_data_dict(self):
        for f in self.f_list:
            self.data_dict[f"{self.set_fce_reference(f):.0f}"] = GamryDataSet(os.path.join(self.top_dir, f))
        return None

    def set_test_reference(self, test_name):
        t_nbr = find_test_number(test_name)
        c_nbr = set_cycle_number(t_nbr)
        t_ref = f'{self.cell_id}_{int(c_nbr)}FCE'
        return t_ref

    def set_fce_reference(self, test_name):
        t_nbr = find_test_number(test_name)
        c_nbr = set_cycle_number(t_nbr)
        return c_nbr

    def plot_set_nyquist(self,
                         fce_fltr='',
                         label='',
                         c='',
                         new_fig=True):
        if new_fig:
            fig, ax = plt.subplots(1, 1)
        for k in self.data_dict:
            if fce_fltr:
                if self.data_dict[k].fce_nbr in fce_fltr:
                    self.data_dict[k].plot_nyquist(c=c,
                                                   ext_label=f'FCE_{k} {label}')
            else:
                self.data_dict[k].plot_nyquist(c=c,
                                               ext_label=f'FCE_{k} {label}')
        try:
            return fig
        except UnboundLocalError:
            return None

    def plot_set_diff(self,
                      fce_fltr='',
                      label='',
                      c='',
                      new_fig=True):
        if new_fig:
            fig, ax = plt.subplots(1, 1)
        for k in self.data_dict:
            if fce_fltr:
                if self.data_dict[k].fce_nbr in fce_fltr:
                    self.data_dict[k].plot_re_z_v_sigma(c=c,
                                                        ext_label=f'FCE_{k:.0f} {label}')
            else:
                self.data_dict[k].plot_re_z_v_sigma(c=c,
                                                    ext_label=f'FCE_{k:.0f} {label}')
        try:
            return fig
        except UnboundLocalError:
            return None

    def fit_set_randles(self, mode='std', fltr=''):
        for k in self.data_dict:
            if fltr:
                if self.data_dict[k].fce_nbr in fltr:
                    self.fit_dict[k] = self.data_dict[k].alternative_fit(mode=mode)
                    self.fit_name = self.data_dict[k]
            else:
                self.fit_dict[k] = self.data_dict[k].alternative_fit(mode=mode)
        df = pd.DataFrame(index=[par for par in self.fit_dict[k].params],
                          columns=[self.data_dict[k].fce_nbr for k in self.data_dict])
        for key in self.fit_dict:
            col = self.data_dict[key].fce_nbr
            for p in self.fit_dict[key].params:
                df.loc[p, col] = self.fit_dict[key].params[p].value
        return df.astype('float')

    def fill_set_diff_slope(self):
        if not self.data_dict:
            self.fill_data_dict()
        slope_list = []
        for k in self.data_dict:
            slope_list.append(self.data_dict[k].diff_slope)
        return np.array(slope_list)

    def plot_parameter_evolution(self, mode='std'):
        par_df = self.fit_set_randles(mode=mode)
        poss_param_dict = {
            'res': 'r_ct',
            'cap': 'c_',
            'war': 'sig',
            'ind': 'l'}
        par_figs = {k: plt.subplots(1, 1) for k in poss_param_dict
                    if self.check_fit_parameters(par_df, poss_param_dict[k])}
        for key in par_figs:
            for idx in par_df.index:
                if poss_param_dict[key] in idx:
                    par_figs[key][1].plot(par_df.columns, par_df.loc[idx, :], '-*', label=idx)
                    par_figs[key][1].set_xlabel('Full cycle equivalents')
                    par_figs[key][1].set_ylabel(f'{self.find_circ_elem_type(idx)} in ciruit')
                    # par_figs[key][1].set_title(f'{self.find_circ_elem_type(idx)} fitted in {self.set_fit_type(mode)} ECM')
                    par_figs[key][1].legend()
        return par_figs

    def plot_diff_slopes(self, c=''):
        if not c:
            c = self.plot_col
        plt.plot(self.fce_arr, self.s_arr,
                 color=c, marker='.', linestyle='dashed', label=self.cell_id)
        plt.xlabel('FCE')
        plt.ylabel('Slope [$\Omega / \sqrt{s}$]')
        return None

    def check_fit_parameters(self, df, par):
        return any(par in idx for idx in df.index)


class AllGamryData(object):

    def __init__(self, dir):
        self.master_dir = dir
        self.gamry_data_dict = {}
        self.rpt_data_dict = {}
        self.color_dict = {}
        self.cmap_dict = {}
        self.fill_data()
        self.fill_rpt_data()
        self.set_cell_colors()
        self.set_cell_cmap()
        self.max_set_fce = self.find_max_fce_in_set()
        self.output = r'Z:\Provning\EIS\Analysis'

    def fill_rpt_data(self):
        file_dict = {
            'Cell_661': r"\\sol.ita.chalmers.se\groups\batt_lab_data\20200923\pickle_files_channel_1_1\1_1_rpt_summary_dump.pkl",
            'Cell_707': r"\\sol.ita.chalmers.se\groups\batt_lab_data\20200923\pickle_files_channel_2_1\2_1_rpt_summary_dump.pkl",
            'Cell_714': r"\\sol.ita.chalmers.se\groups\batt_lab_data\20200923\pickle_files_channel_2_5\2_5_rpt_summary_dump.pkl",
            'Cell_717': r"\\sol.ita.chalmers.se\groups\batt_lab_data\20200923\pickle_files_channel_2_7\2_7_rpt_summary_dump.pkl",
            'Cell_727': '',
            'Cell_731': '',
            'Cell_733': '',
            'Cell_736': '',
            'Cell_738': ''
        }
        for key in file_dict:
            self.rpt_data_dict[key] = self.read_rpt_df(file_dict[key])
        return None

    def set_cell_colors(self):
        """
        Set a color for each cell for the case where one FCE level is plotted with multiple cells
        :return:
        """
        self.color_dict = {
            'Cell_661': 'maroon',
            'Cell_707': 'forestgreen',
            'Cell_714': 'darkorange',
            'Cell_717': 'mediumblue',
            'Cell_727': 'crimson',
            'Cell_731': 'chartreuse',
            'Cell_733': 'darkviolet',
            'Cell_736': 'black',
            'Cell_738': 'indianred'
        }
        return None

    def set_cell_cmap(self):
        """
        Set a qualitative colormap for each cell for the case where plots include both several FCE levels and several
        cell keys.
        :return:
        """
        self.cmap_dict = {
            'Cell_661': cm.Dark2,
            'Cell_707': cm.tab20,
            'Cell_714': cm.tab20b,
            'Cell_717': cm.tab20c
        }
        return None

    def find_max_fce_in_set(self):
        set_max_list = [self.gamry_data_dict[k].fce_arr.max() for k in self.gamry_data_dict]
        return max(set_max_list)

    def read_rpt_df(self, file_name):
        """
        Read RPT data sheet for corresponding degradation data set and fill with FCE values.
        :param file_name: 
        :return: 
        """
        try:
            df = pd.read_pickle(file_name)
            df.loc[:, 'FCE'] = [look_up_fce(idx) for idx in df.index]
        except FileNotFoundError:
            print('RPTs not yet indexed for this cell')
            df = pd.DataFrame()
        return df

    def fill_data(self):
        for root, dir, files in os.walk(self.master_dir):
            for d in dir:
                self.gamry_data_dict[find_cell_id(d)] = GroupGamryDatasets(os.path.join(root, d))

    def plot_filtered_fce(self,
                          fce_filter=[0, 300, 600, 900],
                          ax_setting='fix_ratio',
                          ind_fig=True,
                          save_fig=False):
        if not ind_fig:
            tmp = plt.figure()
        for fce in fce_filter:
            if ind_fig:
                tmp = plt.figure()
            for k in self.gamry_data_dict:
                gmy_data = self.gamry_data_dict[k].data_dict
                for key in gmy_data:
                    if float(key) == fce:
                        if not ind_fig:
                            lbl = f'{k}__{fce}FCE'
                            # col = self.cmap_dict[k](fce / self.max_set_fce)
                            col = self.color_dict[k]
                            title_str = f'FCE {" and ".join([str(f) for f in fce_filter])}'
                        else:
                            lbl = k
                            col = self.color_dict[k]
                            title_str = f'FCE {fce}'
                        # if ax_setting == 'fix_ratio':
                        #     x_val, y_val = gmy_data[key].find_axis_lim()
                        gmy_data[key].plot_nyquist(c=col, ext_label=lbl)
            tmp.gca().legend()
            tmp.gca().set_title(title_str)
            tmp.gca().grid(True)
            if ax_setting.lower() == 'fix':
                x_val = (0.01, 0.05)
                y_val = (0, 0.04)
            elif ax_setting.lower() == 'fix_ratio':
                x_val, y_val = set_lim_for_plot(tmp)
            if ax_setting:
                tmp.gca().set_xlim(x_val)
                tmp.gca().set_ylim(y_val)
            if save_fig and ind_fig:
                tmp.savefig(os.path.join(self.output, f"All_cells_fce_{fce}_{ax_setting}.png"))
        if save_fig and not ind_fig:
            tmp.savefig(os.path.join(self.output,
                                     f"All_cells_fce_{'_'.join([str(f) for f in fce_filter])}_{ax_setting}.png"),
                        dpi=tmp.dpi)
        return None

    def plot_diff_cell_wise(self, fce_filter=[0, 300, 600, 900], save_fig=False, new_fig=True):
        for k in self.gamry_data_dict:
            tmp = self.gamry_data_dict[k].plot_set_diff(c=self.color_dict[k],
                                                        fce_fltr=fce_filter,
                                                        new_fig=new_fig)
            tmp.gca().set_title(k)
            if save_fig:
                tmp.savefig(os.path.join(self.output,
                                         f"diffusion_cell_{k}_fce{'_'.join([str(f) for f in fce_filter])}.png"))
        return None

    def plot_filtered_diffusion_fce(self,
                                    fce_filter=[0, 300, 600, 900],
                                    ind_fig=True,
                                    save_fig=False,
                                    mode='ind_cell'):
        if not ind_fig:
            tmp = plt.figure()
        for fce in fce_filter:
            if ind_fig:
                tmp = plt.figure()
            for k in self.gamry_data_dict:
                gmy_data = self.gamry_data_dict[k].data_dict
                for key in gmy_data:
                    if key == fce:
                        if not ind_fig:
                            lbl = f'{k}__{fce}FCE'
                            col = self.color_dict[k]
                            title_str = f'FCE {" and ".join([str(f) for f in fce_filter])}'
                        else:
                            lbl = k
                            col = self.color_dict[k]
                            title_str = f'FCE {fce}'
                        gmy_data[key].plot_re_z_v_sigma(c=col, ext_label=lbl)  #
            tmp.gca().legend()
            tmp.gca().set_title(title_str)
            tmp.gca().grid(True)
            if save_fig and ind_fig:
                tmp.savefig(os.path.join(self.output, f"diffusion_all_cells_at_fce_{fce}.png"))
        if save_fig and not ind_fig:
            tmp.savefig(os.path.join(self.output,
                                     f"diffusion_all_cells_fce_{'_'.join([str(f) for f in fce_filter])}.png"),
                        dpi=tmp.dpi)
        return None


    def plot_cap_with_eis(self,
                          cell_key,
                          plt_canvas='',
                          fce_filter=[0, 300, 600, 900]):
        if not plt_canvas:
            plt_canvas, ax = plt.subplots(2, 1)
        else:
            try:
                ax = plt_canvas.axes
            except AttributeError:
                raise AssertionError('Input variable should be matplotlib figure')
        plt.sca(ax[0])
        self.gamry_data_dict[cell_key].plot_set_nyquist(c=self.color_dict[cell_key],
                                                        new_fig=False,
                                                        fce_fltr=fce_filter,
                                                        label=cell_key)
        rpt_df = self.rpt_data_dict[cell_key]
        rpt_df = rpt_df[rpt_df['FCE'].isin([0, 300, 600, 900])]
        ax[1].plot(rpt_df['FCE'], rpt_df['cap_relative'] * 100,
                   linestyle='dashed',
                   marker='.',
                   color=self.color_dict[cell_key],
                   label=cell_key)
        ax[1].set_xlabel('FCE')
        ax[1].set_ylabel('Relative capacity [%]')
        ax[1].legend()
        return None


if __name__ == '__main__':
    test_case = r"Z:\Provning\EIS\EIS_data\EIS_Cell_714\HYBRIDEIS_TEST1_Cell_714_50_SOC.DTA"
    gamry_data = GamryDataSet(test_case)
    res = gamry_data.alternative_fit(mode='d_war')
    plt.figure()
    #plt.plot(res.best_fit.real, -res.best_fit.imag, label='d_war')
    res_cpe = gamry_data.alternative_fit(mode='std_cpe')
    res_2rc_cpe = gamry_data.alternative_fit(mode='2rc_cpe')
    res_2rc = gamry_data.alternative_fit(mode='2rc')
    res_std = gamry_data.alternative_fit(mode='std')
    plt.plot(res_2rc_cpe.best_fit.real, -res_2rc_cpe.best_fit.imag, label='2RC with cpe')
    #plt.plot(res_cpe.best_fit.real, -res_cpe.best_fit.imag, label='Standard warburg with cpe')
    gamry_data.plot_nyquist()
    diffusion_figure = plt.figure()
    gamry_data.plot_re_z_v_sigma()
    # test_dir = r"Z:\Provning\EIS\EIS_data\EIS_Cell_661"
    test_dir = r"Z:\Provning\EIS\EIS_data\EIS_Cell_714"
    my_test_set = GroupGamryDatasets(test_dir)
    for ds in my_test_set.data_dict:
        res_2rc_cpe = my_test_set.data_dict[ds].alternative_fit(mode='std_cpe')
        plt.figure()
        plt.plot(res_2rc_cpe.best_fit.real, -res_2rc_cpe.best_fit.imag, label=f'Fitted model {ds} FCE')
        my_test_set.data_dict[ds].plot_nyquist()
    fig_dict = my_test_set.plot_parameter_evolution(mode='std_cpe')
    my_test_set.find_diffusivity()
    nyq_fig = my_test_set.plot_set_nyquist()
    lbl_list = ['Fresh cell', '300 FCE', '600 FCE', '900 FCE']
    # for i, item in enumerate(nyq_fig.gca().get_lines()):
    #     item.set_linewidth(2)
    for ln, lbl in zip(nyq_fig.gca().lines, lbl_list):
        ln.set_linewidth(2)
        ln.set_label(lbl)
    nyq_fig.gca().legend(loc='upper left')
    nyq_fig.savefig(r"Z:\Documents\Papers\TeslaLowFrequencyPulse\updated_images\nyquist_plot_cell661_0_300_600_900.pdf")
    # par_figs = my_test_set.plot_parameter_evolution(mode='std')
    # data_set_2rc = my_test_set.fit_set_randles(mode='d_war')
    my_full_set = AllGamryData(r'Z:\Provning\EIS\EIS_data')

    complement_set = AllGamryData(r'Z:\Provning\EIS_data_complement_study')
    my_full_set.plot_filtered_fce(fce_filter=[0, 900], ax_setting='fix_ratio', ind_fig=False, save_fig=True)
    complement_set.plot_filtered_fce(fce_filter=[0], ind_fig=False, save_fig=True)
    """
    eis_cap_fig, ax = plt.subplots(2, 1)
    my_full_set.plot_cap_with_eis(cell_key='Cell_717', fce_filter=[0, 900], plt_canvas=eis_cap_fig)
    my_full_set.plot_cap_with_eis(cell_key='Cell_661', fce_filter=[0, 900], plt_canvas=eis_cap_fig)
    my_full_set.plot_cap_with_eis(cell_key='Cell_707', fce_filter=[0, 900], plt_canvas=eis_cap_fig)
    my_full_set.plot_cap_with_eis(cell_key='Cell_714', fce_filter=[0, 900], plt_canvas=eis_cap_fig)
    eis_cap_fig.savefig(r"Z:\Provning\EIS\Analysis\eis_and_capacity_fig_cell717_661_707_714")
    my_full_set.plot_filtered_fce(ind_fig=True, save_fig=True, ax_setting='fix_ratio')
    my_full_set.plot_filtered_fce(ind_fig=True, save_fig=True, ax_setting='')
    my_full_set.plot_filtered_fce(ind_fig=False, save_fig=True, ax_setting='fix_ratio')
    plt.close('all')
    ocv = my_test_set.find_ocv()
    """
    bol_fig = plt.figure()
    name_map = {
        'Cell_661': ['1s', 'green'],
        'Cell_707': ['16s', 'blue'],
        'Cell_714': ['64s', 'brown'],
        'Cell_717': ['128s', 'orange']
    }
    for k in name_map.keys():
        tmp_ds = my_full_set.gamry_data_dict[k].data_dict['0']
        tmp_ds.plot_nyquist(c=name_map[k][1], ext_label=name_map[k][0])
    bol_fig.savefig(r'Z:\Provning\EIS\Analysis\bol_nyquist_plot.png', dpi=200)
