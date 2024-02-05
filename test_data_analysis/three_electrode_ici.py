import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz
from PythonScripts.test_data_analysis.tesla_half_cell import gaussianfilterint
from PythonScripts.test_data_analysis.BaseNewareDataClass import BaseNewareData as bnd
from PythonScripts.test_data_analysis.ica_analysis import ica_on_arb_data
import matplotlib as mpl
import os
from PythonScripts.backend_fix import fix_mpl_backend
from scipy.io import savemat
plt.rcParams['axes.grid'] = True
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.titlesize'] = 14
fix_mpl_backend()
# plt.style.use('kelly_colors')
plt.style.use('ml_colors')


def save_df_to_mat(df, fname):
    tmp_to_mat = {n: col.values for n, col in df.items()}
    savemat(fname, tmp_to_mat)
    return None


def characterise_steps(df):
    gb = df.groupby('arb_step2')
    attr = {k: [gb.get_group(k)['abs_time'].iloc[0],
                gb.get_group(k).volt.max(),
                gb.get_group(k).volt.min(),
                gb.get_group(k).curr.mean(),
                gb.get_group(k).cap.abs().max(),
                gb.get_group(k).step_time_float.max(),
                gb.get_group(k)['mode'].mode().values[0],
                k]
            for k in gb.groups} #if gb.get_group(k).step_time_float.max() < 500}
    df_out = pd.DataFrame.from_dict(attr, orient='index',
                                    columns=[
                                        'stp_date',
                                        'maxV',
                                        'minV',
                                        'curr',
                                        'cap',
                                        'step_duration',
                                        'step_mode',
                                        'step_nbr']
                                    )
    df_out['dV'] = df_out['maxV'] - df_out['minV']
    return df_out


def clean_data_no_volt_lim(df):
    chdf = characterise_steps(df)
    chdf.loc[(chdf['step_mode'] == 'CC Chg') & (chdf['step_duration'] == 270), 'ici'] = True
    chdf.loc[(chdf['step_mode'] == 'CC DChg') & (chdf['step_duration'] == 270), 'ici'] = True
    chdf['ici'].fillna(False, inplace=True)
    ici_stps = chdf[chdf.ici]['step_nbr']
    ici_step_dict = {}
    for stp in ici_stps:
        tmp_rng = np.arange(stp-1, stp+1, 1)
        ici_df = df[df.arb_step2.isin(tmp_rng)]
        bnd.sum_idx(ici_df, 'step_time_float')
        ici_step_dict[stp] = ici_df
    return ici_step_dict


def read_neware_v80(fname):
    xl_file = pd.ExcelFile(fname)
    for sh in xl_file.sheet_names:
        if 'record' in sh:
            df = xl_file.parse(sh)
            df.columns = ['measurement', 'arb_step2', 'arb_step1', 'mode', 'rel_time', 'total_time', 'curr',
                          'volt', 'cap', 'spec_cap', 'chrg_cap', 'chrg_spec_cap', 'dchg_cap',
                          'dchg_spec_cap', 'egy', 'spec_egy', 'chrg_egy', 'chrg_spec_egy',
                          'dchg_egy', 'dchg_spec_egy', 'abs_time', 'power', 'ica', 'ica_spec',
                          'contact_resistance', 'module_strt_stop']
            df['step_time'] = pd.to_timedelta(df.rel_time)
            df['abs_time'] = pd.to_datetime(df['abs_time'], format='%Y-%m-%d %H:%M:%S')
            df['float_time'] = (df.abs_time - df.abs_time[0]).astype('timedelta64[s]')
            df['unq_step_nbr'] = df['mode'].shift().fillna(df['mode']).ne(df['mode']).astype(int).cumsum()
    return df


def read_neware_aux_channels(fname):
    xl_file = pd.ExcelFile(fname)
    for sh in xl_file.sheet_names:
        if 'Detail_' in sh:
            df = xl_file.parse(sh)
            df.columns = ['measurement', 'mode', 'step', 'arb_step1', 'arb_step2',
                          'curr', 'volt', 'cap', 'egy', 'rel_time', 'abs_time']
            df['step_time'] = pd.to_timedelta(df.rel_time)
            df['abs_time'] = pd.to_datetime(df['abs_time'], format='%Y-%m-%d %H:%M:%S')
            df['float_time'] = (df.abs_time - df.abs_time[0]).astype('timedelta64[s]').dt.seconds
        elif 'DetailVol' in sh:
            df_vol = xl_file.parse(sh)
            if len(df_vol.columns) == 9:
                df_vol.columns = ['meas', 'mode', 'rel_time', 'real_time', 'aux_un', 'aux_pos', 'aux_neg',
                                  'aux_pressure', 'auxp']
            elif len(df_vol.columns) == 7:
                df_vol.columns = ['meas', 'mode', 'rel_time', 'real_time', 'aux_neg', 'aux_un', 'aux_pressure']
            for col in df_vol.columns:
                if 'aux' in col:
                    df.loc[:, col] = df_vol.loc[:, col]
    return df


def find_rest_step(df):
    import numpy as np
    df.loc[:, 'ici_bool'] = np.nan
    df.loc[df[df['curr'] != 0].index, 'ici_bool'] = False
    df['ici_bool'].fillna(True, inplace=True)
    return df


def find_ica_step(df):
    df['ica_bool'] = False
    chdf = characterise_steps(df)
    ica_steps = chdf[(chdf.dV > 1) & (chdf.step_mode!='CCCV DChg')]['step_nbr']
    ica_index = df[df.arb_step2.isin(ica_steps)].index
    df.loc[ica_index, 'ica_bool'] = True
    return df


def run_ic_dv_analysis(df, potential_series='volt'):
    if not 'ica_bool' in df.columns:
        df = find_ica_step(df)
    x = df[df.ica_bool].mAh
    y = df[df.ica_bool][potential_series]
    ic_dv = pd.DataFrame.from_dict(ica_on_arb_data(x, y, prespan=0.0105))
    ic_dv.columns = ['cap', 'volt', 'ica_gauss', 'dva_gauss']
    ic_dv.loc[:, 'cap'] = ic_dv.loc[:, 'cap'] - ic_dv.loc[:, 'cap'].min()
    return ic_dv


def fit_lin_vs_sqrt_time(df, col):
    use_df = df[(df['step_time_float'] > 1)] # & (df['step_time_float'] <= 5)]
    coeffs = np.polyfit(np.sqrt(use_df['step_time_float']), use_df[col], 1)
    return coeffs


def fit_lin_ocp_slope(df, col):
    use_df = df[df['step_time_float'] > df['step_time_float'].max() - 120]
    fit_tuple = np.polyfit(use_df['step_time_float'], use_df[col], 1, full=True)
    coeffs = fit_tuple[0]
    residual = fit_tuple[1]
    return coeffs, residual


def analysis_rest_step(rdf, cdf):
    col_combined = '\t'.join(rdf.columns)
    if 'aux' in col_combined:
        cols = ['volt', 'aux_pos', 'aux_neg']
        name_dict = {
            'volt': 'full cell',
            'aux_pos': 'positive electrode',
            'aux_neg': 'negative electrode'
        }
        coeff_dict = {name_dict[k]: [*fit_lin_vs_sqrt_time(rdf, k)] for k in cols}
    else:
        cols = ['volt']
        name_dict = {'volt': 'full cell'}
        coeff_dict = {'full cell': [*fit_lin_vs_sqrt_time(rdf, 'volt')]}
    result_df = pd.DataFrame.from_dict(coeff_dict, orient='index', columns=['volt_slope', 'volt_intercept'])
    # for c in cols:
    #     rdf.loc[:, f'{c}_filt'] = gaussianfilterint(rdf.loc[:, 'step_time_float'], rdf.loc[:, c], span=0.05*rdf.shape[0])[1]
    cdf_fin = cdf.last_valid_index()
    rdf_init = rdf.first_valid_index()
    i_stp = cdf.curr.mean() / 1000
    for col in cols:
        coeff_dEdt, res_dEdt = fit_lin_ocp_slope(cdf, col)
        if res_dEdt[0] < 1e-5:
            result_df.loc[name_dict[col], 'ocp_slope'] = coeff_dEdt[0]
            result_df.loc[name_dict[col], 'k'] = abs(-result_df.loc[name_dict[col], 'volt_slope'] / i_stp)
            result_df.loc[name_dict[col], 'r0_rct'] = \
                abs((cdf.loc[cdf_fin, col] - result_df.loc[name_dict[col], 'volt_intercept']) / i_stp)
        else:
            result_df.loc[name_dict[col], 'ocp_slope'] = np.nan
            result_df.loc[name_dict[col], 'k'] = np.nan
            result_df.loc[name_dict[col], 'r0_rct'] = np.nan
        result_df.loc[name_dict[col], 'ocp_residual'] = res_dEdt[0]
        result_df.loc[name_dict[col], 'pOCV'] = rdf.loc[rdf_init, col]
    result_df.loc[:, 'Q'] = cdf.loc[cdf_fin, 'mAh']
    result_df.loc[:, 'V'] = cdf.loc[cdf_fin, 'volt']

    return result_df


def calc_diffusion(df, manufacturer='lifesize'):
    """
    Based on equation 15 in [1] Z. Geng, Y. C. Chien, M. J. Lacey, T. Thiringer, and D. Brandell,
    “Validity of solid-state Li+ diffusion coefficient estimation by electrochemical approaches for lithium-ion batteries,”
    Electrochim. Acta, vol. 404, p. 139727, Feb. 2022.

    :param df:
    :return:
    """
    df = df.reset_index()
    t_p = 310
    if manufacturer=='lifesize':
        if df['index'].unique()[0] == 'positive electrode':
            rp = 4e-6 / 2
        elif df['index'].unique()[0] == 'negative electrode':
            rp = 16e-6 / 2
        else:
            rp = (4 + 16)*1e-6/2
    else:
        if df['index'].unique()[0] == 'positive electrode':
            rp = 13.8e-6 / 2
        elif df['index'].unique()[0] == 'negative electrode':
            rp = 17e-6 / 2
        else:
            rp = (13.8 + 17)*1e-6/2
    df.loc[:, 'D'] = 4 * rp**2 / (9 * np.pi) * df.loc[:, 'ocp_slope']**2 * df.loc[:, 'volt_slope']**-2
    # df.loc[:, 'D_pseudoOCP'] = 4 * rp**2 / (9 * np.pi) * df.loc[:, 'pOCV']**2 * df.loc[:, 'volt_slope']**-2
    for i in range(0, df.shape[0] - 1, 1):
        deltaE = df.loc[i + 1, 'pOCV'] - df.loc[i, 'pOCV']
        dEdT = deltaE / t_p
        df.loc[i, 'D_pOCV'] = 4 * rp**2 / (9 * np.pi) * dEdT**2 * df.loc[i, 'volt_slope']**-2
        df.loc[i, 'dEdt_pOCV'] = dEdT
    return df


def visualise_fit(df, col):
    coeffs = fit_lin_vs_sqrt_time(df, col)
    a, b = [*coeffs]
    # fig = plt.figure()
    sqrtt = np.sqrt(df['step_time_float'])
    plt.plot(sqrtt, df[col], '.', label='Raw data')
    plt.plot(sqrtt, np.poly1d(coeffs)(sqrtt), label='Fitted data')
    plt.axvline(1, linestyle='dashed', linewidth=0.6)
    ax = plt.gca()
    ax.set_xlabel(r'$t^{\frac{1}{2}}\quad /  \quad s^{\frac{1}{2}}$')
    ax.set_ylabel('Volt / V')
    plt.annotate(f'Fit params:\nSlope = {a:.2e}\nIntercept = {b:.4f}', xy=(0.85, 0.5), xycoords='axes fraction')
    ax.legend()
    return fig


def visualise_analysis(res_df, inp_fig=None):
    if inp_fig:
        ax = inp_fig.gca()
    else:
        fig, ax = plt.subplots(2, 3, sharey='col', sharex='row', figsize=(14, 10))
    chrg_pt = res_df[res_df.type == 'chrg']
    dchg_pt = res_df[res_df.type == 'dchg']
    mid_cap = chrg_pt.Q.mean()
    mid_volt = chrg_pt.V.mean()
    ax[0, 0].plot(chrg_pt.Q, chrg_pt.r0_rct, marker='.', color='blue')
    ax[0, 0].plot(dchg_pt.Q, dchg_pt.r0_rct, marker='.', color='orange')
    ax[0, 0].set_xlabel('Charge capacity [mAh]')
    ax[0, 0].set_ylabel(r'$R_{reg}$ [m$\Omega$]')
    ax[0, 1].plot(chrg_pt.Q, chrg_pt.k, marker='.', color='blue')
    ax[0, 1].plot(dchg_pt.Q, dchg_pt.k, marker='.', color='orange')
    ax[0, 1].set_xlabel('Charge capacity [mAh]')
    ax[0, 1].set_ylabel(r'$k_{diff}$ [$m\Omega/s^{\frac{1}{2}}$]')
    ax[1, 0].plot(chrg_pt.V, chrg_pt.r0_rct, marker='.', color='blue')
    ax[1, 0].plot(dchg_pt.V, dchg_pt.r0_rct, marker='.', color='orange')
    ax[1, 0].set_xlabel('Full cell voltage [V]')
    ax[1, 0].set_ylabel(r'$R_{reg}$ [m$\Omega$]')
    ax[1, 1].plot(chrg_pt.V, chrg_pt.k, marker='.', color='blue')
    ax[1, 1].plot(dchg_pt.V, dchg_pt.k, marker='.', color='orange')
    ax[1, 1].set_xlabel('Full cell voltage [V]')
    ax[1, 1].set_ylabel(r'$k_{diff}$ [$m\Omega/s^{\frac{1}{2}}$]')
    ax[0, 2].plot(chrg_pt.Q, chrg_pt.D, marker='.', color='blue')
    ax[0, 2].plot(dchg_pt.Q, dchg_pt.D, marker='.', color='orange')
    ax[0, 2].set_xlabel('Charge capacity [mAh]')
    ax[0, 2].set_ylabel(r'$D$ [$m^2/s$]')
    ax[1, 2].plot(chrg_pt.V, chrg_pt.D, marker='.', color='blue')
    ax[1, 2].plot(dchg_pt.V, dchg_pt.D, marker='.', color='orange')
    ax[1, 2].set_xlabel('Full cell voltage [V]')
    ax[1, 2].set_ylabel(r'$D$ [$m^2/s$]')
    # ax[1, 2].set_ylim(-5e-17, 2e-15)
    plt.tight_layout()
    return fig


def vis_analysis_portrait(res_df, x_mode='cap', param_scale=1):
    if x_mode == 'cap':
        x_label = 'Charge capacity [mAh]'
    elif x_mode == 'soc':
        x_label = 'SOC [-]'
        res_df['Q'] = res_df['Q'] / res_df['Q'].max()
    if param_scale != 1:
        y_suffix = "$\cdot cm^2$]"
    else:
        y_suffix = ']'
    fig, ax = plt.subplots(3, 2, sharey='row', sharex='col', figsize=(9, 15))
    chrg_pt = res_df[res_df.type == 'chrg']
    dchg_pt = res_df[res_df.type == 'dchg']
    mid_cap = chrg_pt.Q.mean()
    mid_volt = chrg_pt.V.mean()
    ax[0, 0].plot(chrg_pt.Q, chrg_pt.r0_rct*param_scale, marker='.', color='blue', label='Charge')
    ax[0, 0].plot(dchg_pt.Q, dchg_pt.r0_rct*param_scale, marker='.', color='orange', label='Discharge')
    ax[0, 0].set_xlabel(x_label)
    ax[0, 0].set_ylabel(r'$R_{reg}$ [m$\Omega$' + f'{y_suffix}')
    ax[0, 0].legend()
    ax[1, 0].plot(chrg_pt.Q, chrg_pt.k*param_scale, marker='.', color='blue')
    ax[1, 0].plot(dchg_pt.Q, dchg_pt.k*param_scale, marker='.', color='orange')
    ax[1, 0].set_xlabel(x_label)
    ax[1, 0].set_ylabel(r'$k_{diff}$ [$m\Omega/s^{\frac{1}{2}}$' + f'{y_suffix}')
    ax[2, 0].plot(chrg_pt.Q, chrg_pt.D, marker='.', color='blue')
    ax[2, 0].plot(dchg_pt.Q, dchg_pt.D, marker='.', color='orange')
    ax[2, 0].set_xlabel(x_label)
    ax[2, 0].set_ylabel(r'$D$ [$m^2/s$]')
    ax[0, 1].plot(chrg_pt.V, chrg_pt.r0_rct*param_scale, marker='.', color='blue')
    ax[0, 1].plot(dchg_pt.V, dchg_pt.r0_rct*param_scale, marker='.', color='orange')
    ax[0, 1].set_xlabel('Full cell voltage [V]')
    ax[0, 1].set_ylabel(r'$R_{reg}$ [m$\Omega$' + f'{y_suffix}')
    ax[1, 1].plot(chrg_pt.V, chrg_pt.k*param_scale, marker='.', color='blue')
    ax[1, 1].plot(dchg_pt.V, dchg_pt.k*param_scale, marker='.', color='orange')
    ax[1, 1].set_xlabel('Full cell voltage [V]')
    ax[1, 1].set_ylabel(r'$k_{diff}$ [$m\Omega/s^{\frac{1}{2}}$' + f'{y_suffix}')
    ax[2, 1].plot(chrg_pt.V, chrg_pt.D, marker='.', color='blue')
    ax[2, 1].plot(dchg_pt.V, dchg_pt.D, marker='.', color='orange')
    ax[2, 1].set_xlabel('Full cell voltage [V]')
    ax[2, 1].set_ylabel(r'$D$ [$m^2/s$]')
    plt.tight_layout()
    return fig


def visualise_multiple_analysis(res_dct):
    fig, ax = plt.subplots(2, 3, sharey='col', sharex='row', figsize=(14, 10))
    for k in res_dct:
        chrg_lab = f'Charge {k}'
        dchg_lab = f'Discharge {k}'
        res_df = res_dct[k]
        chrg_pt = res_df[res_df.type == 'chrg']
        dchg_pt = res_df[res_df.type == 'dchg']
        mid_cap = chrg_pt.Q.mean()
        mid_volt = chrg_pt.V.mean()
        ax[0, 0].plot(chrg_pt.Q, chrg_pt.r0_rct, marker='.', label=chrg_lab)
        ax[0, 0].plot(dchg_pt.Q, dchg_pt.r0_rct, marker='.', label=dchg_lab)
        ax[0, 0].set_xlabel('Charge capacity [mAh]')
        ax[0, 0].set_ylabel(r'$R_{reg}$ [m$\Omega$]')
        ax[0, 0].legend()
        ax[1, 0].plot(chrg_pt.Q, chrg_pt.k, marker='.', label=chrg_lab)
        ax[1, 0].plot(dchg_pt.Q, dchg_pt.k, marker='.', label=dchg_lab)
        ax[1, 0].set_xlabel('Charge capacity [mAh]')
        ax[1, 0].set_ylabel(r'$k_{diff}$ [$m\Omega/s^{\frac{1}{2}}$]')
        ax[1, 0].legend()
        ax[1, 0].plot(chrg_pt.V, chrg_pt.r0_rct, marker='.', label=chrg_lab)
        ax[1, 0].plot(dchg_pt.V, dchg_pt.r0_rct, marker='.', label=dchg_lab)
        ax[1, 0].set_xlabel('Full cell voltage [V]')
        ax[1, 0].set_ylabel(r'$R_{reg}$ [m$\Omega$]')
        ax[1, 0].legend()
        ax[1, 1].plot(chrg_pt.V, chrg_pt.k, marker='.', label=chrg_lab)
        ax[1, 1].plot(dchg_pt.V, dchg_pt.k, marker='.', label=dchg_lab)
        ax[1, 1].set_xlabel('Full cell voltage [V]')
        ax[1, 1].set_ylabel(r'$k_{diff}$ [$m\Omega/s^{\frac{1}{2}}$]')
        ax[1, 1].legend()
        ax[0, 2].plot(chrg_pt.Q, chrg_pt.D, marker='.', label=chrg_lab)
        ax[0, 2].plot(dchg_pt.Q, dchg_pt.D, marker='.', label=dchg_lab)
        ax[0, 2].set_xlabel('Charge capacity [mAh]')
        ax[0, 2].set_ylabel(r'$D$ [$m^2/s$]')
        ax[0, 2].legend()
        ax[1, 2].plot(chrg_pt.V, chrg_pt.D, marker='.', label=chrg_lab)
        ax[1, 2].plot(dchg_pt.V, dchg_pt.D, marker='.', label=dchg_lab)
        ax[1, 2].set_xlabel('Full cell voltage [V]')
        ax[1, 2].set_ylabel(r'$D$ [$m^2/s$]')
        ax[1, 2].legend()
    # ax[1, 2].set_ylim(-5e-17, 2e-15)
    plt.tight_layout()
    return fig


def vis_multiple_analysis_portrait(res_dct, x_mode='cap', param_scale=1):
    if x_mode == 'cap':
        x_label = 'Charge capacity [mAh]'
        x_col = 'Q'
    elif x_mode == 'soc':
        x_label = 'SOC [-]'
        x_col = 'soc'
    if isinstance(param_scale, dict):
        y_suffix = "$\cdot cm^2]$"
        scale_dict_bool = True
    elif type(param_scale) == int or type(param_scale) == float:
        scale_dict_bool = False
        if param_scale == 1:
            y_suffix = ']'
        else:
            y_suffix = "$\cdot cm^2]$"
    else:
        print('Unknown scaling provided, exiting')
        return None
    fig, ax = plt.subplots(3, 2, sharey='row', sharex='col', figsize=(9, 15))
    for k in res_dct:
        if scale_dict_bool:
            scale_ = param_scale[k]
        else:
            scale_ = param_scale
        chrg_lab = f'Charge {k.replace("_", " ").replace(".pkl", "")}'
        dchg_lab = f'Discharge {k.replace("_", " ").replace(".pkl", "")}'
        res_df = res_dct[k]
        res_df['soc'] = res_df['Q'] / res_df['Q'].max()
        chrg_pt = res_df[res_df.type == 'chrg']
        dchg_pt = res_df[res_df.type == 'dchg']
        mid_cap = chrg_pt.Q.mean()
        mid_volt = chrg_pt.V.mean()
        ax[0, 0].plot(chrg_pt[x_col], chrg_pt.r0_rct*scale_, marker='.', label=chrg_lab)
        ax[0, 0].plot(dchg_pt[x_col], dchg_pt.r0_rct*scale_, marker='.', label=dchg_lab)
        ax[0, 0].set_xlabel(x_label)
        ax[0, 0].set_ylabel(r'$R_{reg}$ [m$\Omega$' + f'{y_suffix}')
        ax[0, 0].legend()
        ax[1, 0].plot(chrg_pt[x_col], chrg_pt.k*scale_, marker='.', label=chrg_lab)
        ax[1, 0].plot(dchg_pt[x_col], dchg_pt.k*scale_, marker='.', label=dchg_lab)
        ax[1, 0].set_xlabel(x_label)
        ax[1, 0].set_ylabel(r'$k_{diff}$ [$m\Omega/s^{\frac{1}{2}}$' + f'{y_suffix}')
        ax[1, 0].legend()
        ax[2, 0].plot(chrg_pt[x_col], chrg_pt.D, marker='.', label=chrg_lab)
        ax[2, 0].plot(dchg_pt[x_col], dchg_pt.D, marker='.', label=dchg_lab)
        ax[2, 0].set_xlabel(x_label)
        ax[2, 0].set_ylabel(r'$D$ [$m^2/s$]')
        ax[2, 0].legend()
        ax[0, 1].plot(chrg_pt.V, chrg_pt.r0_rct*scale_, marker='.', label=chrg_lab)
        ax[0, 1].plot(dchg_pt.V, dchg_pt.r0_rct*scale_, marker='.', label=dchg_lab)
        ax[0, 1].set_xlabel('Full cell voltage [V]')
        ax[0, 1].set_ylabel(r'$R_{reg}$ [m$\Omega$' + f'{y_suffix}')
        ax[0, 1].legend()
        ax[1, 1].plot(chrg_pt.V, chrg_pt.k*scale_, marker='.', label=chrg_lab)
        ax[1, 1].plot(dchg_pt.V, dchg_pt.k*scale_, marker='.', label=dchg_lab)
        ax[1, 1].set_xlabel('Full cell voltage [V]')
        ax[1, 1].set_ylabel(r'$k_{diff}$ [$m\Omega/s^{\frac{1}{2}}$' + f'{y_suffix}')
        ax[1, 1].legend()
        ax[2, 1].plot(chrg_pt.V, chrg_pt.D, marker='.', label=chrg_lab)
        ax[2, 1].plot(dchg_pt.V, dchg_pt.D, marker='.', label=dchg_lab)
        ax[2, 1].set_xlabel('Full cell voltage [V]')
        ax[2, 1].set_ylabel(r'$D$ [$m^2/s$]')
        ax[2, 1].legend()
    # ax[1, 2].set_ylim(-5e-17, 2e-15)
    plt.tight_layout()
    return fig


def visualise_diffusion_coeff(df):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    chrg_pt = df[df.type == 'chrg']
    dchg_pt = df[df.type == 'dchg']
    mid_cap = chrg_pt.Q.mean()
    ax.semilogy(chrg_pt.Q, chrg_pt.D, marker='.', color='blue')
    ax.semilogy(dchg_pt.Q, dchg_pt.D, marker='.', color='orange')
    ax.set_xlabel('Charge capacity [mAh]')
    ax.set_ylabel(r'$D$ [$m^2/s$]')
    ax.set_ylim(1e-19, df.D.mean()*10)
    ax.text(mid_cap, df.D.mean(), f'Mean diffusivity: \n{df.D.mean():.3e}', fontsize=14)
    return fig


def compare_diffusion_coeff_methods(df):
    fig, ax = plt.subplots(2, 1, figsize=(14, 10))
    chrg_pt = df[df.type == 'chrg']
    dchg_pt = df[df.type == 'dchg']
    mid_cap = chrg_pt.Q.mean()
    ax[0].semilogy(chrg_pt.Q, chrg_pt.D, marker='.', color='blue', label='linear fit')
    ax[0].semilogy(chrg_pt.Q, chrg_pt.D_pOCV, marker='*', color='indianred', label='pOCV fit', linestyle='dashed')
    ax[1].semilogy(dchg_pt.Q, dchg_pt.D, marker='.', color='blue', label='linear fit - dchg')
    ax[1].semilogy(dchg_pt.Q, dchg_pt.D_pOCV, marker='.', color='indianred', label='pOCV fit - dchg', linestyle='dashed')
    ax[0].set_xlabel('Charge capacity [mAh]')
    ax[0].set_ylabel(r'$D$ [$m^2/s$]')
    ax[1].set_ylabel(r'$D$ [$m^2/s$]')
    ax[0].set_title('Charge')
    ax[1].set_title('Discharge')
    # ax[0].set_ylim(1e-20, chrg_pt.D.mean() * 2)
    # ax[1].set_ylim(1e-20, dchg_pt.D.mean() * 2)
    # ax[0].text(mid_cap, chrg_pt.D.mean() / 2,
    #         f'Mean diffusivity linear fit: \n{chrg_pt.D.mean():.3e}\nMean pOCV fit: \n{chrg_pt.D_pOCV.mean():.3e}',
    #         fontsize=14)
    # ax[1].text(mid_cap, dchg_pt.D.mean() / 2,
    #         f'Mean diffusivity linear fit: \n{dchg_pt.D.mean():.3e}\nMean pOCV fit: \n{dchg_pt.D_pOCV.mean():.3e}',
    #         fontsize=14)
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    return fig


def compare_parameters_eol_bol(inp_dct, max_cap=1):
    key_params = ['D', 'k', 'r0_rct']
    op_df = pd.DataFrame()
    curr_cases = ['chrg', 'dchg']
    for k in inp_dct:
        case_name = os.path.splitext(k)[0].replace('_', ' ')
        df = inp_dct[k]
        for c in curr_cases:
            case_df = df.groupby(by='type').get_group(c)
            avg_list = [case_df.loc[:, p].mean()*max_cap for p in key_params]
            tmp_df = pd.DataFrame(data=avg_list, columns=[f'{case_name} {c}'], index=key_params)
            op_df = pd.concat([op_df, tmp_df], axis='columns')
    return op_df


if __name__ == '__main__':
    two_electrode = {
        'cell1': r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\cell1.pkl",
        'cell2': r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\cell2.pkl",
        'cell3': r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\cell3.pkl"
    }
    three_electrode = {
        "cell5": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\cell5.pkl",
        "cell6": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\cell6.pkl",
        "cell7": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\cell7.pkl"
    }
    cidetec_cells = {
        "cell1": r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\BatteryCharacterization\SEC_CellbuildingAndPrototypes\LabCelltests\CidetecLabcells\RawData\CidetecLabcells\240072-1-1-2818575844.xlsx",
        "cell2": r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\BatteryCharacterization\SEC_CellbuildingAndPrototypes\LabCelltests\CidetecLabcells\RawData\CidetecLabcells\240072-1-2-2818575845.xlsx",
        "cell3": r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\BatteryCharacterization\SEC_CellbuildingAndPrototypes\LabCelltests\CidetecLabcells\RawData\CidetecLabcells\240072-1-3-2818575844.xlsx",
        "cell4": r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\BatteryCharacterization\SEC_CellbuildingAndPrototypes\LabCelltests\CidetecLabcells\RawData\CidetecLabcells\240072-1-4-2818575844.xlsx"
    }
    pat_cell_lifesize = {
        "cell1": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\pat_core_cell\cell1.pkl",
        "cell2": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\pat_core_cell\cell2.pkl",
        "cell3": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\pat_core_cell\cell3.pkl"
    }
    pat_cell_cidetech = {
        "cell1": r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\BatteryCharacterization\SEC_CellbuildingAndPrototypes\CellTesting\PAT_cell_ici\Cidetech\cell2.pkl",
        "cell2": r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\BatteryCharacterization\SEC_CellbuildingAndPrototypes\CellTesting\PAT_cell_ici\Cidetech\cell3.pkl",
        "cell3": r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\BatteryCharacterization\SEC_CellbuildingAndPrototypes\CellTesting\PAT_cell_ici\Cidetech\cell4.pkl",
        "cell4": r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\BatteryCharacterization\SEC_CellbuildingAndPrototypes\CellTesting\PAT_cell_ici\Cidetech\cell1.pkl"
         }
    eol_cidetec = {
        "cell1": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\Cidetech\240095-1-8-2818575202.xlsx",
        "cell2": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\Cidetech\240095-1-1-2818575210.xlsx",
        "cell4": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\Cidetech\240095-1-8-2818575210.xlsx",
        "cell5": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\Cidetech\240095-1-1-2818575212.xlsx",
        "cell6": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\Cidetech\240095-1-8-2818575212.xlsx"
    }
    eol_lifesize_thr_el = {
        "cell5": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\eol_ici\cell5.pkl",
        "cell6": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\eol_ici\Cell6_merged.pkl"
    }
    eol_lifesize_std_el = {
        "cell1": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\eol_ici\Cell1_merged.pkl",
        "cell2": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\eol_ici\cell2.pkl",
        "cell3": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\eol_ici\cell3.pkl"
    }
    eol_lifesize_comb = {
        "cell1": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\eol_ici\Cell1_merged.pkl",
        "cell2": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\eol_ici\cell2.pkl",
        "cell3": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\eol_ici\cell3.pkl",
        "cell5": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\eol_ici\cell5.pkl",
        "cell6": r"\\sol.ita.chalmers.se\groups\batt_lab_data\LifeSize_Cidetech\LifeSize\eol_ici\Cell6_merged.pkl"
    }

    a_pat_cell = 0.9**2*np.pi
    a_ls_pouch = 20*4.5*5.8
    a_ct_pouch = 12*9.8*5.9

    fresh_analysis_bool = 1
    if fresh_analysis_bool:
        two_df = {k: pd.read_pickle(two_electrode[k]) for k in two_electrode}
        thr_df = {k: pd.read_pickle(three_electrode[k]) for k in three_electrode}
        eol_thr_df = {k: pd.read_pickle(eol_lifesize_thr_el[k]) for k in eol_lifesize_thr_el}
        eol_ls_dct = {k: pd.read_pickle(eol_lifesize_std_el[k]) for k in eol_lifesize_std_el}
        eol_ls_comb = {k: pd.read_pickle(eol_lifesize_comb[k]) for k in eol_lifesize_comb}
        pat_cell_ls_dict = {k: pd.read_pickle(pat_cell_lifesize[k]) for k in pat_cell_lifesize}
        pat_cell_cide_dict = {k: pd.read_pickle(pat_cell_cidetech[k]) for k in pat_cell_cidetech}
        # cidetec_dict = {k: read_neware_v80(cidetec_cells[k]) for k in cidetec_cells}
        cidetec_dict_eol = {k: read_neware_aux_channels(eol_cidetec[k]) for k in eol_cidetec}


        color_lookup = {
            'chrg': 'red',
            'dchg': 'blue'
        }
        pouch_cell_bool = 1
        cell_maker = 'cidetech'
        for cell_num in ['1', '2', '4', '5']:  # ['1', '2', '3']
            if pouch_cell_bool:
                if cell_maker == 'lifesize':
                    # cell_num = '6'
                    cell_area = 20*4.5*5.8
                    op_folder = r"Z:\LifeSize_Cidetech\lifesize\bol"
                    df = two_df[f'cell{cell_num}']
                    file_name = f'lifesize_eol_cell{cell_num}'
                elif cell_maker == 'cidetech':
                    # cell_num = '1'
                    cell_area = 12*5.9*9.8
                    op_folder = r"Z:\LifeSize_Cidetech\cidetech\eol"
                    df = cidetec_dict_eol[f'cell{cell_num}']
                    file_name = f'cidetech_eol_cell{cell_num}'
                else:
                    print('Unknown manufacturer. \nPlease state \'lifesize\' or \'cidetech\'')
            else:
                curr_multiplier = 1000
                cell_area = 0.9**2*np.pi
                if cell_maker == 'lifesize':
                    op_folder = r"Z:\LifeSize_Cidetech\lifesize\bol"
                    df = pat_cell_ls_dict[f'cell{cell_num}']
                    if df.curr.max() < 5e-3:
                        df.loc[:, 'curr'] = df.loc[:, 'curr'] * curr_multiplier
                    file_name = f'lifesize_patcore_cell{cell_num}'
                elif cell_maker=='cidetech':
                    op_folder = r"Z:\LifeSize_Cidetech\cidetech\bol"
                    df = pat_cell_cide_dict[f'cell{cell_num}']
                    if df.curr.max() < 5e-3:
                        df.loc[:, 'curr'] = df.loc[:, 'curr'] * curr_multiplier
                    file_name = f'cidetech_patcore_cell{cell_num}'
            if not os.path.isdir(op_folder):
                os.mkdir(op_folder)
            df = find_rest_step(df)
            df.loc[:, 'step_time_float'] = pd.to_timedelta(df.step_time).astype('timedelta64[ms]') / 1000
            df.loc[:, 'mAh'] = cumtrapz(df.curr, df.float_time / 3600, initial=0)
            df.loc[:, 'mAh'] = df.loc[:, 'mAh'] - df.mAh.min()
            fig, ax = plt.subplots(1, 1)
            aux_bool = any('aux' in word for word in df.columns)
            if aux_bool and not 'aux_pos' in df.columns:
                df['aux_pos'] = df['volt'] + df['aux_neg']
            chdf = characterise_steps(df)
            bol_bool = 300 in chdf.step_duration.unique()

            if aux_bool:
                ax.plot(df.float_time, df.volt, label='Cell voltage')
                ax.plot(df.float_time, df.aux_neg, label='Neg electrode')
                ax.plot(df.float_time, df.aux_pos, label='Pos electrode')
            else:
                ax.plot(df.float_time, df.volt, label='Cell voltage')
            gb_stp = df[df.ici_bool].groupby(by='arb_step2')
            ici_stp_dct = {k: gb_stp.get_group(k) for k in gb_stp.groups
                           if gb_stp.get_group(k).step_time.max() < dt.timedelta(seconds=30)}
            if bol_bool:
                gb_curr = df[~df.ici_bool].groupby(by='arb_step2')
                ici_pulse_dct = {k: gb_curr.get_group(k) for k in gb_curr.groups
                                 if gb_curr.get_group(k).step_time.max() == dt.timedelta(seconds=300)}
                soh_string = 'bol'
            else:
                ici_pulse_dct = clean_data_no_volt_lim(df)
                soh_string = 'eol'
            dct = {}
            neg_df = pd.DataFrame()
            pos_df = pd.DataFrame()
            fc_df = pd.DataFrame()
            for k in ici_pulse_dct:
                rdf = df[df.arb_step2 == k + 1]
                if ici_pulse_dct[k].curr.mean() > 0:
                    tp = 'chrg'
                else:
                    tp = 'dchg'
                result_df = analysis_rest_step(rdf, ici_pulse_dct[k])
                result_df.loc[:, 'type'] = tp
                if aux_bool:
                    neg_df = pd.concat([neg_df, result_df.loc['negative electrode', :].to_frame().T], ignore_index=True)
                    pos_df = pd.concat([pos_df, result_df.loc['positive electrode', :].to_frame().T], ignore_index=True)
                    # neg_df = neg_df.append(result_df.loc['negative electrode', :])
                    # pos_df = pos_df.append(result_df.loc['positive electrode', :])
                # fc_df = fc_df.append(result_df.loc['full cell', :])
                fc_df = pd.concat([fc_df, result_df.loc['full cell', :].to_frame().T], ignore_index=True)
                dct[f'{ici_pulse_dct[k].volt.max():.3f}'] = result_df

            # for k in ici_stp_dct:
            #     cdf = df[df.arb_step2 == k - 1]
            #     if cdf.curr.mean() > 0:
            #         tp = 'chrg'
            #     else:
            #         tp = 'dchg'
            #     result_df = analysis_rest_step(ici_stp_dct[k], cdf)
            #     result_df.loc[:, 'type'] = tp
            #     if aux_bool:
            #         neg_df = pd.concat([neg_df, result_df.loc['negative electrode', :].to_frame().T], ignore_index=True)
            #         pos_df = pd.concat([pos_df, result_df.loc['positive electrode', :].to_frame().T], ignore_index=True)
            #         # neg_df = neg_df.append(result_df.loc['negative electrode', :])
            #         # pos_df = pos_df.append(result_df.loc['positive electrode', :])
            #     # fc_df = fc_df.append(result_df.loc['full cell', :])
            #     fc_df = pd.concat([fc_df, result_df.loc['full cell', :].to_frame().T], ignore_index=True)
            #     dct[f'{cdf.volt.max():.3f}'] = result_df

            fc_df = calc_diffusion(fc_df, manufacturer=cell_maker)
            if not os.path.isdir(os.path.join(op_folder, 'df_dir')):
                os.mkdir(os.path.join(op_folder, 'df_dir'))
            fc_df.to_pickle(os.path.join(op_folder, 'df_dir', f'fc_df_cell{file_name}_{soh_string}.pkl'))
            fc_df.to_csv(os.path.join(op_folder, 'df_dir', f'fc_df_cell{file_name}_{soh_string}.csv'), index=False)
            result_dict_fc = {
                f'cell{cell_num}_{soh_string}': fc_df
            }
            fig_fc = vis_analysis_portrait(fc_df, param_scale=cell_area)
            fig_fc.savefig(os.path.join(op_folder, f"full_cell_summary_{file_name}_portrait.png"))
            full_cell_D_comp = compare_diffusion_coeff_methods(fc_df)
            if aux_bool:
                neg_df = calc_diffusion(neg_df, manufacturer=cell_maker)
                pos_df = calc_diffusion(pos_df, manufacturer=cell_maker)
                neg_df.to_pickle(os.path.join(op_folder, 'df_dir', f'neg_df_cell{file_name}_{soh_string}.pkl'))
                neg_df.to_csv(os.path.join(op_folder, 'df_dir', f'neg_df_cell{file_name}_{soh_string}.csv'), index=False)
                pos_df.to_pickle(os.path.join(op_folder, 'df_dir', f'pos_df_cell{file_name}_{soh_string}.pkl'))
                pos_df.to_csv(os.path.join(op_folder, 'df_dir', f'pos_df_cell{file_name}_{soh_string}.csv'), index=False)
                result_dict_neg = {
                    f'cell{cell_num}_{soh_string}': neg_df
                }
                result_dict_pos = {
                    f'cell{cell_num}_{soh_string}': pos_df
                }
                fig_neg = vis_analysis_portrait(neg_df, param_scale=cell_area)
                fig_neg.savefig(rf"Z:\LifeSize_Cidetech\negative_electrode_summary_{file_name}_portrait.png")
                fig_pos = vis_analysis_portrait(pos_df, param_scale=cell_area)
                fig_pos.savefig(rf"Z:\LifeSize_Cidetech\positive_electrode_summary_{file_name}_portrait.png")
                negative_electrode_D_comp = compare_diffusion_coeff_methods(neg_df)
                positive_electrode_D_comp = compare_diffusion_coeff_methods(pos_df)
            print(f'Average diffusive resistance in full cell {file_name} is k={fc_df.k.mean():.2e}')
            key_params = ['D', 'k', 'r0_rct']
            res_str_list = [f'{p}_avg = {fc_df.loc[:, p].mean()*cell_area:.2e} w/ cell area scaling' for p in key_params]
            with open(os.path.join(op_folder, f'key_results_{file_name}_scaled_w_area.txt'), mode='wt', encoding='utf-8') as my_file:
                my_file.write('\n'.join(res_str_list))
            plt.close('all')
    else:
        op_folder = r"Z:\LifeSize_Cidetech\lifesize\custom"
        df_dir = os.walk(os.path.join(op_folder, 'df_dir'))
        fc_dict = {}
        neg_dict = {}
        pos_dict = {}
        for root, dir, files in df_dir:
            for f in files:
                if 'cell' in f and 'pkl' in f:
                    fc_dict[f.replace('fc_df_', '')] = pd.read_pickle(os.path.join(root, f))
                elif 'neg' in f and 'pkl' in f:
                    neg_dict[f.replace('fc_df_', '')] = pd.read_pickle(os.path.join(root, f))
                elif 'pos' in f and 'pkl' in f:
                    pos_dict[f.replace('fc_df_', '')] = pd.read_pickle(os.path.join(root, f))
        scale_dict = dict.fromkeys(fc_dict.keys(), 1)
