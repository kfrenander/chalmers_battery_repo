def define_neware_renaming(version_key):
    if version_key == 'v80':
        neware_native = ['DataPoint',
                         'Cycle Index',
                         'Step Index',
                         'Step Type',
                         'Time',
                         'Total Time',
                         'Current(μA)',
                         'Current(mA)',
                         'Voltage(V)',
                         'Capacity(mAh)',
                         'Spec. Cap.(mAh/g)',
                         'Chg. Cap.(mAh)',
                         'Chg. Spec. Cap.(mAh/g)',
                         'DChg. Cap.(mAh)',
                         'DChg. Spec. Cap.(mAh/g)',
                         'Energy(Wh)',
                         'Spec. Energy(mWh/g)',
                         'Chg. Energy(Wh)',
                         'Chg. Spec. Energy(mWh/g)',
                         'DChg. Energy(Wh)',
                         'DChg. Spec. Energy(mWh/g)',
                         'Date',
                         'Power(W)',
                         'dQ/dV(mAh/V)',
                         'dQm/dV(mAh/V.g)',
                         'Contact resistance(mΩ)',
                         'Module start-stop switch',
                         'V1',
                         'V2',
                         'Aux. ΔV',
                         'T1',
                         'T2',
                         'Aux. ΔT']
        local_names = ['measurement',
                       'arb_step2',
                       'arb_step1',
                       'mode',
                       'rel_time',
                       'total_time',
                       'curr',
                       'curr',
                       'volt',
                       'cap',
                       'spec_cap',
                       'chrg_cap',
                       'chrg_spec_cap',
                       'dchg_cap',
                       'dchg_spec_cap',
                       'egy',
                       'spec_egy',
                       'chrg_egy',
                       'chrg_spec_egy',
                       'dchg_egy',
                       'dchg_spec_egy',
                       'abs_time',
                       'pwr',
                       'ica',
                       'ica_spec',
                       'contact_resistance',
                       'module_strt_stop',
                       'aux_volt_1',
                       'aux_volt_2',
                       'aux_dv',
                       'aux_T_1',
                       'aux_T_2',
                       'aux_dT']
        return dict(zip(neware_native, local_names))
    elif version_key == 'v76':
        neware_native = [
            'Record number',
            'State',
            'Jump',
            'Cycle',
            'Steps',
            'Current(mA)',
            'Current(A)',
            'Voltage(V)',
            'Capacity(mAh)',
            'Energy(mWh)',
            'Relative Time(h:min:s.ms)',
            'Real time(h:min:s.ms)',
            'Auxiliary channel TU1 T(°C)',
            'Auxiliary Δtemperature',
            'Real time',
            'Auxiliary channel TU1 U(V)',
            'Auxiliary Δpressure'
        ]
        local_names = [
            'Measurement',
            'mode',
            'step',
            'arb_step1',
            'arb_step2',
            'curr',
            'curr',
            'volt',
            'cap',
            'step_egy',
            'rel_time',
            'abs_time',
            'temperature',
            'aux_temp',
            'abs_time',
            'aux_volt',
            'aux_dv'
        ]
        return dict(zip(neware_native, local_names))
    else:
        print('Unknown neware version, not able to return renaming dictionary.')
        return None