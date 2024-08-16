import pyvisa
import time
import numpy as np


rm = pyvisa.ResourceManager()
devices = rm.list_resources()

op_file = '/data/entropy-logs/entropy-measurement-1'

instr21 = 'GPIB0::21::INSTR'
instr22 = 'GPIB0::22::INSTR'
instr25 = 'GPIB0::25::INSTR'

obj21 = rm.open_resource(instr21)
obj22 = rm.open_resource(instr22)
obj25 = rm.open_resource(instr25)

obj21.write('END ALWAYS')
obj21.write('NPLC 100')
obj22.write('END ALWAYS')
obj22.write('NPLC 100')
obj21.write('DCI 1')

start_time = time.time()

while time.time() < start_time + 10:
    meas_timestamp = time.time()
    try:
        meas_current = float(obj21.query('DCI?'))
    except:
        print(f'Current measurement failed at {meas_timestamp}')
        meas_current = np.nan

    try:
        meas_voltage = float(obj22.query('DCV?'))
    except:
        print(f'Voltage measurement failed at {meas_timestamp}')
        meas_voltage = np.nan

    try:
        meas_fres = float(obj25.query('MEAS:FRES?'))
    except:
        print(f'Resistance measurement failed at {meas_timestamp}')
        meas_fres = np.nan

    print(f'\rLatest measurement:\n\tCurrent: {meas_current:.2e}A\n\tVoltage:{meas_voltage:.2e}V\n\tResistance:{meas_fres}')
    with open(op_file, 'a+', buffering=1) as f:
        output_data = f'{meas_timestamp}, {meas_voltage}, {meas_current}, {meas_fres}\n'
        f.write(output_data)
    time.sleep(3)
