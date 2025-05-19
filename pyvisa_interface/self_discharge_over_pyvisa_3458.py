import pyvisa
import time
import numpy as np


rm = pyvisa.ResourceManager()
devices = rm.list_resources()
print(devices)

op_file = '/data/self-discharge-logs/self-discharge-measurement-1'

instr21 = 'GPIB0::21::INSTR'
instr22 = 'GPIB0::22::INSTR'
instr25 = 'GPIB0::25::INSTR'

obj21 = rm.open_resource(instr21)
obj22 = rm.open_resource(instr22)

obj21.write('END ALWAYS')
obj21.write('NPLC 100')
obj22.write('END ALWAYS')
obj22.write('NPLC 100')


start_time = time.time()

while 1:
    meas_timestamp = time.time()
    try:
        meas_voltage = float(obj21.query('DCV?'))
    except:
        print(f'Current measurement failed at {meas_timestamp}')
        meas_voltage = np.nan

    try:
        meas_fres = float(obj22.query('OHMF?'))
    except:
        print(f'Voltage measurement failed at {meas_timestamp}')
        meas_fres = np.nan

    formatted_time = time.strftime("%y-%m-%d_%H.%M", time.localtime(meas_timestamp))
    print(f'\rLatest measurement at {formatted_time}:\n\tVoltage:{meas_voltage:.7f}V\n\tResistance:{meas_fres}')
    with open(op_file, 'a+', buffering=1) as f:
        output_data = f'{meas_timestamp}, {meas_voltage}, {meas_fres}\n'
        f.write(output_data)
    time.sleep(30)
