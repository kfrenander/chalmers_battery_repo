import pyvisa
import time


rm = pyvisa.ResourceManager()
devices = rm.list_resources()

instr21 = 'GPIB0::21::INSTR'
instr22 = 'GPIB0::22::INSTR'
instr25 = 'GPIB0::25::INSTR'

obj21 = rm.open_resource(instr21)
obj22 = rm.open_resource(instr22)
obj25 = rm.open_resource(instr25)

obj21.write('END ALWAYS')
obj22.write('END ALWAYS')

meas_curren = obj21.query('DCI?')
meas_voltage = obj22.query('DCV?')
print(f'\rLatest measurement: {meas_curren} and {meas_voltage}')
