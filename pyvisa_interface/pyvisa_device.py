import pyvisa
import time
import re
import numpy as np


class MeasurementDevice:
    def __init__(self, visa_obj):
        self.visa_obj = visa_obj
        self.instr_name = visa_obj.resource_name
        self.instr_number = visa_obj.primary_address
        self.set_auto_impedance()

    def set_auto_impedance(self):
        try:
            self.visa_obj.write("INP:IMP:AUTO ON")
        except:
            print(f'Setting automatic impedance failed on {self.instr_name}')

    def get_voltage(self, measurement_mode='default'):
        # Query voltage to instrument with required accuracy
        if measurement_mode == 'fast':
            v = self.measure_voltage_fast()
        elif measurement_mode == 'accurate':
            v = self.measure_voltage_accurate()
        else:
            v = self.measure_voltage_default()
        # Timestamp the measurement last since voltage reading might take a while
        t = self.make_ms_timestamp()
        return t, v

    def get_impedance(self):
        t = self.make_ms_timestamp()
        z = self.measure_impedance()
        return t, z

    def measure_voltage_fast(self):
        # Send command to the instrument to measure voltage
        try:
            self.visa_obj.write("MEASURE:VOLTAGE:DC? 10, 1E-3")
            voltage = float(self.visa_obj.read())
        except pyvisa.errors.VisaIOError:
            print('Measurement timed out before completion, set voltage to NaN')
            voltage = np.nan
        except:
            print('Undefined error, voltage set to NaN')
            voltage = np.nan
        return voltage

    def measure_voltage_accurate(self):
        # Increase timeout of visa_obj to 5000ms to avoid timeout failure
        self.visa_obj.timeout = 5000
        # Send command to the instrument to measure voltage
        try:
            self.visa_obj.write("MEASURE:VOLTAGE:DC? 10, 1E-5")
            voltage = float(self.visa_obj.read())
        except pyvisa.errors.VisaIOError:
            print('Measurement timed out before completion, set voltage to NaN')
            voltage = np.nan
        except:
            print('Undefined error, voltage set to NaN')
            voltage = np.nan
        return voltage

    def measure_voltage_default(self):
        # Send command to the instrument to measure voltage
        try:
            self.visa_obj.write("MEASURE:VOLTAGE:DC? 10, 1E-4")
            voltage = float(self.visa_obj.read())
        except pyvisa.errors.VisaIOError:
            print('Measurement timed out before completion, set voltage to NaN')
            voltage = np.nan
        except:
            print('Undefined error, voltage set to NaN')
            voltage = np.nan
        return voltage

    def measure_impedance(self):
        # Send command to instrument to measure impedance
        try:
            self.visa_obj.write('MEASURE:IMPEDANCE?')
            # Read the response
            impedance = float(self.visa_obj.read())
        except pyvisa.errors.VisaIOError:
            print('Measurement timed out before completion, set impedance to NaN')
            impedance = np.nan
        except:
            print('Undefined error, impedance set to NaN')
            impedance = np.nan
        return impedance

    def get_instrument_number_from_name(self):
        number = int(re.findall(r'(?<=::)(\d+)(?:::INSTR)', self.instr_name)[0])
        return number

    @staticmethod
    def make_ms_timestamp():
        # Returns unix epoch timestamp in milliseconds
        return int(time.time() * 1000)


if __name__ == '__main__':
    from datetime import datetime
    rm = pyvisa.ResourceManager()
    devices = rm.list_resources()

    measurement_devices_dict = {}
    for device in devices:
        try:
            visa_obj = rm.open_resource(device)
            measurement_device = MeasurementDevice(visa_obj)
            measurement_devices_dict[device] = measurement_device
        except Exception as e:
            print(f"Error instantiating MeasurementDevice for {device}: {e}")

    # Filter out only GPIB connections
    pattern = 'GPIB0'
    measurement_devices_dict = {k: dvc for k, dvc in measurement_devices_dict.items() if pattern in k}

    # Run test cases for voltage measurement
    t, v = measurement_devices_dict['GPIB0::25::INSTR'].get_voltage(measurement_mode='default')
    print(f'Tested voltage measurement at {datetime.fromtimestamp(t / 1000):%Y-%m-%d_%H.%M.%S:%f} with V={v}')
    t, v = measurement_devices_dict['GPIB0::25::INSTR'].get_voltage(measurement_mode='fast')
    print(f'Tested voltage measurement at {datetime.fromtimestamp(t / 1000):%Y-%m-%d_%H.%M.%S:%f} with V={v}')
    t, v = measurement_devices_dict['GPIB0::25::INSTR'].get_voltage(measurement_mode='accurate')
    print(f'Tested voltage measurement at {datetime.fromtimestamp(t / 1000):%Y-%m-%d_%H.%M.%S:%f} with V={v}')
