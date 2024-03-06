import datetime
from pyvisa_interface.pyvisa_device import MeasurementDevice
from pyvisa_interface.prompt_user_input import choice_handler_msmt_setting, choice_handler_device_setting
import pyvisa
import time
import csv
import os


class MeasurementSettings:
    def __init__(self, prompt_user=True):
        self.prompt_bool = prompt_user
        self.data_type = 'voltage'
        self.interval_seconds = 1  # Interval between measurements - 0 for max speed
        self.duration_seconds = 3600 * 12  # Duration of measurement in seconds
        self.dvc_mode = None
        self.msmt_mode = None
        if not self.dvc_mode:
            self._set_device_settings()
        if not self.msmt_mode:
            self._set_msmt_settings()

    def _set_device_settings(self):
        if self.prompt_bool:
            self.dvc_mode = choice_handler_device_setting()
        else:
            self.dvc_mode = 'default'

    def _set_msmt_settings(self):
        if self.prompt_bool:
            self.msmt_mode = choice_handler_msmt_setting()
        else:
            self.msmt_mode = 'default'


class Main:
    def __init__(self):
        prompt_boolean = True
        self.msmt_settings = MeasurementSettings(prompt_user=prompt_boolean)
        self.output_folder = r'C:\Users\lab-admin\Documents\multimeter_measurement'

    def run(self):
        # Find all connected VISA devices
        rm = pyvisa.ResourceManager()
        visa_devices = rm.list_resources()

        # Iterate through each VISA device and create MeasurementDevice instance
        devices = []
        for idx, visa_device in enumerate(visa_devices):
            # Open communication with the instrument
            inst = rm.open_resource(visa_device)

            # Create instance of MeasurementDevice class if it is gpib
            if 'GPIB' in visa_device:
                device = MeasurementDevice(inst)

                # Append the device to the list
                devices.append(device)

        if self.msmt_settings.dvc_mode == 'individual':
            self.measure_individual(devices)
        else:
            self.measure_combined(devices)

    def measure_combined(self, devices):
        # Create output file and initiate
        filename = self._set_filename()
        open(filename, 'w+').close()

        # Call the wrapper function to produce one common csv file for all devices
        self.measure_and_append_to_common_csv(devices, filename)

    def measure_individual(self, devices):
        # Create output file and initiate
        end_time = time.time() + self.msmt_settings.duration_seconds
        fnames = {dvc.instr_name: self._set_filename(dvc) for dvc in devices}
        for fname in fnames.values():
            open(fname, 'w+').close()
        while time.time() < end_time:
            for dvc in devices:
                self.measure_and_append_to_individual_csv(dvc, fnames[dvc.instr_name])

    def measure_and_append_to_common_csv(self, devices, filename):
        with open(filename, 'a+', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            counter = 0
            end_time = time.time() + self.msmt_settings.duration_seconds
            while time.time() < end_time:
                row_data = []
                for device in devices:
                    timestamp, voltage = device.get_voltage(measurement_mode=self.msmt_settings.msmt_mode)
                    row_data.append(timestamp)
                    row_data.append(voltage)
                if counter % 50 == 0:
                    meas_time = self.reformat_timestamp(timestamp)
                    print(f'Voltage at {meas_time} is {row_data[3]:.3f}V and {row_data[1]:.3f}V')
                csv_writer.writerow(row_data)
                time.sleep(self.msmt_settings.interval_seconds)
                counter += 1

    def measure_and_append_to_individual_csv(self, device, filename):
        row_data = []
        with open(filename, 'a+', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            timestamp, voltage = device.get_voltage(measurement_mode=self.msmt_settings.msmt_mode)
            if self.is_even_timing(60, 5):
                meas_time = self.reformat_timestamp(timestamp)
                print(f'Voltage at {meas_time} is {voltage:.3f}V for device {device.instr_name}')
            row_data.append(timestamp)
            row_data.append(voltage)
            csv_writer.writerow(row_data)
            time.sleep(self.msmt_settings.interval_seconds)

    def _set_filename(self, device=None):
        measurement_start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f')[:-3]
        if device:
            filename = os.path.join(self.output_folder, f'{self.msmt_settings.data_type}_data_'
                                                        f'device{device.instr_number}_{measurement_start_time}.csv')
        else:
            filename = os.path.join(self.output_folder, f'{self.msmt_settings.data_type}_data_{measurement_start_time}.csv')
        return filename

    @staticmethod
    def reformat_timestamp(ts):
        return datetime.datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d_%H.%M.%S')

    @staticmethod
    def is_even_ten_minute():
        current_time = datetime.datetime.now()
        # Round down the minute value to the nearest ten
        minute_rounded = current_time.minute // 10 * 10
        # Calculate the nearest even ten minute
        even_ten_minute = current_time.replace(minute=minute_rounded, second=0, microsecond=0)
        # Check if the difference between current time and even ten minute is within 5 seconds
        time_difference = current_time - even_ten_minute
        return abs(time_difference.total_seconds()) <= 5

    @staticmethod
    def is_even_timing(interval_seconds, tolerance_seconds):
        from datetime import timedelta
        current_time = datetime.datetime.now()
        # Calculate the nearest even timing
        nearest_even_timing = current_time - timedelta(seconds=current_time.second % interval_seconds)
        # Check if the difference between current time and nearest even timing is within 5 seconds
        time_difference = current_time - nearest_even_timing
        return abs(time_difference.total_seconds()) <= tolerance_seconds


if __name__ == '__main__':
    main_class = Main()
    main_class.run()
