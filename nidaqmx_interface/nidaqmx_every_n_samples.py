import pprint
import queue
import nidaqmx
from nidaqmx.constants import AcquisitionType
import time
import json
from datetime import datetime
import os
import csv
import numpy as np
import threading


class NIDAQmxDevice:
    def __init__(self, dev):
        self.dev_type = dev.product_type
        self.serial_num = dev.serial_num
        self.product_num = dev.product_num
        self.channels = None
        self.find_ai_channels(dev)

    def get_device_info(self):
        return {
            "device_type": self.dev_type,
            "serial_number": self.serial_num,
            "product_number": self.product_num,
            "analog_input_channels": self.channels
        }

    def find_ai_channels(self, dev):
        if dev.ai_physical_chans:
            self.channels = [ai.name for ai in dev.ai_physical_chans]
        else:
            self.channels = []


class NIDAQmxManager:
    def __init__(self):
        self.system = nidaqmx.system.System.local()
        self.devices = []

    def list_available_devices(self):
        """List all available NI-DAQmx devices."""
        devices = self.system.devices
        if devices:
            print("Available NI-DAQmx devices by name:")
            for device in devices:
                print(f"  - {device.name}")
        else:
            print("No NI-DAQmx devices found.")

    def list_available_device_types(self):
        """List all available NI-DAQmx devices."""
        devices = self.system.devices
        if devices:
            print("Available NI-DAQmx device types:")
            for device in devices:
                print(f"  - {device.product_type}")
        else:
            print("No NI-DAQmx devices found.")

    def reset_devices(self):
        """Reset all available NI-DAQmx devices."""
        devices = self.system.devices
        if devices:
            print("Resetting NI-DAQmx Devices:")
            for device in devices:
                device.reset_device()
                print(f"  - Reset {device.name}")
        else:
            print("No NI-DAQmx devices found to reset.")

    def list_analog_input_channels(self):
        """List all available analog input channels."""
        devices = self.system.devices
        if devices:
            print("Available Analog Input Channels:")
            for device in devices:
                for ai_physical_channel in device.ai_physical_chans:
                    print(f"  - Device: {device.product_type}, Channel: {ai_physical_channel}")
        else:
            print("No NI-DAQmx devices found.")

    def take_channel_readings(self, num_readings=10):
        """Take readings from each channel."""
        readings_dict = {}
        devices = self.system.devices
        if devices:
            with nidaqmx.Task() as task:
                print("Taking Channel Readings:")
                for device in devices:
                    for ai_physical_channel in device.ai_physical_chans:
                        try:
                            task.ai_channels.add_ai_voltage_chan(f"{ai_physical_channel.name}")
                            print(f"  - Device: {device.product_type}, Channel: {ai_physical_channel.name}")
                            readings_dict[ai_physical_channel.name] = None
                        except:
                            print(f'No reading for {ai_physical_channel}')
                task.timing.cfg_samp_clk_timing(rate=1000, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
                data = np.array(task.read(number_of_samples_per_channel=num_readings))
            for nm, dta in zip(readings_dict.keys(), np.mean(data, axis=1)):
                readings_dict[nm] = dta
        else:
            print("No NI-DAQmx devices found.")
        return readings_dict

    def check_reasonable_readings(self, threshold=0., num_readings=10):
        chnl_readout = self.take_channel_readings(num_readings=num_readings)
        reasonable_channels = []
        for k, val in chnl_readout.items():
            if val > threshold:
                reasonable_channels.append(k)
                print(f"  - Channel: {k} passed to logging.")
        print(f"A total of {len(reasonable_channels)} channels show reasonable data and will be logged.")
        return reasonable_channels

    def store_device_information(self, device_output_filename):
        self.devices = [NIDAQmxDevice(dev) for dev in self.system.devices]
        with open(device_output_filename, 'w+') as file:
            for idx, dev in enumerate(self.devices, start=1):
                info_dict = dev.get_device_info()
                file.write(f'Device #{idx}\n')
                for key, value in info_dict.items():
                    file.write(f'\t{key}: {value}\n')
                file.write('\n')


class DataAcquisitionSettings:
    def __init__(self, channel_names, sampling_rate, output_rate, duration, filename, output_file, nbr_of_digits=7):
        self.channel_names = channel_names
        self.nbr_of_channels = int(len(channel_names))
        self.sampling_rate = sampling_rate
        self.output_rate = output_rate
        self.n_samples = int(self.sampling_rate // self.output_rate)
        self.duration = duration
        self.filename = filename
        self.output_file = output_file
        self.nbr_of_digits = nbr_of_digits

    def to_json(self):
        settings_dict = {
            "channel_name": self.channel_names,
            "sampling_rate": self.sampling_rate,
            "output_rate": self.output_rate,
            "duration": self.duration,
            "filename": self.filename
        }
        with open(self.output_file, "w") as json_file:
            json.dump(settings_dict, json_file, indent=4)


class DataAcquisition:
    def __init__(self, settings):
        self.settings = settings
        self.data_queue = queue.Queue()
        self.time_queue = queue.Queue()
        self.data = []
        self.start_time = time.time()
        self.last_log_time = 0

    def acquire_data(self):
        try:
            with nidaqmx.Task() as task:
                try:
                    for channel_name in self.settings.channel_names:
                        task.ai_channels.add_ai_voltage_chan(channel_name, min_val=-10.0, max_val=10.0)
                    task.timing.cfg_samp_clk_timing(rate=self.settings.sampling_rate,
                                                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

                    def callback(task_handle, every_n_samples_event_type, number_of_samples, callback_data):
                        """Callback function for reading signals."""
                        if task.is_task_done():
                            return 1
                        else:
                            tmp_data = task.read(number_of_samples_per_channel=self.settings.n_samples)
                            self.data_queue.put(tmp_data)
                            self.time_queue.put(time.time())
                            return 0

                    task.register_every_n_samples_acquired_into_buffer_event(self.settings.n_samples, callback)

                    task.start()

                    while time.time() - self.start_time < self.settings.duration:
                        time.sleep(1 / (10 * self.settings.output_rate))

                finally:
                    task.stop()
        except KeyboardInterrupt:
            print('KeyboardInterrupt detected, exiting')

    def average_data(self):
        try:
            with open(self.settings.filename, 'a+', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['time', *self.settings.channel_names])

                data_accumulator = np.empty((self.settings.nbr_of_channels, 20000))

                while time.time() - self.start_time < self.settings.duration:
                    try:
                        data = self.data_queue.get(timeout=0.5)
                        timestamp = self.time_queue.get(timeout=0.5)  # Non-blocking queue read with a timeout
                    except queue.Empty:
                        continue  # Continue if no data is available

                    data_accumulator[:, :len(data[0])] = data
                    samples_accumulated = np.count_nonzero(~np.isnan(data_accumulator))
                    if samples_accumulated >= self.settings.nbr_of_channels*int(self.settings.sampling_rate /
                                                                                self.settings.output_rate):
                        data_accumulator = data_accumulator[:, :len(data[0])]
                        filtered_data = np.mean(data_accumulator, axis=1)
                        output_data_row = [timestamp]
                        for val in filtered_data:
                            output_data_row.append(round(val, self.settings.nbr_of_digits))
                        csv_writer.writerow(output_data_row)
                        if timestamp - self.last_log_time >= 300:
                            ts_human_readable = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                            self.display_latest_measurement(ts_human_readable, filtered_data)
                            self.last_log_time = timestamp
                            csvfile.flush()
                        data_accumulator = np.empty((self.settings.nbr_of_channels, 20000))
        except KeyboardInterrupt:
            print('KeyboardInterrupt detected, exiting')

    @staticmethod
    def display_latest_measurement(timestamp, data):
        formatted_data = [f'Voltage{n + 1}={val:.4f}' for n, val in enumerate(data)]
        print(f"Latest Measurement at {timestamp}: {', '.join(formatted_data)}")

    def start_acquisition_and_averaging(self):
        try:
            self.settings.to_json()
            acquisition_thread = threading.Thread(target=self.acquire_data)
            averaging_thread = threading.Thread(target=self.average_data)
            acquisition_thread.start()
            averaging_thread.start()

            # Wait for threads to finish
            acquisition_thread.join()
            averaging_thread.join()

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Exiting gracefully...")


class Main:
    def __init__(self):
        self.base_folder = r"E:\HaliBatt\nidaqmx_logs"
        self.output_folder = ''
        self.data_file = ''
        self.device_info_file = ''
        self.measurement_settings_file = ''

    def run(self):
        self.set_up_logging()
        daq_mngr = NIDAQmxManager()
        daq_mngr.list_available_device_types()
        daq_mngr.list_available_devices()
        daq_mngr.store_device_information(self.device_info_file)
        channel_names = daq_mngr.check_reasonable_readings(threshold=0.05)
        settings = DataAcquisitionSettings(channel_names=channel_names,
                                           sampling_rate=10000,
                                           output_rate=25,
                                           duration=48*3600,
                                           output_file=self.measurement_settings_file,
                                           filename=self.data_file)
        acquisition = DataAcquisition(settings)
        acquisition.start_acquisition_and_averaging()

    def set_up_logging(self):
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.output_folder = os.path.join(self.base_folder, f'log_data_{ts}')
        os.mkdir(self.output_folder)
        self.data_file = os.path.join(self.output_folder, 'output_data.csv')
        self.measurement_settings_file = os.path.join(self.output_folder, 'measurement_settings.json')
        self.device_info_file = os.path.join(self.output_folder, 'device_info.txt')


if __name__ == "__main__":
    main = Main()
    main.run()
