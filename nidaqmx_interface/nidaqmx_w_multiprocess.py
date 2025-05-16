import queue
import nidaqmx
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader
import numpy as np
import time
import csv
import pandas as pd
from scipy.stats import norm
import threading
import argparse
import re
import datetime
import os
import tables as tb
import multiprocessing
from multiprocessing import Process, Queue, Manager


class JitterLogger:
    def __init__(self, expected_interval_s, verbose=False):
        self.expected_interval = expected_interval_s
        self.timestamps = []
        self.jitter_us = []  # store jitter in microseconds
        self.verbose = verbose

    def record_callback(self):
        now = time.time()
        self.timestamps.append(now)

        if len(self.timestamps) >= 2:
            dt = self.timestamps[-1] - self.timestamps[-2]
            jitter = (dt - self.expected_interval) * 1e6  # in µs
            self.jitter_us.append(jitter)

            if self.verbose:
                print(f"[JitterLogger] Interval: {dt:.6f}s | Jitter: {jitter:+.2f} µs")

    def get_jitter_array(self):
        return np.array(self.jitter_us)

    def reset(self):
        self.timestamps.clear()
        self.jitter_us.clear()

    def save_to_csv(self, filename="jitter_log.csv"):
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Callback Index", "Jitter (us)"])
            for i, jitter in enumerate(self.jitter_us):
                writer.writerow([i, jitter])
        print(f"[JitterLogger] Saved jitter log to {filename}")


def print_timer(tic, toc):
    """
    Common function to print out write duration to log file.
    Args:
        tic: Start time (float
        toc: End time (float)

    Returns: None

    """
    datestamp = f"{pd.to_datetime(time.time(), unit='s', utc=True).tz_convert('Europe/Berlin'):%y%m%d_%H%M}"
    print(f'[WRITER] [{datestamp}] Duration for flushing to disk was {toc - tic:.3f} s...')


def drain_queue(q):
    print(f"[WRITER] Current queue size: {q.qsize()} items", flush=True)
    items = []
    while True:
        try:
            item = q.get_nowait()
            items.append(item)
        except queue.Empty:
            break
    print(f"[WRITER] Queue size after drain: {q.qsize()} items", flush=True)
    return items


def flush_queue_with_stop(queue, flush_interval, stop_marker="STOP"):
    """Drains the queue and checks for STOP signal."""
    time.sleep(flush_interval)
    datestamp = f"{pd.to_datetime(time.time(), unit='s', utc=True).tz_convert('Europe/Berlin'):%y%m%d_%H%M}"
    print(f"[WRITER] [{datestamp}] Checking for items in queue...")
    all_items = drain_queue(queue)

    stop_received = False
    if stop_marker in all_items:
        print("[WRITER] Received STOP signal.")
        stop_received = True
        all_items = [item for item in all_items if item != stop_marker]

    return all_items, stop_received


def check_channel_type(device):
    prod_type = device.product_type
    if '9252' in prod_type:
        return 'voltage'
    if '9253' in prod_type:
        return 'current'
    if '9236' in prod_type:
        return 'bridge'
    else:
        raise ValueError(f"Unknown measurement device, please check type for {prod_type}")


def write_settings_to_log_file(file_location, dev):
    os.makedirs(file_location, exist_ok=True)
    with open(os.path.join(file_location, 'output_log.txt'), 'a+') as f:
        f.write(f'Product name: {dev.name}\n')
        f.write(f'Product type: {dev.product_type}\n')
        f.write(f'Serial number: {dev.serial_num}\n')
        f.write('\n\n')


def init_hdf_file(filepath, column_headers):
    # Check for invalid names
    bad_headers = [h for h in column_headers if not re.match(r'^[a-zA-Z_]\w*$', h)]
    if bad_headers:
        print("Invalid HDF5 column names detected:", bad_headers)
        raise ValueError("Fix invalid column names before proceeding.")

    # Create a dictionary of sanitized column names and Float64Col types
    columns_dict = {col: tb.Float64Col() for col in column_headers}

    # Dynamically create a new IsDescription subclass
    LogTable = type("LogTable", (tb.IsDescription,), columns_dict)

    with tb.open_file(filepath, mode='w') as h5file:
        group = h5file.create_group('/', 'log', 'Logged Data')
        h5file.create_table(group, 'data', LogTable, 'DAQ Log')


def write_buffer_to_hdf5_with_pytables(filepath, colnames, queue, test_settings):
    """
        Appends a chunk of data to the HDF5 file using PyTables directly.
        `buffer` is a list of tuples or lists, each representing a row.
        """
    buffer_flush_interval = test_settings['buffer_flush_interval']
    print(f'[WRITER] Sleep {buffer_flush_interval} s between flushes to disk', flush=True)

    while True:
        all_items, stop_received = flush_queue_with_stop(queue, buffer_flush_interval)
        if not all_items and stop_received:
            break

        if all_items:
            tic = time.time()
            with tb.open_file(filepath, mode='a') as h5file:
                table = h5file.root.log.data
                row = table.row

                for row_data in all_items:
                    for name, val in zip(colnames, row_data):
                        row[name] = val
                    row.append()

                table.flush()
            toc = time.time()
            print_timer(tic, toc)

        if stop_received:
            print("[WRITER] Finished writing all data. Exiting.")
            break


def write_buffer_to_hdf5_file(debug_file, header, queue, test_settings):
    buffer_flush_interval = test_settings['buffer_flush_interval']
    print(f'Sleep {buffer_flush_interval} s between flushes to disk')
    output_file = debug_file + '.h5'
    while True:
        all_items, stop_received = flush_queue_with_stop(queue, test_settings['buffer_flush_interval'])
        if not all_items and stop_received:
            break

        if all_items:
            if len(header) != len(all_items[0]):
                print(f"[WRITER] Mismatch: {len(header)} header columns vs {len(all_items[0])} row columns.")

            df = pd.DataFrame(all_items, columns=list(header))
            tic = time.time()
            df.to_hdf(output_file, key='data', mode='a', format='table', append=True)
            toc = time.time()
            print_timer(tic, toc)

        if stop_received:
            print("[WRITER] Finished writing all data. Exiting.")
            break


def write_buffer_to_hdf5_channel_file(file_names, header, queue, test_settings, file_location):
    """Function to periodically flush data buffer to one .h5 file per channel."""
    while True:
        all_items, stop_received = flush_queue_with_stop(queue, test_settings['buffer_flush_interval'])
        if all_items:
            data_arr = np.array(all_items)
            tic = time.time()
            for i, fname in enumerate(file_names):
                channel_data = data_arr[:, i].astype(np.float64)
                # Create a DataFrame
                df = pd.DataFrame(channel_data, columns=[header[i]])
                # Write to HDF5 with append mode
                h5_name = fname.replace('.csv', '.h5')
                h5_path = os.path.join(file_location, h5_name)
                # Append to HDF5 file
                df.to_hdf(h5_path, key='data', mode='a', format='table', append=True)
            toc = time.time()
            print_timer(tic, toc)

        if stop_received:
            print("[WRITER] Finished writing all data. Exiting.")
            break


def write_buffer_to_csv_file(output_file, header, queue, test_settings):
    """Function to periodically flush data buffer to single csv file."""
    header_written = False
    output_file = output_file + '.csv'
    while True:
        all_data, stop_received = flush_queue_with_stop(queue, test_settings['buffer_flush_interval'])
        # print(all_data)
        if all_data:
            with open(output_file, 'a+', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                if not header_written:
                    csv_writer.writerow(header)
                    header_written = True
                tic = time.time()
                csv_writer.writerows(all_data)
            toc = time.time()
            print_timer(tic, toc)

        if stop_received:
            print("[WRITER] Finished writing all data. Exiting.")
            break


def write_buffer_to_csv_channel_file(file_names, header, queue, test_settings, file_location):
    """Function to periodically flush data buffer to one csv file per channel."""
    time.sleep(0.5)
    header_written = [False] * len(file_names)
    print(header)
    print(header_written)
    while True:
        all_data, stop_received = flush_queue_with_stop(queue, test_settings['buffer_flush_interval'])

        if all_data:
            data_arr = np.array(all_data)
            tic = time.time()
            for i, fname in enumerate(file_names):
                column_data = data_arr[:, i]
                with open(os.path.join(file_location, fname), 'a+', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    if not header_written[i]:
                        csv_writer.writerow([header[i]])
                        header_written[i] = True
                    csv_writer.writerows([[val] for val in column_data])
            toc = time.time()
            print_timer(tic, toc)
        if stop_received:
            print("[WRITER] Finished writing all data. Exiting.")
            break


# --- DAQ READER FUNCTION ---
def reader_func(header, file_names, writer_queue, unit_to_log, n_samples,
                test_settings, start_time, file_location, jitter_logger):
    system = nidaqmx.system.System.local()
    channels = []

    def callback(task_handle, event_type, number_of_samples, callback_data):
        # jitter_logger.record_callback()
        if task.is_task_done():
            return 1
        ts = time.time()
        reader.read_many_sample(buffer, n_samples)
        # print(buffer)
        try:
            averaged = np.mean(buffer, axis=1)
            row = [ts] + averaged.tolist()
            writer_queue.put_nowait(row)
            # writer_queue.put(buffer)
        except queue.Full:
            print("[CALLBACK] Writer queue is full — dropping a row!")
            queue_full_counter += 1
        return 0

    if system.devices:
        with nidaqmx.Task() as task:
            print("Taking Channel Readings:")
            header.append('time')  # Add the time column to header
            file_names.append('timestamps.csv')

            for device in system.devices:
                if unit_to_log in device.name:
                    dev_id = re.search(r'\d+', device.product_type).group()
                    write_settings_to_log_file(file_location, device)

                    for ch in device.ai_physical_chans:
                        ch_nm = re.search(r'Mod\d/ai\d', ch.name).group()
                        ch_type = check_channel_type(device)
                        if ch_type == 'voltage':
                            task.ai_channels.add_ai_voltage_chan(ch.name, min_val=-10.0, max_val=10.0)
                            print(f"  - Added {ch_type} channel: {ch.name}")
                            ch_name = '-'.join(ch.name.replace('/', '-').split('-')[1:])
                            channels.append(ch_name)
                            header.append(f"ch_{dev_id}_{ch_nm.replace('/', '_')}")  # Dynamically adding to header
                            file_names.append(f'{ch_name}.csv')
                        elif ch_type == 'current':
                            pass
                            # task.ai_channels.add_ai_current_chan(ch.name, min_val=-0.021, max_val=0.021)
                            # print(f"  - Added {ch_type} channel: {ch.name}")
                            # ch_name = '-'.join(ch.name.replace('/', '-').split('-')[1:])
                            # channels.append(ch_name)
                            # header.append(f"ch_{dev_id}_{ch_nm.replace('/', '_')}")  # Dynamically adding to header
                            # file_names.append(f'{ch_name}.csv')
                        else:
                            print(f"Device of type {device.product_type} not relevant in this test.")

                        # print(f"  - Added {ch_type} channel: {ch.name}")
                        # ch_name = '-'.join(ch.name.replace('/', '-').split('-')[1:])
                        # channels.append(ch_name)
                        # file_names.append(f'{ch_name}.csv')

            print(f'Logging N={len(channels)} channels with {n_samples} samples per data chunk')
            # Ensure the header and file names are populated by the time we start logging
            print(f"[READER] Header populated: {header}")
            print(f"[READER] File names populated: {file_names}")
            sample_rate = test_settings['internal_sample_f']
            callback_interval = n_samples
            buffer_duration = 2.0
            buffer_size = int(sample_rate * buffer_duration)

            # Round up to nearest multiple of callback_interval
            # buffer_size = ((buffer_size // callback_interval) + 1) * callback_interval
            # task.in_stream.input_buf_size = buffer_size
            task.timing.cfg_samp_clk_timing(
                rate=test_settings['internal_sample_f'],
                sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
            )

            buffer = np.empty((len(channels), n_samples), dtype=np.float64)
            reader = AnalogMultiChannelReader(task.in_stream)
            task.register_every_n_samples_acquired_into_buffer_event(n_samples, callback)
            task.start()

            while time.time() < start_time + test_settings['test_duration']:
                time.sleep(1)

            task.stop()


def write_log_header(file_location, settings, n_samples):
    with open(os.path.join(file_location, 'output_log.txt'), 'w') as f:
        f.write(f'Output frequency:   {settings["output_f"]} Hz\n')
        f.write(f'Internal sampling frequency:   {settings["internal_sample_f"] / 1e3:.0f} kHz\n')
        f.write(f'N samples per output:   {n_samples}\n')
        f.write(f'Data is written to disk every {settings["buffer_flush_interval"] / 60:.2f} min\n')
        f.write(f'Output format: {settings["output_format"]}\n')
        f.write(f'Logging method: {settings["log_type"]}\n\n\n')


def main(unit_to_log):
    # --- CONFIGURATION ---
    test_settings = {
        'internal_sample_f': 20000,
        'output_f': 25,
        'test_duration': 15*24*3600,
        'buffer_flush_interval': 10,
        'output_format': 'one_file_csv',
        'log_type': 'callback',
        'debug_mode': 0
    }

    n_samples = int(test_settings['internal_sample_f'] / test_settings['output_f'])
    ts_start_log = f'{datetime.datetime.now():%Y_%m_%d__%H_%M}'
    start_time = time.time()

    print(f'Test starts at {ts_start_log}')
    print(f'Files are written in {test_settings["output_format"]} format.')

    # --- PATHS ---
    file_location = f'/data/nidaqmx-logs/si-gr-batch3-spring25/faulttrace/log_{unit_to_log}_{ts_start_log}'
    os.makedirs(file_location, exist_ok=True)
    debug_file = os.path.join(file_location, f'unit{unit_to_log}_{ts_start_log}')

    # --- INITIALIZE ---
    manager = Manager()  # Create a manager for shared state
    writer_queue = Queue(maxsize=20000)
    header = manager.list()  # Use a managed list to share header between processes
    file_names = manager.list()
    jitter_logger = JitterLogger(1 / test_settings['output_f'], verbose=True)
    queue_full_counter = 0

    write_log_header(file_location, test_settings, n_samples)

    # --- START LOGGING ---
    daq_thread = threading.Thread(target=reader_func, args=(
    header, file_names, writer_queue, unit_to_log, n_samples, test_settings, start_time, file_location, jitter_logger))
    daq_thread.start()

    if test_settings['output_format'] == 'one_file_h5':
        writer_process = Process(
            target=write_buffer_to_hdf5_file,
            args=(debug_file, header, writer_queue, test_settings)
        )
    elif test_settings['output_format'] == 'one_file_h5_pytables':
        print("[MAIN] Waiting for header to populate...")
        while len(header) < 10:
            time.sleep(0.1)  # 100 ms wait

        local_header = list(header)
        print('[DEBUG] local_header populated:', local_header)
        print('[DEBUG] ready to proceed with file initialisation...')
        debug_file = debug_file + '.h5'
        init_hdf_file(debug_file, local_header)
        print(f'[MAIN] hdf file initiated')
        writer_process = Process(
            target=write_buffer_to_hdf5_with_pytables,
            args=(debug_file, local_header, writer_queue, test_settings)
        )
    elif test_settings['output_format'] == 'unique_file_h5':
        writer_process = Process(
            target=write_buffer_to_hdf5_channel_file,
            args=(file_names, header, writer_queue, test_settings, file_location)
        )
    elif test_settings['output_format'] == 'one_file_csv':
        writer_process = Process(
            target=write_buffer_to_csv_file,
            args=(debug_file, header, writer_queue, test_settings)
        )
    elif test_settings['output_format'] == 'unique_file_csv':
        writer_process = Process(
            target=write_buffer_to_csv_channel_file,
            args=(file_names, header, writer_queue, test_settings, file_location)
        )
    else:
        raise ValueError("Unknown output format provided, allowed modes: "
                         "\n\t'one_file_h5' "
                         "\n\t'one_file_csv' "
                         "\n\t'unique_file_h5' "
                         "\n\t'unique_file_csv' "
                         "\n\t'one_file_h5_pytables' ")
    print('[MAIN] Starting write process')
    writer_process.start()

    daq_thread.join()
    writer_queue.put("STOP")
    writer_process.join()
    print(f'Queue was full on {queue_full_counter} occasions.')
    # jitter_logger.save_to_csv(os.path.join(file_location, 'jitter_log.csv'))


if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Log data from a specific unit to a file.')
    parser.add_argument('unit_to_log', type=str, help='The unit identifier to log data from.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the command-line argument
    main(args.unit_to_log)
    # main()
