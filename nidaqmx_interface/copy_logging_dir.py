import shutil
import time
import os
import socket
import re
from datetime import datetime


def copy_file(source, destination):
    try:
        shutil.copytree(source, destination, dirs_exist_ok=True)
        copytime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
        print(f"File copied on {copytime} to {destination}")
    except FileNotFoundError:
        print("Source file not found.")
    except PermissionError:
        print("Permission denied.")
    except shutil.Error as e:
        print(f"Error while copying file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def extract_date_from_string(date_str):
    pattern = r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    match = re.search(pattern, date_str)
    if match:
        datetime_str = match.group(1)
        return datetime.strptime(datetime_str, '%Y-%m-%d_%H-%M-%S')
    else:
        return None


def find_latest_date_folder(base_path):
    latest_date = None
    latest_foldername = None

    for content in os.listdir(base_path):
        c_path = os.path.join(base_path, content)
        if os.path.isdir(c_path):
            date = extract_date_from_string(c_path)
            if date:
                if latest_date is None or date > latest_date:
                    latest_date = date
                    latest_foldername = c_path
    return latest_foldername


def main():
    # Specify the source and destination paths
    source_folder = r"E:\HaliBatt\nidaqmx_logs"
    f_name = find_latest_date_folder(source_folder)
    source_file = os.path.join(source_folder, f_name)
    destination_folder = r"\\sol.ita.chalmers.se\groups\eom-et-alla\Research\HaliBatt\SiGr_materials\nidaqmx_logs"
    destination_file = os.path.join(destination_folder, os.path.basename(source_file))

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    try:
        while True:
            # Copy the file
            copy_file(source_file, destination_file)

            # Wait for 30 minutes
            time.sleep(30 * 60)
    except KeyboardInterrupt:
        print("Program terminated by user.")
    except socket.error as e:
        print(f"Network error occurred: {e}")


if __name__ == "__main__":
    main()
