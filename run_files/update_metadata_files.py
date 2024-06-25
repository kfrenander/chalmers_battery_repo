from test_data_analysis.parse_test_metadata import parse_metadata_to_dataframe
from check_current_os import get_base_path_batt_lab_data
import os
import pandas as pd
import glob


BATT_LAB_DATA_BASE_PATH = get_base_path_batt_lab_data()
XL_DB_FILE = r"neware_test_inventory.xlsx"
XL_DB_PATH = os.path.join(BATT_LAB_DATA_BASE_PATH, XL_DB_FILE)


def read_cell_inventory():
    return pd.read_excel(XL_DB_PATH)


def update_test_metadata(xldf, meta_data_file):
    df = parse_metadata_to_dataframe(os.path.join(BATT_LAB_DATA_BASE_PATH, meta_data_file))
    df = pd.merge(df, xldf, on=['UNIT', 'MACHINE', 'CHANNEL', 'TEST'])
    return df


def write_metadata_to_txt_file(df, fname):
    fname = os.path.join(BATT_LAB_DATA_BASE_PATH, fname)
    with open(fname, 'w') as file:
        # Iterate through each row in the dataframe
        for index, row in df.iterrows():
            # Iterate through each column in the row
            for col in df.columns:
                # Write the formatted string to the file
                file.write(f"{col}\t:\t{row[col]}\n")
            # Write a newline character to separate rows
            file.write("\n")
    return None


def update_all_metadata_files_in_directory(dir_path):
    dir_path = os.path.join(BATT_LAB_DATA_BASE_PATH, dir_path)
    xldf = read_cell_inventory()
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            # Find all metadata files in current directory that has not been updated.
            if glob.fnmatch.fnmatch(file, 'metadata_*.txt') and not 'updated' in file:
                file_path = os.path.join(root, file)
                df = update_test_metadata(xldf, file_path)
                write_metadata_to_txt_file(df, file_path.replace('.txt', '_updated.txt'))
                print(f'Updated file {file}')


if __name__ == '__main__':
    METADATA_FILE = ("pulse_chrg_test\cycling_data\pickle_files_channel_240095_2_3"
                     "\metadata_240095_2_3_test_2818575226.txt")
    DIRECTORY_PATH = "stat_test/cycling_data"
    xldf = read_cell_inventory()
    df = update_test_metadata(xldf, meta_data_file=METADATA_FILE)
    METADATA_UPDATE_FILE = METADATA_FILE.replace('.txt', '_updated.txt')
    update_all_metadata_files_in_directory(DIRECTORY_PATH)
