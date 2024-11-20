import pandas as pd
from check_current_os import get_base_path_batt_lab_data
import os


class MetadataReader:
    excel_file_path = None
    excel_data = None

    def __init__(self, unit=None, machine=None, channel=None, test=None, file_path=None):
        """
        Initializes the MetadataReader either from a text file or directly from values.

        Parameters:
            unit (str, optional): The unit identifier.
            machine (str, optional): The machine identifier.
            channel (str, optional): The channel identifier.
            test (str, optional): The test identifier.
            file_path (str, optional): The path to the metadata text file.
        """
        self.unit = unit
        self.machine = machine
        self.channel = channel
        self.test = test
        self.test_condition = None
        self.cell_id = None
        self.cell_type = None
        self.project = None

        # Read and store metadata from the file
        if file_path:
            self._read_metadata(file_path)
        MetadataReader._read_excel_file()
        # Fetch additional information from the Excel file if it is set
        if MetadataReader.excel_file_path:
            self._fetch_additional_info()

    def _read_metadata(self, file_path):
        """
        Reads the metadata from the file and stores it as attributes.

        Parameters:
            file_path (str): The path to the metadata text file.
        """
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key, value = parts[0].strip().lower(), parts[1].strip()
                        if key == 'unit':
                            self.unit = value
                        elif key == 'machine':
                            self.machine = value
                        elif key == 'channel':
                            self.channel = value
                        elif key == 'test':
                            self.test = value
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
        except Exception as e:
            print(f"An error occurred while reading the metadata file: {e}")

    def _fetch_additional_info(self):
        """
        Fetches additional information from the stored Excel DataFrame based on metadata.
        """
        try:
            # Filter the DataFrame based on the metadata
            matching_rows = MetadataReader.excel_data[
                (MetadataReader.excel_data['UNIT'] == int(self.unit)) &
                (MetadataReader.excel_data['MACHINE'] == int(self.machine)) &
                (MetadataReader.excel_data['CHANNEL'] == int(self.channel)) &
                (MetadataReader.excel_data['TEST'] == int(self.test))
                ]

            # If a matching row is found, extract the additional information
            if not matching_rows.empty:
                self.test_condition = matching_rows.iloc[0]['TEST_CONDITION']
                self.cell_id = matching_rows.iloc[0]['CELL_ID']
                self.cell_type = matching_rows.iloc[0]['CELL_TYPE']
                self.project = matching_rows.iloc[0]['PROJECT']
                self.output_name = matching_rows.iloc[0]['OUTPUT_NAME']
            else:
                print("No matching data found in the Excel file.")

        except Exception as e:
            print(f"An error occurred while processing the Excel data: {e}")

    @classmethod
    def _read_excel_file(cls):
        """
            Sets the class variable for the Excel file path and loads the Excel data.

            Parameters:
                path (str): The path to the Excel file.
        """
        BASE_PATH = get_base_path_batt_lab_data()
        cls.excel_file_path = os.path.join(BASE_PATH,
                                       'neware_test_inventory.xlsx')  # Class variable for the path to the inventory
        try:
            cls.excel_data = pd.read_excel(cls.excel_file_path)
        except FileNotFoundError:
            print(f"Error: The file {cls.excel_file_path} was not found.")
        except Exception as e:
            print(f"An error occurred while reading the Excel file: {e}")

    @classmethod
    def from_values(cls, unit, machine, channel, test):
        """
        Alternative constructor that initializes the MetadataReader from given values.

        Parameters:
            unit (str): The unit identifier.
            machine (str): The machine identifier.
            channel (str): The channel identifier.
            test (str): The test identifier.

        Returns:
            MetadataReader: An instance of MetadataReader initialized with the given values.
        """
        return cls(unit=unit, machine=machine, channel=channel, test=test)


    def output_to_file(self, output_path):
        """
        Outputs all metadata information to a text file.

        Parameters:
            output_path (str): The path to the output text file.
        """
        try:
            with open(output_path, 'w') as file:
                file.write(f"UNIT: {self.unit}\n")
                file.write(f"MACHINE: {self.machine}\n")
                file.write(f"CHANNEL: {self.channel}\n")
                file.write(f"TEST: {self.test}\n")
                if self.test_condition:
                    file.write(f"TEST_CONDITION: {self.test_condition}\n")
                if self.cell_id:
                    file.write(f"CELL_ID: {self.cell_id}\n")
                if self.cell_type:
                    file.write(f"CELL_TYPE: {self.cell_type}\n")
                if self.project:
                    file.write(f"PROJECT: {self.project}\n")
                if self.output_name:
                    file.write(f"OUTPUT_NAME: {self.output_name}\n")
        except Exception as e:
            print(f"An error occurred while writing to the output file: {e}")

    def __repr__(self):
        return (f"MetadataReader(unit={self.unit}, machine={self.machine}, channel={self.channel}, "
                f"test={self.test}, test_condition={self.test_condition}, cell_id={self.cell_id}, "
                f"cell_type={self.cell_type}, project={self.project})")


if __name__ == '__main__':
    BASE_PATH = get_base_path_batt_lab_data()
    # Example usage
    fname = "pulse_chrg_test/cycling_data_repaired/pickle_files_channel_240095_2_7_100mHz/metadata_240095_2_7_test_2818575237.txt"
    xl_data_file = 'neware_test_inventory.xlsx'
    xl_file_path = os.path.join(BASE_PATH, xl_data_file)
    file_path = os.path.join(BASE_PATH, fname)

    # Alternative constructor to create an instance from values
    meta_data_from_values = MetadataReader.from_values(unit='240095', machine='2', channel='1', test='2818575226')

    # Replace with the actual path to your metadata file
    meta_data = MetadataReader(file_path=file_path)

    # Output to file
    meta_data.output_to_file(os.path.join(BASE_PATH, 'pulse_chrg_test/cycling_data_repaired/test_output_metadata.txt'))

    empty_meta_data = MetadataReader()

    # Access metadata attributes
    print(f"Unit: {meta_data.unit}")
    print(f"Machine: {meta_data.machine}")
    print(f"Channel: {meta_data.channel}")
    print(f"Test: {meta_data.test}")
    print(f"Test Condition: {meta_data.test_condition}")
    print(f"Cell ID: {meta_data.cell_id}")
    print(f"Cell Type: {meta_data.cell_type}")
    print(f"Project: {meta_data.project}")

