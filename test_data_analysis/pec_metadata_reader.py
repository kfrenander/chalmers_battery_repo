import pandas as pd
from check_current_os import get_base_path_batt_lab_data
import os
import warnings


class PECMetadata:
    def __init__(self):
        """
        Initialize the PECMetadata class.

        """
        BASE_PATH_BATT_LAB_DATA = get_base_path_batt_lab_data()
        excel_path = os.path.join(BASE_PATH_BATT_LAB_DATA, 'pec_test_inventory.xlsx')
        self.df = pd.read_excel(excel_path)
        self.df.columns = self.df.columns.str.strip()  # Remove whitespace from column names

        # Define the required types for validation
        self.required_types = {
            'UNIT': str,
            'SHELF': str,
            'CHANNEL': (int, float),
            'TEST_NBR': (int, float),
            'CELL_NBR': (int, float),
            'CELL_ID': (int, float),
            'TEST_CONDITION': str,
            'CELL_TYPE': str,
            'PROJECT': str,
            'OUTPUT_NAME': str,
        }

    def _validate_and_convert(self, key, value):
        """
        Validate and, if necessary, convert the value to the required type.

        Args:
            key (str): The metadata column being checked.
            value: The value to validate and convert.

        Returns:
            The validated and potentially converted value.

        Raises:
            TypeError: If the value cannot be converted to the required type.
        """
        expected_type = self.required_types[key]

        # Check if the value is already of the correct type
        if isinstance(value, expected_type):
            return value

        # Handle numeric types (int/float) with string input
        if expected_type == (int, float) or isinstance(expected_type, tuple):
            if isinstance(value, str) and value.replace('.', '', 1).isdigit():
                # Determine whether to cast to int or float
                return int(value) if value.isdigit() else float(value)

        # Raise TypeError if value cannot be validated
        raise TypeError(f"Invalid type for '{key}'. Expected {expected_type}, got {type(value)}.")

    def query(self, **kwargs):
        """
        Query the metadata using provided key-value pairs to identify a unique test.

        Args:
            kwargs: Key-value pairs to filter the metadata.

        Returns:
            dict: Metadata of the uniquely identified test.

        Raises:
            ValueError: If the query identifies zero or more than one unique test.
        """

        # Validate and convert query parameters
        query_params = {key: self._validate_and_convert(key, value) for key, value in kwargs.items()}

        # Filter the dataframe based on the query
        filtered_df = self.df
        for key, value in query_params.items():
            filtered_df = filtered_df[filtered_df[key] == value]

        # Check if we have a unique result
        if len(filtered_df) == 0:
            warnings.warn('No matching cell found in metadata')
            return {}
        elif len(filtered_df) > 1:
            raise ValueError("Query matches more than one test. Please provide additional metadata.")

        # Convert the row to a dictionary
        return filtered_df.iloc[0].to_dict()

    def save_query_to_file(self, filepath, **kwargs):
        """
        Query the metadata and save the result to a text file.

        Args:
            filepath (str): Path to save the result.
            kwargs: Key-value pairs to filter the metadata.
        """
        result = self.query(**kwargs)
        with open(filepath, 'w') as f:
            for key, value in result.items():
                f.write(f"{key}: {value}\n")


if __name__ == '__main__':
    pec_metadata = PECMetadata()
    cell_220 = pec_metadata.query(CELL_ID='220')
    test2805_cell2 = pec_metadata.query(TEST_NBR=2805, CELL_NBR='2')
    try:
        multicell = pec_metadata.query(TEST_NBR=2805)
    except ValueError as e:
        print(f"Could not use prompt, raised {e}")
    no_match = pec_metadata.query(CELL_ID=2031)
