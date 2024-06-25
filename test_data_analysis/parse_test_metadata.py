import pandas as pd
import re


def parse_metadata_to_dataframe(fname):
    data = {}

    # Open and read the file
    with open(fname, 'r') as f:
        for line in f:
            # Strip whitespace and split by ' : '
            key, value = re.split(r'\s*:\s*', line.strip())

            try:
                value = int(value)
            except ValueError:
                pass
            # Add to dictionary
            data[key] = value

    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(data, orient='index', columns=['ID']).T
    df.columns = [col.replace('_ID', '') for col in df.columns]
    return df


if __name__ == '__main__':
    # Example usage
    file_path = (r"\\sol.ita.chalmers.se\groups\batt_lab_data\pulse_chrg_test\cycling_data"
                 r"\pickle_files_channel_240095_2_3\metadata_240095_2_3_test_2818575226.txt")
    df = parse_metadata_to_dataframe(file_path)
    print(df)
