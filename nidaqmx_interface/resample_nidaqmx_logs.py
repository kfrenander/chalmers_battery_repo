import pandas as pd
import os
from numba import njit
import numpy as np
import sys


@njit
def downsample_numba(cols_of_interest, u_thrshld):
    nrows, ncols = cols_of_interest.shape
    downsampled_indices = [0]
    last_stored_value = cols_of_interest[0, :]

    for i in range(1, nrows):
        current_row = cols_of_interest[i, :]
        prev_row = cols_of_interest[i - 1, :]

        # Check both cumulative and fast change conditions
        if (
                np.any(np.abs(current_row - last_stored_value) > u_thrshld) or
                np.any(np.abs(current_row - prev_row) > u_thrshld)
        ):
            downsampled_indices.append(i)
            last_stored_value = current_row

    return np.array(downsampled_indices)


def downsample_df_numba(df, u_thrshld=1e-5):
    cols_of_interest = df.filter(like='92').values
    downsampled_indices = downsample_numba(cols_of_interest, u_thrshld)
    return df.iloc[downsampled_indices]


def downsample_df(df, u_thrshld=1e-5):
    sdf = df.copy()
    data_buffer = []
    last_data = sdf.filter(like='92').iloc[0, :]
    for idx, row in sdf.iterrows():
        if not data_buffer:
            data_buffer.append(row)
        tmp_data = row.filter(like='92')
        if any(abs(tmp_data - last_data) > u_thrshld):
            data_buffer.append(row)
        elif any(abs(tmp_data - data_buffer[-1].filter(like='92')) > u_thrshld):
            data_buffer.append(row)
        last_data = tmp_data
    return pd.DataFrame(data_buffer)


def main(input_file, output_dir, u_thrshld=1e-5):
    # Read the input CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)
    if not isinstance(u_thrshld, float):
        u_thrshld = float(u_thrshld)

    # Perform downsampling
    tic = time.time()
    downsampled_df = downsample_df_numba(df, u_thrshld)
    toc = time.time()
    print(f'Elapsed time for numba is {toc - tic:.2f}s.')

    # Create output file path
    base_name = os.path.basename(input_file)  # Extract the filename
    file_name, file_ext = os.path.splitext(base_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{file_name}_downsampled")

    # Save the downsampled DataFrame to a CSV file
    downsampled_df.to_csv(output_file, index=False)
    print(f"Downsampled data saved to {output_file}")


if __name__ == '__main__':
    import time
    tic = time.time()
    main_args = [*sys.argv][1:]
    main(*main_args)
    toc = time.time()
    print(f'Elapsed time for downsampling is {toc - tic:.2f}s.')
