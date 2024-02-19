import unittest
from datetime import datetime
import pandas as pd
from test_data_analysis.read_mux_file_to_df import (
    unix_time_to_local_datetime,
    convert_unix_time_columns_to_local_datetime,
    filter_sudden_jumps
)


class TestMyFunctions(unittest.TestCase):

    def test_unix_time_to_local_datetime(self):
        unix_time = 1613430000000  # February 16, 2021, 00:00:00 UTC
        expected_datetime = datetime(2021, 2, 15, 23, 0)
        self.assertEqual(unix_time_to_local_datetime(unix_time), expected_datetime)

    def test_convert_unix_time_columns_to_local_datetime(self):
        df = pd.DataFrame({'time1': [1613472000000], 'time2': [1613475600000]})
        expected_op_df = pd.DataFrame({
            'time1_Local_DateTime': [datetime(2021, 2, 16, 10, 40)],
            'time2_Local_DateTime': [datetime(2021, 2, 16, 11, 40)]
        })
        expected_df = pd.concat([pd.DataFrame({'time1': [1613472000000], 'time2': [1613475600000]}),
                                 expected_op_df], axis=1)
        result_df = convert_unix_time_columns_to_local_datetime(df, ['time1', 'time2'])
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_filter_sudden_jumps(self):
        series = pd.Series([1, 2, 10, 3, 4, 5])
        threshold = 2
        expected_series = pd.Series([1, 2, 4, 5])
        result_series = filter_sudden_jumps(series, threshold)
        pd.testing.assert_series_equal(result_series, expected_series, check_index=False)


if __name__ == '__main__':
    unittest.main()