import unittest
import pandas as pd


from run_files.stat_test_trial import (
    pass_fail_stat_test,
    pass_fail_pt_estim,
    perform_stat_test,
    calc_group_var,
    calc_t_stat
)


class TestData(unittest.TestCase):
    def setUp(self):
        self.sample_data_tstat = pd.DataFrame([[0.92936619, 0.93071128, 0.94011947],
                                               [0.96993959, 0.97002744, 0.97201836]])


# noinspection SpellCheckingInspection
class TestYourModule(TestData):

    def test_pass_fail_stat_test(self):
        # Test when p-value is greater than threshold
        self.assertEqual(pass_fail_stat_test(0.05), 0)
        # Test when p-value is less than threshold
        self.assertEqual(pass_fail_stat_test(0.001), 1)

    def test_pass_fail_pt_estim(self):
        # Test when values are equal
        self.assertEqual(pass_fail_pt_estim(5, 5), 1)
        # Test when values are not equal
        self.assertEqual(pass_fail_pt_estim(3, 5), 0)

    def test_perform_stat_test_ttest_rel(self):
        # Write test cases for perform_stat_test function
        # For example:
        self.assertAlmostEqual(perform_stat_test(self.sample_data_tstat.loc[0, :],
                                                 self.sample_data_tstat.loc[1, :],
                                                 method='ttest'), 0.005233, places=2)

    def test_perform_stat_test_ttest_ind(self):
        # Write test cases for perform_stat_test function
        # For example:
        self.assertAlmostEqual(perform_stat_test(self.sample_data_tstat.loc[0, :],
                                                 self.sample_data_tstat.loc[1, :],
                                                 method='ttest_ind'), 0.0004167, places=2)

    def test_perform_stat_test_anova(self):
        # Write test cases for perform_stat_test function
        # For example:
        self.assertAlmostEqual(perform_stat_test(self.sample_data_tstat.loc[0, :],
                                                 self.sample_data_tstat.loc[1, :],
                                                 method='anova'), 0.000416793, places=2)

    def test_calc_group_var(self):
        # Write test cases for calc_group_var function
        # For example:
        self.assertAlmostEqual(calc_group_var(self.sample_data_tstat.loc[0, :]), 3.4326e-5, places=2)

    def test_calc_t_stat(self):
        # Write test cases for calc_t_stat function
        # For example:
        self.assertAlmostEqual(calc_t_stat(self.sample_data_tstat.loc[0, :],
                                           self.sample_data_tstat.loc[1, :]), -13.7683, places=2)


if __name__ == '__main__':
    unittest.main()
