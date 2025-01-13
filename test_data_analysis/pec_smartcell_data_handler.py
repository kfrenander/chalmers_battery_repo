import os
import re
import time
from collections import defaultdict
from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import trapz
from test_data_analysis.PecSmartCellData import PecSmartCellData


class PecSmartCellDataHandler:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_time = None
        start_time = time.time()
        self.pscd_dict = self._load_pscd_objects()
        finish_time = time.time()
        self.load_time = finish_time - start_time
        self.grouped_by_cell_id = self._group_by_cell_id()
        self.sorted_pscd = self._sort_grouped_pscd()
        self.merged_pscd = self._merge_grouped_pscd()
        self.grouped_by_condition = self._group_by_test_condition()
        self.merged_condition_data = {}

    def _find_name(self, fname):
        try:
            test = re.search(r'\d+_', fname).group().strip('_')
            cell = re.search(r'-\d+', fname).group().strip('-')
            return f'{test}_cell{cell}'
        except AttributeError:
            print(f"Error parsing filename: {fname}")
            return None

    def _load_pscd_objects(self):
        return {
            self._find_name(f): PecSmartCellData(os.path.join(self.data_dir, f), op_bool=1)
            for f in os.listdir(self.data_dir) if f.endswith('csv') and self._find_name(f) is not None
        }

    def _group_by_cell_id(self):
        grouped = defaultdict(list)
        for _, pscd in self.pscd_dict.items():
            grouped[pscd.formatted_metadata['CELL_ID']].append(pscd)
        return grouped

    def _group_by_test_condition(self):
        grouped = defaultdict(dict)
        for key, pscd in self.merged_pscd.items():
            grouped[pscd.formatted_metadata['TEST_CONDITION']][key] = pscd
        return grouped

    def _sort_grouped_pscd(self):
        for k, pscd_list in self.grouped_by_cell_id.items():
            self.grouped_by_cell_id[k] = sorted(pscd_list, key=lambda x: int(x.test_nbr))
        return self.grouped_by_cell_id

    def _merge_grouped_pscd(self):
        merged = {}
        for cell_id, pscd_list in self.grouped_by_cell_id.items():
            merged[cell_id] = self._merge_pscds(pscd_list) if len(pscd_list) > 1 else pscd_list[0]
            merged[cell_id].op_folder = f'{merged[cell_id].op_folder}_merged_key_data'
            merged[cell_id].write_files()
        return merged

    def _merge_pscds(self, pscd_list):
        pscd_list[0].ici_dict = self._merge_ici_dicts(pscd_list)
        pscd_list[0].rpt_obj.rpt_summary = self._merge_rpt_objs(pscd_list)
        return pscd_list[0]

    def _merge_ici_dicts(self, pscd_list):
        merged_dict = {}
        current_key_offset = 0

        for pscd_instance in sorted(pscd_list, key=lambda x: int(x.test_nbr)):
            for key, value in pscd_instance.ici_dict.items():
                new_key = current_key_offset + key
                while new_key in merged_dict:
                    new_key += 1
                merged_dict[new_key] = value
            current_key_offset += key + 1

        return merged_dict

    def _merge_rpt_objs(self, pscd_list):
        merged_df = pd.concat(
            [pscd.rpt_obj.rpt_summary.copy() for pscd in pscd_list],
            ignore_index=True
        )
        merged_df = merged_df.sort_values(by='date').reset_index(drop=True)
        merged_df['cap_normalised'] = merged_df['cap'] / merged_df['cap'].iloc[0]

        offset = 0
        last_fce = None
        for i in range(len(merged_df)):
            current_fce = merged_df.at[i, 'fce']
            if last_fce is not None and current_fce + offset <= last_fce:
                offset = last_fce
                if current_fce == 0:
                    offset += 20
            merged_df.at[i, 'fce'] = current_fce + offset
            last_fce = merged_df.at[i, 'fce']

        return merged_df

    def merge_test_condition_data(self):
        """
        Merge data by test condition, calculate mean and standard deviation across replicates.
        """
        for condition, group in self.grouped_by_condition.items():
            # Collect rpt_summary dataframes for this condition
            df_dict = {cell: pscd.rpt_obj.rpt_summary for cell, pscd in group.items()}
            suffixes = [f'_{cell}' for cell in df_dict.keys()]
            merged_df = reduce(lambda df1, df2: pd.merge(df1, df2, on='fce', suffixes=suffixes),
                               df_dict.values())
            # Calculate mean and standard deviation for normalized capacity
            merged_df['mean_capacity'] = merged_df.filter(like='cap_norm').mean(axis=1)
            merged_df['sigma_capacity'] = merged_df.filter(like='cap_norm').std(axis=1)

            self.merged_condition_data[condition] = {
                'merged_df': merged_df,
                'style': list(group.values())[0].style  # Assume style is consistent within the group
            }

    def filter_by_cell_id(self, cell_id):
        return self.merged_pscd.get(cell_id, None)

    def filter_by_test_condition(self, test_condition):
        return {
            k: pscd for k, pscd in self.merged_pscd.items()
            if pscd.formatted_metadata['TEST_CONDITION'] == test_condition
        }

    def calculate_mean_temperature(self):
        mean_temperature_dict = {
            k: trapz(pscd.dyn_data.temperature, pscd.dyn_data.float_time) / pscd.dyn_data.float_time.max()
            for k, pscd in self.merged_pscd.items()
        }
        return pd.DataFrame.from_dict(mean_temperature_dict, orient='index', columns=['Mean Temperature'])


if __name__ == "__main__":
    data_dir = r"\\sol.ita.chalmers.se\groups\batt_lab_data\pulse_chrg_test\high_frequency_testing\PEC_export"
    # data_dir = r"D:\PEC_logs\MWE_merge"
    handler = PecSmartCellDataHandler(data_dir)

    mean_temps = handler.calculate_mean_temperature()
    print(mean_temps)

    cap_90_dict = {}
    for cell, pscd in handler.merged_pscd.items():
        pscd.fit_degradation_function()
        cap_90_dict[cell] = pscd.find_fce_at_given_q(0.9)
    cap_90_df = pd.DataFrame.from_dict(cap_90_dict, orient='index')

    handler.merge_test_condition_data()
