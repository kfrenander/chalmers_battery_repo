from test_data_analysis.BasePecDataClass import BasePecData
from test_data_analysis.ici_analysis_class import ICIAnalysis
from test_data_analysis.pec_metadata_reader import PECMetadata


class PecLifeTestData(BasePecData):

    def __init__(self, filename):
        super().__init__(filename)
        self.rpt_dict = {}
        self.ici_dict = {}
        self.find_all_rpt()
        self.find_ici()
        self.formatted_metadata = self.make_formatted_metadata()

    def find_all_rpt(self):
        fast_rpt_dict = self.find_rpt(rpt_type='fast')
        comp_rpt_dict = self.find_rpt(rpt_type='comp')
        self.rpt_dict = {
            'fast': fast_rpt_dict,
            'comp': comp_rpt_dict
        }
        return None

    def make_formatted_metadata(self):
        pec_meta_data = PECMetadata()
        return pec_meta_data.query(TEST_NBR=self.test_nbr, CELL_NBR=self.cell_nbr)

    def find_rpt_old(self, rpt_type):
        if rpt_type == 'comp':
            bool_col = 'rpt_comp_bool'
        elif rpt_type == 'fast':
            bool_col = 'rpt_fast_bool'
        else:
            print(f'No analysis possible for rpt key of \'{rpt_type}\'. Ending.')
            return None
        rpt_df = self.dyn_data[self.dyn_data[bool_col] == 1]
        gb = rpt_df.groupby(by='CycNbrOuter')
        rpt_dict = {k: gb.get_group(k) for k in gb.groups if gb.get_group(k).shape[0] > 100}
        return rpt_dict

    def find_rpt(self, rpt_type):
        rpt_col_map = {
            'comp': 'rpt_comp_bool',
            'fast': 'rpt_fast_bool'
        }

        rpt_col = rpt_col_map.get(rpt_type)
        if not rpt_col:
            print(f"No analysis possible for rpt key '{rpt_type}'. Ending.")
            return None

        rpt_df = self.dyn_data.loc[self.dyn_data[rpt_col] == 1]
        rpt_dict = {
            k: v for k, v in rpt_df.groupby('CycNbrOuter') if len(v) > 100
        }
        return rpt_dict

    def find_ici(self):
        ici_df = self.dyn_data.loc[self.dyn_data['ici_bool'] == 1]
        if ici_df.empty:
            self.ici_dict = {}
            return

        self.ici_dict = {
            k: ICIAnalysis(df) for k, df in ici_df.groupby('CycNbrOuter')
        }
        for ici_obj in self.ici_dict.values():
            ici_obj.perform_ici_analysis()


if __name__ == '__main__':
    fname = r"\\sol.ita.chalmers.se\groups\batt_lab_data\smart_cell_JG\TestBatch2_autumn2023\Test2484_Cell-1.csv"
    pltd_obj = PecLifeTestData(fname)
