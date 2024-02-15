from test_data_analysis.BasePecDataClass import BasePecData, BasePecRpt, find_step_characteristics


class PecLifeTestData(BasePecData):

    def __init__(self, fname, data_init_row=24):
        super().__init__(fname, data_init_row=24)

    def find_all_rpt(self):
        fast_rpt_dict = self.find_rpt(rpt_type='fast')
        comp_rpt_dict = self.find_rpt(rpt_type='comp')
        self.rpt_dict = {
            'fast': fast_rpt_dict,
            'comp': comp_rpt_dict
        }
        return None

    def find_rpt(self, rpt_type):
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

    def find_ici(self):
        ici_df = self.dyn_data[self.dyn_data['ici_bool'] == 1]
        gb = ici_df.groupby(by='CycNbrOuter')
        self.ici_dict = {k: gb.get_group(k) for k in gb.groups}
        return None

if __name__ == '__main__':
    fname = r"\\sol.ita.chalmers.se\groups\batt_lab_data\smart_cell_JG\TestBatch2_autumn2023\Test2484_Cell-1.csv"
    pltd_obj = PecLifeTestData(fname)
