import numpy as np

class ElectrodeMaterial(object):

    def __init__(self, spec_cap, coat_wt, loading):
        self.spec_cap = spec_cap  # In mAh / g
        self.coat_wt = coat_wt * 1e-3  # transfer to g (given in mg/cm^2)
        self.loading = loading
        self.area_cap = self.coat_wt * self.spec_cap * self.loading

    def calc_coin_cell_cap(self, d):
        A = np.pi * (d / 2)**2  # Should be in cm^2
        return A * self.area_cap

    def calc_coin_cell_wt(self, d):
        A = np.pi * (d / 2)**2
        return A * self.coat_wt

    def calc_pouch_cell_cap(self, A):
        return A * self.area_cap


if __name__ == '__main__':
    nmc811 = ElectrodeMaterial(spec_cap=183, coat_wt=20, loading=0.955)
    gr = ElectrodeMaterial(spec_cap=340, coat_wt=12.89, loading=0.957)
    sigr05 = ElectrodeMaterial(spec_cap=393, coat_wt=11.22, loading=0.945)
    sigr10 = ElectrodeMaterial(spec_cap=445, coat_wt=9.95, loading=0.945)
    sigr15 = ElectrodeMaterial(spec_cap=498, coat_wt=8.91, loading=0.945)
    nm_list = ['nmc', 'gr', 'si05', 'si10', 'si15']
    el_dct = dict(zip(nm_list, [nmc811, gr, sigr05, sigr10, sigr15]))
    cap_dct = {k: el.calc_coin_cell_cap(1.8) for k, el in el_dct.items()}
    wt_dct = {k: el.calc_coin_cell_wt(1.8) for k, el in el_dct.items()}
    c_rates = {f'c_by{c}': cap_dct['nmc'] / c for c in [1, 5, 10, 20]}

