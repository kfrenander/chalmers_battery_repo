from pyDRTtools.runs import EIS_object, Bayesian_run, simple_run, BHT_run
import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


class DrtSettings:
    def __init__(self, log_lambda_init_guess=-7, shape_control='FWHM Coefficient', rbf_type='Inverse Quadratic',
                 cv_type='mGCV', data_used='Combined Re-Im Data', use_inductive_part=0, shape_control_coeff=0.5,
                 use_data=1, der_used = '1st order', reg_param = 1E-3, NMC_sample = 2000):
        self.log_lambda_init_guess = log_lambda_init_guess
        self.shape_control = shape_control
        self.shape_control_coeff = shape_control_coeff
        self.rbf_type = rbf_type
        self.cv_type = cv_type
        self.data_used = data_used
        self.use_inductive = use_inductive_part
        self.use_data = use_data
        self.der_used = der_used
        self.reg_param = reg_param
        self.NMC_sample = NMC_sample

    def export_to_file(self, file_path):
        """Exports the current settings to a specified file for logging."""
        # Check if the file already exists and increment the ID number if it does
        base_name, ext = os.path.splitext(file_path)
        i = 1
        new_file_path = file_path
        while os.path.exists(new_file_path):
            new_file_path = f"{base_name}{i}{ext}"
            i += 1

        with open(new_file_path, 'w') as f:
            f.write("DRT Fit Settings:\n")
            f.write(f"Log Lambda Initial Guess: \t{self.log_lambda_init_guess}\n")
            f.write(f"Shape Control: \t{self.shape_control}\n")
            f.write(f"Shape Control Coefficient: \t{self.shape_control_coeff}\n")
            f.write(f"RBF Type: \t{self.rbf_type}\n")
            f.write(f"Cross-Validation Type: \t{self.cv_type}\n")
            f.write(f"Inductance Used: \t{self.use_inductive}\n")
            f.write(f"Data Used: \t{self.data_used}\n")
            f.write(f"Regularization Derivative: \t{self.der_used}\n")
            f.write(f"Regularization Parameter: \t{self.reg_param}\n")
            f.write(f"Number of Samples: \t{self.NMC_sample}\n")


class DrtAnalyzerWrapper:
    def __init__(self, log_file, settings):
        self.log_file = log_file
        self.settings = settings
        self.drt_data_set = EIS_object.from_file(self.log_file)
        self.df = pd.read_csv(self.log_file, names=['Freq', 'Real', 'Imag'], header=None)
        self.cell_nbr = re.search(r"\d{3}", self.log_file).group()
        self.current_soc = extract_soc(self.log_file)
        self.test_condition = None
        self.id_test_condition()

    def id_test_condition(self):
        base_path = get_base_path_batt_lab_data()
        db_df = pd.read_excel(os.path.join(base_path, 'neware_test_inventory.xlsx'))
        self.test_condition = db_df[db_df.CELL_ID == float(self.cell_nbr)]['TEST_CONDITION'].iloc[0]

    def run_bayesian(self):
        self.drt_data_set = Bayesian_run(self.drt_data_set,
                                         rbf_type=self.settings.rbf_type,
                                         data_used=self.settings.data_used,
                                         induct_used=self.settings.use_inductive,
                                         der_used=self.settings.der_used,
                                         cv_type=self.settings.cv_type,
                                         reg_param=self.settings.reg_param,
                                         shape_control=self.settings.shape_control,
                                         coeff=self.settings.shape_control_coeff,
                                         NMC_sample=self.settings.NMC_sample
                                         )

    def run_simple(self):
        self.drt_data_set = simple_run(self.drt_data_set,
                                       rbf_type=self.settings.rbf_type,
                                       data_used=self.settings.data_used,
                                       induct_used=self.settings.use_inductive,
                                       der_used=self.settings.der_used,
                                       cv_type=self.settings.cv_type,
                                       reg_param=self.settings.reg_param,
                                       shape_control=self.settings.shape_control,
                                       coeff=self.settings.shape_control_coeff
                                       )

    def run_bht(self):
        self.drt_data_set = BHT_run(self.drt_data_set,
                                    rbf_type=self.settings.rbf_type,
                                    der_used=self.settings.der_used,
                                    shape_control=self.settings.shape_control,
                                    coeff=self.settings.shape_control_coeff
                                    )

    def plot_fit(self):
        """ Handles plotting of fit in Nyquist plot"""
        fig, ax = plt.subplots(1, 1)
        ax.scatter(self.df['Real'].values, -self.df['Imag'].values, color='red', label='Raw data')
        ax.plot(self.drt_data_set.mu_Z_re, -self.drt_data_set.mu_Z_im, color='black', label='Recovered Nyquist plot')
        ax.legend()
        ax.set_xlabel(r'$Z_{\rm re} [\Omega]$')
        ax.set_ylabel(r'$-Z_{\rm im} [\Omega]$')
        ax.set_title(f'{self.cell_nbr} at {self.current_soc}')

    def plot_individual_drt(self, label=''):
        """Handles plotting of DRT results"""
        fig, ax = plt.subplots(1, 1)
        plt_label = f'DRT {label} at {self.current_soc}'
        if self.drt_data_set.method == 'simple':
            ax.semilogx(self.drt_data_set.tau_fine, self.drt_data_set.gamma,
                        label=plt_label,
                        linewidth=1)
        elif self.drt_data_set.method == 'credit':
            ax.semilogx(self.drt_data_set.tau_fine, self.drt_data_set.gamma,
                        label='MAP',
                        color='black',
                        linewidth=1)
            ax.semilogx(self.drt_data_set.tau_fine, self.drt_data_set.mean,
                        label='Mean',
                        color='darkblue',
                        linewidth=1)
            ax.fill_between(self.drt_data_set.tau_fine, self.drt_data_set.lower_bound, self.drt_data_set.upper_bound,
                            facecolor='lightblue', alpha=0.7)
        ax.set_xlabel(r'$\tau$  $[s]$')
        ax.set_ylabel(r'$\gamma$  $[\Omega]$')
        ax.legend()
        return fig, ax


def extract_soc(log_file):
    """Extracts the state of charge (SOC) from the log file name."""
    match = re.search(r'(SOC\d+)(?:_afterwait)?', log_file)
    if match:
        return match.group(1) + ('_afterwait' if '_afterwait' in log_file else '_shortwait')


def collect_eis_files(parent_folder):
    from natsort import natsorted
    import glob
    cell_data = {}
    # Get all subfolders starting with 'Cell' under the parent folder
    for subfolder in natsorted(os.listdir(parent_folder)):
        # if subfolder.startswith('Cell'):
        # Extract cell number from the folder name
        cell_number = subfolder
        # Construct the full path to the subfolder
        subfolder_path = os.path.join(parent_folder, subfolder)

        # Ensure it's a directory
        if os.path.isdir(subfolder_path):
            # Find all files in this subfolder containing 'for_DRT'
            eis_for_drt_files = glob.glob(os.path.join(subfolder_path, '*for_DRT*'))

            # Add the list of files to the dictionary with the cell number as key
            cell_data[cell_number] = eis_for_drt_files
    return cell_data


if __name__ == '__main__':
    from check_current_os import get_base_path_batt_lab_data
    from natsort import natsorted
    BASE_PATH = get_base_path_batt_lab_data()
    settings = DrtSettings(use_inductive_part=2)
    settings.export_to_file("C:/Work/test_output.txt")
    eis_data_files = collect_eis_files(os.path.join(BASE_PATH, 'pulse_chrg_test/EIS_for_DRT'))
    db_df = pd.read_excel(os.path.join(BASE_PATH, 'neware_test_inventory.xlsx'))
    drt_analyser_obj = DrtAnalyzerWrapper(eis_data_files['180'][0], settings)
    drt_analyser_dict = {cell: {extract_soc(file): DrtAnalyzerWrapper(log_file=file, settings=settings)
                                for file in files if '_after' in file}
                         for cell, files in eis_data_files.items()}
    # for cell, dct in drt_analyser_dict.items():
    #     for soc, drt_obj in natsorted(dct.items()):
    #         drt_obj.run_bayesian()
