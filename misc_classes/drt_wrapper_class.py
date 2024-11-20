from pyDRTtools.runs import EIS_object, Bayesian_run, simple_run, BHT_run
import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import dill


class CellStyle:
    def __init__(self, color, marker):
        self.color = color
        self.marker = marker


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

    def plot_drt(self, fig=None, ax=None, label=''):
        plt_label = f'DRT for {self.test_condition} {label}at {self.current_soc}'
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)
        if self.drt_data_set.method == 'simple':
            ax.semilogx(self.drt_data_set.tau_fine, self.drt_data_set.gamma,
                        label=plt_label,
                        linewidth=1)
        elif self.drt_data_set.method == 'credit':
            soc_lvl = re.search(r'\d+', self.current_soc).group()
            drt_lbl = f'MAP {self.test_condition} at SOC{soc_lvl}'
            ax.semilogx(self.drt_data_set.tau_fine, self.drt_data_set.gamma,
                        label=drt_lbl,
                        linewidth=1)
        ax.set_xlabel(r'$\tau$  $[s]$')
        ax.set_ylabel(r'$\gamma$  $[\Omega]$')
        return fig, ax

    def dump_drt(self, path=None, file_tag=''):
        import dill
        if path is None:
            output_dir = "Z:/Provning/Analysis/pulse_charge/DRT/full_data"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        fname = f'DRT_class_{file_tag}{self.cell_nbr}_at_{self.current_soc}.pkl'
        path = os.path.join(output_dir, fname)
        with open(path, 'wb') as f:
            dill.dump(self, f)

    def export_drt(self, path=None, file_tag=''):
        import csv
        if path is None:
            output_dir = "Z:/Provning/Analysis/pulse_charge/DRT/fit_data"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        if self.drt_data_set.method == 'credit':
            if path is None:
                fname = f'Bayesian_DRT_export_{file_tag}{self.cell_nbr}_at_{self.current_soc}.txt'
                path = os.path.join(output_dir, fname)
            with open(path, 'w', newline='') as save_file:
                writer = csv.writer(save_file)

                # first save L and R
                writer.writerow(['L', self.drt_data_set.L])
                writer.writerow(['R', self.drt_data_set.R])
                writer.writerow(['tau', 'MAP', 'Mean', 'Upperbound', 'Lowerbound'])

                # after that, save tau, gamma, mean, upper bound, and lower bound
                for n in range(self.drt_data_set.out_tau_vec.shape[0]):
                    writer.writerow([self.drt_data_set.out_tau_vec[n], self.drt_data_set.gamma[n],
                                     self.drt_data_set.mean[n], self.drt_data_set.upper_bound[n],
                                     self.drt_data_set.lower_bound[n]])
        elif self.drt_data_set.method == 'simple':
            if path is None:
                fname = f'Simple_DRT_export_{file_tag}{self.cell_nbr}_at_{self.current_soc}.txt'
                path = os.path.join(output_dir, fname)
            with open(path, 'w', newline='') as save_file:
                writer = csv.writer(save_file)

                # first save L and R
                writer.writerow(['L', self.drt_data_set.L])
                writer.writerow(['R', self.drt_data_set.R])
                writer.writerow(['tau', 'gamma'])

                # after that, save tau and gamma
                for n in range(self.drt_data_set.out_tau_vec.shape[0]):
                    writer.writerow([self.drt_data_set.out_tau_vec[n], self.drt_data_set.gamma[n]])
        elif self.drt_data_set.method == 'none':
            print(f'No fit available for {self.cell_nbr} at {self.current_soc}')


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
    import time
    import itertools
    BASE_PATH = get_base_path_batt_lab_data()
    settings = DrtSettings(use_inductive_part=1)
    # settings.export_to_file("C:/Work/test_output.txt")
    eis_data_files = collect_eis_files(os.path.join(BASE_PATH, 'pulse_chrg_test/EIS_for_DRT'))
    db_df = pd.read_excel(os.path.join(BASE_PATH, 'neware_test_inventory.xlsx'))
    drt_analyser_obj = DrtAnalyzerWrapper(eis_data_files['180'][0], settings)
    drt_analyser_dict = {cell: {extract_soc(file): DrtAnalyzerWrapper(log_file=file, settings=settings)
                                for file in files if '_after' in file}
                         for cell, files in eis_data_files.items()}
    i = 1
    global_start = time.time()
    for cell, dct in drt_analyser_dict.items():
        for soc, drt_obj in natsorted(dct.items()):
            soc_val = re.search(r'SOC(\d+)', soc).group()
            soc_lvls = range(0, 110, 10)
            if soc_val in [f'SOC{val}' for val in soc_lvls]:
                exp_vals = len(drt_analyser_dict) * len(soc_lvls)
                print(f'Fit number {i} of expected {exp_vals}\nPerforming Bayesian fit for {soc_val} from {cell}. \n')
                tic = time.time()
                drt_obj.run_bayesian()
                toc = time.time()
                print(f'Elapsed time is {toc - tic:.2f}s for iteration {i}\n\n')
                i += 1
                try:
                    drt_obj.export_drt(file_tag='with_inductive_')
                except FileExistsError:
                    print(f'Case {cell} at {soc_val} already exported')
                except Exception as e:
                    print(f'Error {e} for {cell} at {soc_val}')

    plot_boolean = 1
    if plot_boolean:
        plt.style.use(['widthsixinches'])
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Times New Roman",
            "text.latex.preamble": r'\usepackage{siunitx}'
        })
        output_dir = "Z:/Provning/Analysis/pulse_charge/DRT/figures_w_inductance"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get the default matplotlib color cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Define the markers to cycle through for each cell
        marker_cycle = itertools.cycle(['o', 's', '^', 'D', 'P', '*', 'x', '+', 'v'])

        # Create a style dictionary using the default matplotlib colors and cycling through markers
        style_dict = {
            str(cell): CellStyle(color=color, marker=next(marker_cycle))
            for cell, color in zip([180, 182, 184, 186, 190, 192, 194, 196, 199], color_cycle)
        }
        x_fig, y_fig = plt.rcParams['figure.figsize']
        plt.rcParams['figure.autolayout'] = False

        # Create a 2x1 grid of subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2 * x_fig, 2 * y_fig))  # Increase width and height

        # List of cells to plot
        # 180 is 10mHz
        # 182 is 320mHz
        # 184 is 500mHz
        # 186 is reference cycled
        # 199 is reference fresh
        # 190 is 1000mHz no pulse dchg
        # 192 is 1000mHz
        # 194 is 100mHz no pulse dchg
        # 196 is 100mHz
        # 199 is fresh cell
        # 210 is replicate fresh cell
        cells_to_plot = [180, 192, 196, 199]

        label_fnt_sz = 15.5
        soc_fltr = 'SOC50'
        # Loop through your data and plot on both subplots
        for cell, dct in drt_analyser_dict.items():
            if int(cell) in cells_to_plot:
                for soc, drt_obj in natsorted(dct.items()):
                    if soc_fltr in soc and drt_obj.drt_data_set.method == 'credit':
                        style = style_dict[cell]
                        # Plot on the first subplot
                        fig, ax2 = drt_obj.plot_drt(fig=fig, ax=ax2)
                        ax2.lines[-1].set_color(style.color)
                        # Scatter on the second subplot
                        lbl = f'{drt_obj.test_condition} at {drt_obj.current_soc.split("_")[0]}'
                        ax1.scatter(drt_obj.df.Real, -drt_obj.df.Imag,
                                    label=lbl,
                                    s=6,
                                    color=style.color,
                                    marker=style.marker)
                        ax1.axhline(0, color='black', linestyle='dotted', linewidth=1)

        # Shrink both axes to make space for the legend
        box1 = ax1.get_position()
        ax1.set_position([box1.x0, box1.y0, box1.width * 0.55, box1.height])
        ax1.set_xlabel(r'Re(Z) [$\unit{\ohm}$]', fontsize=label_fnt_sz)
        ax1.set_ylabel(r'-Im(Z) [$\unit{\ohm}$]', fontsize=label_fnt_sz)
        ymax = ax1.get_ylim()[1]  # Get current upper y-limit
        ax1.set_ylim(-0.005, ymax)

        box2 = ax2.get_position()
        ax2.set_position([box2.x0, box2.y0, box2.width * 0.55, box2.height])
        ax2.yaxis.label.set_fontsize(label_fnt_sz)
        ax2.xaxis.label.set_fontsize(label_fnt_sz)
        # Combine legends from both plots
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        # Create a single legend for both plots outside on the right
        fig.legend(handles1 + handles2, labels1 + labels2, loc='center left', bbox_to_anchor=(0.58, 0.5), fontsize=11.5)
        fname = f'EIS_and_DRT_for_cells_{"_".join([str(c) for c in cells_to_plot])}_and_{soc_fltr}.png'
        fig.savefig(os.path.join(output_dir, fname), dpi=400)
