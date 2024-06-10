import platform
import pathlib
import pandas as pd


win_str = 'Windows'
linux_str = 'Linux'

curr_pf = platform.platform()
file_path = pathlib.PureWindowsPath("stat_test\processed_data\Test2_2.pkl")

if win_str in curr_pf:
    base_path = r"\\sol.ita.chalmers.se\groups\batt_lab_data"
    my_file = pathlib.PureWindowsPath(base_path, *file_path.parts)
elif linux_str in curr_pf:
    base_path = "/mnt/batt_lab_data"
    my_file = pathlib.PurePosixPath(base_path, *file_path.parts)
else:
    print(f'Unknown OS, please check for {curr_pf}')

df = pd.read_pickle(my_file)
