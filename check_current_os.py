import platform
from pathlib import Path, PureWindowsPath, PurePosixPath


def set_correct_path_type(input_path):
    current_os = platform.system()
    update_need = 0
    if current_os == 'Windows' and len(Path(input_path).parts) > 1:
        if Path(input_path).parts[1] == 'mnt':
            print('Posix path provided on Windows, return PurePosixPath, update needed')
            update_need = 1
            return PurePosixPath(input_path), update_need
        else:
            print('Windows path provided on Windows, no update needed')
            update_need = 0
            return PureWindowsPath(input_path), update_need
    elif current_os == 'Windows' and len(Path(input_path).parts) == 1:
        print('Posix path provided on Windows, return PurePosixPath, update needed')
        update_need = 1
        return PurePosixPath(input_path), update_need
    elif current_os == 'Linux' and len(Path(input_path).parts) > 1:
        print('Posix path provided on Linux, no update needed')
        update_need = 0
        return PurePosixPath(input_path), update_need
    elif current_os == 'Linux' and len(Path(input_path).parts) == 1:
        print('Windows path provided on Linux, return PureWindowsPath and update')
        update_need = 1
        return PureWindowsPath(input_path), 1


# Function to get the correct path
def get_correct_path(input_path):
    # Define the base paths for Windows and Linux
    windows_base_path = PureWindowsPath(r'\\sol.ita.chalmers.se\groups\batt_lab_data')
    linux_base_path = PurePosixPath('/mnt/batt_lab_data')

    # Determine the operating system
    current_os = platform.system()
    input_path, update_bool = set_correct_path_type(input_path)

    # If the path is absolute and in Windows format
    if input_path.is_absolute() and current_os == 'Linux' and update_bool:
        # Convert the Windows absolute path to Linux absolute path
        relative_path = input_path.relative_to(windows_base_path)
        correct_path = linux_base_path / relative_path
    elif input_path.is_absolute() and current_os == 'Windows' and update_bool:
        # Convert the Linux absolute path to Windows absolute path
        relative_path = input_path.relative_to(linux_base_path)
        correct_path = windows_base_path / relative_path
    else:
        # Otherwise let path through
        correct_path = input_path

    return correct_path


def get_base_path_batt_lab_data():
    current_os = platform.system()
    if current_os == 'Windows':
        return "//sol.ita.chalmers.se/groups/batt_lab_data"
    elif current_os == 'Linux':
        return "/batt_lab_data"


if __name__ == '__main__':
    # Example usage
    input_path_windows = r'\\sol.ita.chalmers.se\groups\batt_lab_data\stat_test\processed_data\Test1_1.pkl'
    input_path_posix = '/mnt/batt_lab_data/stat_test/processed_data/Test1_1.pkl'
    correct_path_from_win = get_correct_path(input_path_windows)
    correct_path_from_posix = get_correct_path(input_path_posix)

    print(f"The correct path from windows input is: {correct_path_from_win}\nFrom "
          f"posix input it is {correct_path_from_posix}")
