import matplotlib as mpl


def fix_mpl_backend():
    if not mpl.get_backend() == 'Qt5Agg':
        mpl.use('Qt5Agg')
        print(f'Backend set to {mpl.get_backend()}')
    return None


def main():
    fix_mpl_backend()


if __name__ == '__main__':
    main()
