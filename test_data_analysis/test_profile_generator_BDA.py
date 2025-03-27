import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt


def profile_gen(t_tot):
    dt = min(0.1, t_tot/100)
    time = np.arange(0, t_tot + dt, dt)
    curr = np.zeros_like(time)
    curr[time > t_tot / 4] = -6
    return time, curr


def repeat_profile(t, curr, n_reps):
    dt = np.diff(t).mean()
    time = np.arange(0, n_reps*(t.max() + dt), dt)
    i_rep = np.tile(curr, n_reps)
    return time, i_rep


def save_profile(time, curr):
    plt.plot(time, curr)
    today = dt.datetime.now().strftime('%Y-%m-%d')
    op_dir = r'\\sol.ita.chalmers.se\groups\eom-et-alla\Research\Aline_BAD\BDA_testcodes'
    op_file_full = 'BDA_full_profile_t_tot_{}s_{}.csv'.format(int(time.max()), today)
    op_file_curr = 'BDA_curr_profile_t_tot_{}s_{}.csv'.format(int(time.max()), today)
    np.savetxt(os.path.join(op_dir, op_file_full), np.array([time, curr]).T, delimiter=',', fmt='%.4f')
    np.savetxt(os.path.join(op_dir, op_file_curr), curr, delimiter=',', fmt='%.4f')
    print('Average current is {}'.format(np.trapezoid(curr, time) / time.max()))
    return None


if __name__ == '__main__':
    t_arr = 2**np.arange(0, 8, 1)
    for t in t_arr:
        time, curr = profile_gen(t)
        # save_profile(time, curr)
