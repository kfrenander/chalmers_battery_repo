import time
import numpy as np


class JitterLogger:
    def __init__(self, expected_interval_s, verbose=False):
        self.expected_interval = expected_interval_s
        self.timestamps = []
        self.jitter_us = []  # store jitter in microseconds
        self.verbose = verbose

    def record_callback(self):
        now = time.time()
        self.timestamps.append(now)

        if len(self.timestamps) >= 2:
            dt = self.timestamps[-1] - self.timestamps[-2]
            jitter = (dt - self.expected_interval) * 1e6  # in µs
            self.jitter_us.append(jitter)

            if self.verbose:
                print(f"[JitterLogger] Interval: {dt:.6f}s | Jitter: {jitter:+.2f} µs")

    def get_jitter_array(self):
        return np.array(self.jitter_us)

    def reset(self):
        self.timestamps.clear()
        self.jitter_us.clear()

    def save_to_csv(self, filename="jitter_log.csv"):
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Callback Index", "Jitter (us)"])
            for i, jitter in enumerate(self.jitter_us):
                writer.writerow([i, jitter])
        print(f"[JitterLogger] Saved jitter log to {filename}")
