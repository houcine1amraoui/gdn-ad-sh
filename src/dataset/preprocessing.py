import numpy as np

def create_sliding_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size):
        windows.append(data[i:i + window_size])
    return np.array(windows)

def reshape_for_gdn(windows):
    # windows: (samples, window, sensors)
    # Convert to (samples, sensors, window)
    return np.transpose(windows, (0, 2, 1))