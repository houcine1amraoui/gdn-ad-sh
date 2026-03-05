import numpy as np
from sklearn.preprocessing import StandardScaler

def create_sliding_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size):
        windows.append(data[i:i + window_size])
    return np.array(windows)

def reshape_for_gdn(windows):
    # windows: (samples, window, sensors)
    # Convert to (samples, sensors, window)
    return np.transpose(windows, (0, 2, 1))


def normalize(actor1_df, actor2_df):
    sensor_columns = [c for c in actor1_df.columns if c != "Timestamp"]
    scaler = StandardScaler()
    train_array = scaler.fit_transform(actor1_df[sensor_columns].values)
    test_array = scaler.transform(actor2_df[sensor_columns].values)
    return train_array, test_array