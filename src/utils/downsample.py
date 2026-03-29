import pandas as pd

import pandas as pd

def detect_sensor_types(df):
    """
    Automatically detect binary and continuous sensors in a CSV time-series dataset.
    """
    df = df.drop(columns=["Timestamp"])

    binary_cols = []
    continuous_cols = []

    for col in df.columns:
        unique_vals = df[col].dropna().unique()

        # binary sensor condition
        if set(unique_vals).issubset({0, 1}):
            binary_cols.append(col)

        # small number of discrete values → likely event sensor
        elif len(unique_vals) <= 5:
            binary_cols.append(col)

        else:
            continuous_cols.append(col)

    return binary_cols, continuous_cols

def downsample_iot_timeseries(
    input_csv,
    output_csv,
    timestamp_col="Timestamp",
    rule="5s",
    continuous_cols=None,
    binary_cols=None
):
    """
    Downsample IoT time-series data while preserving events.

    continuous sensors → mean
    binary/event sensors → max
    """

    df = pd.read_csv(input_csv)

    # parse timestamp
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col)

    if continuous_cols is None:
        continuous_cols = []

    if binary_cols is None:
        binary_cols = []

    # aggregation dictionary
    agg_dict = {}

    for col in continuous_cols:
        agg_dict[col] = "mean"

    for col in binary_cols:
        agg_dict[col] = "max"

    # downsample
    df_down = df.resample(rule).agg(agg_dict)

    df_down = df_down.reset_index()

    df_down.to_csv(output_csv, index=False)

    print("Saved:", output_csv)
    print("Original rows:", len(df))
    print("Downsampled rows:", len(df_down))