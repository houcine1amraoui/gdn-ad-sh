import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd

def rank_sensor_noise(df, timestamp_col="Timestamp"):
    """
    Rank sensors by noise level using variance of first differences.
    """
    if timestamp_col in df.columns:
        df = df.drop(columns=[timestamp_col])
    noise_scores = {}
    for col in df.columns:
        # compute first difference
        diff = df[col].diff()
        # noise score
        noise_scores[col] = diff.var()
    noise_df = (
        pd.DataFrame.from_dict(noise_scores, orient="index", columns=["noise_score"])
        .sort_values("noise_score", ascending=False)
    )
    return noise_df

def average_sensor_activation_per_day(df):
    """
    Average sensor activation per day
    """
    # Identify sensor columns (exclude timestamp and date)
    sensor_columns = [c for c in df.columns if c != "Timestamp"]
    daily_mean = df.groupby('Timestamp')[sensor_columns].mean().mean(axis=1)

    # plot
    plt.figure()
    plt.plot(daily_mean.index, daily_mean.values)
    plt.xticks(rotation=45)
    plt.title("Average Sensor Activation per Day")
    plt.xlabel("Time")
    plt.ylabel("Mean Activation")
    plt.tight_layout()
    plt.show()

def corr_heatmap(df, start_string, end_string):
    """
    Correlation Heatmap (1 Day Actor 1)
    """
    # Select one specific day
    start = pd.to_datetime(start_string)
    end   = pd.to_datetime(end_string)
    day_data = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)]

    # # Identify sensor columns (exclude timestamp and date)
    sensor_columns = [c for c in df.columns if c != "Timestamp"]
    corr = day_data[sensor_columns].corr()

    plt.figure()
    plt.imshow(corr.values)
    plt.title("Correlation Matrix - 1 Day Actor 1")
    plt.colorbar()
    plt.tight_layout()
    plt.show()