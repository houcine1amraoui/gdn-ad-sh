import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    sensor_columns = [c for c in df.columns if c != "Timestamp"]
    return df, sensor_columns