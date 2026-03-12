import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    sensors = [c for c in df.columns if c != "Timestamp"]
    return df, sensors