import pandas as pd
import matplotlib.pyplot as plt

from exploration.exploration import rank_sensor_noise
def eda():
    # Noise level ranking
    df= pd.read_csv("data/sample/BREMaster-sample2.csv")
    noise_ranking = rank_sensor_noise(df)
    print(noise_ranking.head(10))

    noise_ranking[:10].plot(
        kind="bar",
        figsize=(10,5),
        legend=False
    )
    plt.title("Sensor Noise Ranking")
    plt.ylabel("Noise Score")
    plt.show()

if __name__ == "__main__":
    eda()