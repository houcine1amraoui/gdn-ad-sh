import matplotlib.pyplot as plt

from src.utils.load_data import load_data
from exploration.exploration import rank_sensor_noise
def eda():
    # Noise level ranking
    df, _ = load_data("data/sample/BREMaster-sample2.csv")
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