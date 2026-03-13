import matplotlib.pyplot as plt
import os

def plot_scores(scores, exp_dir):

    plt.figure(figsize=(15,5))
    plt.plot(scores, label="Anomaly Scores")
    plt.legend()
    plt.title("Anomaly Scores")

    path = os.path.join(exp_dir, "anomaly_plot.png")
    plt.savefig(path)

    plt.close()

def plot_actor_comparison(errors1, errors2, threshold, exp_dir):

    plt.figure(figsize=(15,5))

    plt.plot(errors1, label="Actor 1 (Normal)")
    plt.plot(range(len(errors1), len(errors1)+len(errors2)), errors2, label="Actor 2 (Test)")

    plt.axhline(threshold, linestyle="--", label="Threshold")

    plt.legend()
    plt.title("Anomaly Scores")

    plt.savefig(os.path.join(exp_dir, "anomaly_plot.png"))
    plt.close()