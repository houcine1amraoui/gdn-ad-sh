import matplotlib.pyplot as plt

def plot_anomaly_scores_distribution(scores):
    plt.figure(figsize=(15,5))

    # lengths
    n_train = len(scores["train"])
    n_val = len(scores["val"])
    n_actor2 = len(scores["actor2_test"])
    n_actor1 = len(scores["actor1_test"])

    # x ranges (shifted)
    x_train = range(0, n_train)
    x_val = range(n_train, n_train + n_val)
    x_actor2 = range(n_train + n_val, n_train + n_val + n_actor2)
    x_actor1 = range(n_train + n_val + n_actor2, n_train + n_val + n_actor2 + n_actor1)

    # plot
    plt.plot(x_train, scores["train"], label="Actor 1 (Train)")
    plt.plot(x_val, scores["val"], label="Validation")
    plt.plot(x_actor2, scores["actor2_test"], label="Actor 2 (Test)")
    plt.plot(x_actor1, scores["actor1_test"], label="Actor 1 (Test)")

    # optional: vertical separators
    plt.axvline(n_train, linestyle="--")
    plt.axvline(n_train + n_val, linestyle="--")
    plt.axvline(n_train + n_val + n_actor2, linestyle="--")

    plt.legend()
    plt.title("Anomaly Scores (Concatenated Timeline)")
    plt.show()

def plot_boxplot(train_scores, val_scores, actor2_test_scores, actor1_test_scores):
    plt.figure(figsize=(8, 5))

    plt.boxplot([train_scores, val_scores, actor2_test_scores, actor1_test_scores],
                labels=["Train", "Val", "Test A", "Test B"],
                showfliers=False)

    plt.title("Score Distribution Comparison")
    plt.ylabel("Score")

    plt.show()