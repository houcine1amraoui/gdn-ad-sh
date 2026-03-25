import yaml
import numpy as np
import json
import torch.optim as optim

from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.experiment import create_experiment_folder, get_best_experiment
from src.models.builders import build_gdn_model
from src.evaluation.load_checkpoint import load_checkpoint
from src.evaluation.pipeline import create_evaluation_dataloaders, plot_anomaly_score_distributions, compute_anomaly_score_and_detection
from src.evaluation.pipeline import compute_anomaly_scores, errors_computation_pipeline
import matplotlib.pyplot as plt
import argparse

from scipy.stats import ks_2samp
from src.evaluation.viz import plot_boxplot
from src.evaluation.viz import plot_anomaly_scores_distribution

def main_evaluation():
    # 1. Set configuration
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])

    # parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_folder", type=str)
    parser.add_argument("--experiments_folder", type=str)
    args = parser.parse_args()

    # override processed data folder path
    if args.processed_folder:
        config["dataset"]["processed_folder"] = args.processed_folder

    # override experiments_folder
    if args.experiments_folder:
        config["experiments_folder"] = args.experiments_folder

    experiments_folder = config["experiments_folder"]

    exp_dir = create_experiment_folder(config, f"{experiments_folder}/eval")
    best_exp_path, _ = get_best_experiment(experiments_folder+"/train")

    errors_computation_pipeline(config, best_exp_path, exp_dir)

    # Compute a global (normalized) anomaly score per timestamp
    # Compute scores + detection
    # scores, detection_rate, false_positive_rate = compute_anomaly_score_and_detection(
    #     topk_ratio=0.4,
    #     threshold_percentile=99
    # )

    scores = compute_anomaly_scores(exp_dir)
    plot_anomaly_scores_distribution(scores)
    # Plot distributions
    # plot_anomaly_score_distributions(scores)
    # 7. Compute detection rates
    # detection_rate, fp_rate = (
    # compute_detection_rates(scores["train_scores"], scores[]"actor2_test_scores", actor1_test_scores))
    # print("detection rate: ", detection_rate, "false positive rate", false_positive_rate)

    # plot_boxplot(train_scores, val_scores, actor2_test_scores, actor1_test_scores)
    # print("Train mean:", train_scores.mean())
    # print("Val mean:", val_scores.mean())
    # print("Actor 2 Test mean:", actor2_test_scores.mean())
    # print("Actor 1 Test mean:", actor1_test_scores.mean())

    # print("Train 95th percentile:", np.percentile(train_scores, 95))
    # print("Test 95th percentile:", np.percentile(actor2_test_scores, 95))
    # print("Test 95th percentile:", np.percentile(actor1_test_scores, 95))

    # full_scores = np.concatenate([
    #     train_scores,
    #     actor2_test_scores,
    #     actor1_test_scores
    # ])

    # plt.plot(full_scores)
    # plt.show()

    # Visualizae scores distribution
    # plt.figure(figsize=(15,5))
    # plt.plot(scores["train"], label="Actor 1 (Normal)")
    # plt.plot(scores["val"], label="Validation")
    # plt.plot(scores["actor2_test"], label="Actor 2 (Test)")
    # plt.plot(scores["actor1_test"], label="Actor 1 (Test)")
    # plt.legend()
    # plt.title("Anomaly Scores")
    # plt.show()

    # 
    # stat, p = ks_2samp(train_scores, test_scores)
    # print("KS statistic:", stat)
    # print("p-value:", p)
    # # 6. Visualization
    # # plot_scores(test_errors_actor1, exp_dir)
    # # plot_scores(test_errors_actor1, test_errors_actor2, exp_dir)

if __name__ == "__main__":
    main_evaluation()

