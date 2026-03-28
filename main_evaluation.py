import yaml
import os
import argparse

from src.utils.seed import set_seed
from src.evaluation.viz import plot_boxplot, plot_anomaly_scores_distribution, plot_bins
from src.evaluation.compute_errors import compute_errors
from src.evaluation.compute_scores import compute_scores
from src.evaluation.compute_metrics import compute_metrics

def main_evaluation():
    # 1. Set configuration
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])

    # parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_folder", type=str)
    parser.add_argument("--train_experiments_main_folder", type=str)
    parser.add_argument("--eval_results_folder", type=str)
    args = parser.parse_args()

    # override processed data folder
    if args.processed_data_folder:
        config["dataset"]["processed_data_folder"] = args.processed_data_folder

    # override train_experiments_main_folder
    if args.train_experiments_main_folder:
        config["training"]["train_experiments_main_folder"] = args.train_experiments_main_folder

    # override eval_results_folder
    if args.eval_results_folder:
        config["evaluation"]["eval_results_folder"] = args.eval_results_folder

    eval_results_folder = config["evaluation"]["eval_results_folder"]
    # Create a folder if it doesn't exist
    os.makedirs(eval_results_folder, exist_ok=True)

    # compute_errors(config)
    scores = compute_scores(config)
    print("train mean and std: ", scores["train"].mean(), scores["train"].std())
    print("val mean and std: ", scores["val"].mean(), scores["val"].std())
    print("actor 2 test mean and std: ", scores["actor2_test"].mean(), scores["actor2_test"].std())
    print("actor 1 test mean and std: ", scores["actor1_test"].mean(), scores["actor1_test"].std())
    # compute_metrics(scores, config)
    # plot_anomaly_scores_distribution(scores, config)
    # plot_boxplot(scores, eval_results_folder)
    # plot_bins(scores, config)


    # compute_metrics_with_pot_thresholding(scores)

if __name__ == "__main__":
    main_evaluation()

