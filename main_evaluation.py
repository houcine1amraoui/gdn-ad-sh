import yaml
import argparse

from src.utils.seed import set_seed
from src.evaluation.compute_errors import compute_errors
from src.evaluation.compute_scores import compute_scores
# from src.evaluation.compute_metrics import compute_metrics

def main_evaluation():
    # 1. Set configuration
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])

    # parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root_dir", type=str)
    args = parser.parse_args()

    # override project_root_directory
    if args.project_root_dir:
        config["project_root_dir"] = args.project_root_dir

    # compute_errors(config)
    compute_scores(config)
    # evalutation_pipeline(config)
    # print("train mean and std: ", scores["train"].mean(), scores["train"].std())
    # print("val mean and std: ", scores["val"].mean(), scores["val"].std())
    # print("actor 2 test mean and std: ", scores["actor2_test"].mean(), scores["actor2_test"].std())
    # print("actor 1 test mean and std: ", scores["actor1_test"].mean(), scores["actor1_test"].std())
    # compute_metrics(scores, config)


    # compute_metrics_with_pot_thresholding(scores)

if __name__ == "__main__":
    main_evaluation()

