import os
from datetime import datetime
import yaml

def create_experiment_folder(config):

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    exp_dir = os.path.join("experiments", f"exp_{timestamp}")

    os.makedirs(exp_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    return exp_dir