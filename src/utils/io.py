import numpy as np
import os

def save_scores(scores, exp_dir):

    path = os.path.join(exp_dir, "anomaly_scores.npy")
    np.save(path, scores)