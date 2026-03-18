from src.utils.data_sampling import data_sampling
from main_preprocess import main_preprocess
from main_train import main_train
from main_evaluation import main_evaluation

def main():
    path = "data/raw/BREMaster.csv"
    save_path = "data/sample/BREMaster-3.csv"
    data_sampling(path, save_path)

    main_preprocess()
    main_train()
    main_evaluation()


if __name__ == "__main__":
    main()