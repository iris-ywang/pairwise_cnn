import pandas as pd
import numpy as np
import warnings
import os

from pairwise_formulation.pa_basics.import_data import \
    kfold_splits

from run_utils import run_standard_approach, run_pairwise_cnn
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    #####run parameters #####
    input_dir = "./data/qsar_boolean_large_datasets/"
    output_dir = "./output/run_boolean_chembl/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    list_filename = [
        "data_CHEMBL203.csv",
        "data_CHEMBL204.csv",
        "data_CHEMBL228.csv",
    ]
    sizes = [
        200,
        500,
    ]

    ######################################################

    metrics_per_dataset = []
    for filename in list_filename:
        whole_dataset = pd.read_csv(input_dir + filename)
        whole_dataset = whole_dataset.drop(columns=["molecule_id"])

        for size in sizes:
            train_test = whole_dataset.sample(n=size, random_state=0)
            train_test_splits_dict = kfold_splits(train_test=np.array(train_test), fold=10)
            metrics_per_fold = []
            for fold_id, foldwise_data in train_test_splits_dict.items():
                m_rank_sa, m_est_sa = run_standard_approach(
                    foldwise_data=foldwise_data,
                    ML_reg=RandomForestRegressor(random_state=2),
                )
                m_rank_pa, m_est_pa = run_pairwise_cnn(
                        foldwise_data=foldwise_data,
                    )
                metrics_per_fold.append(
                    [m_rank_sa] + m_rank_pa + [m_est_sa, m_est_pa]
                )  # list [5, 6] + [2, 6] = [7, 6]
            metrics_per_dataset.append(metrics_per_fold)
            np.save(output_dir + "/run2.npy", np.array(metrics_per_dataset))
