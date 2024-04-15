import logging

import pandas as pd
import numpy as np
import warnings
import os

from pairwise_formulation.pa_basics.import_data import \
    kfold_splits, rearrange_feature_orders_by_popularity

from run_utils import run_standard_approach, run_pairwise_cnn
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


if __name__ == '__main__':
    #####run parameters #####
    run_msg = "Run 1024 inputs conv, with ordered inputs."  # REMEMBER TO UPDATE RESULT FILE NAME

    logging.info(run_msg)
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
        whole_dataset = pd.read_csv(input_dir + filename, index_col=None)
        whole_dataset = whole_dataset.drop(columns=["molecule_id"])

        for size in sizes:
            train_test = whole_dataset.sample(n=size, random_state=0)
            train_test = train_test.to_numpy().astype(np.float64)
            train_test = rearrange_feature_orders_by_popularity(train_test)

            train_test_splits_dict = kfold_splits(train_test=np.array(train_test), fold=10)
            metrics_per_fold = []
            logging.info(f"Starting {filename}, size {size}.")
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
            np.save(output_dir + "/run3.npy", np.array(metrics_per_dataset))
        logging.info(f"Finished {filename}, size {size}.")