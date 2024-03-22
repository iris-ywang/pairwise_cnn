import numpy as np
import pandas as pd
from itertools import product
import logging

from pairwise_formulation.pa_basics.import_data import \
    filter_data, kfold_splits, transform_categorical_columns
from pairwise_formulation.pa_basics.all_pairs import pair_by_pair_id_per_feature

import torch
import torch.nn as nn

from pairwise_formulation.pairwise_data import PairwiseDataInfo
from pairwise_formulation.pa_basics.rating import rating_trueskill, rating_sbbr
from pairwise_formulation.evaluations.extrapolation_evaluation import ExtrapolationEvaluation
from pairwise_formulation.evaluations.stock_return_evaluation import calculate_returns


def pair_2d_pairwise_features(data: np.ndarray, pair_ids:list):
    data = np.array(data)

    pairwise_features = []
    pairwise_diff = []
    for pair_id in pair_ids:
        id_i, id_j = pair_id

        pairwise_features.append([
            data[id_i, 1:],
            data[id_j, 1:],
        ])
        pairwise_diff.append(data[id_i, 1:] - data[id_j, 1:])

    return np.array(pairwise_features), np.array(pairwise_diff)


class pairwise_cnn_model():
    def __init__(self):
        self.train_set = None
        self.test_set = None
        self.model = None
        self.pairwise_data_info: PairwiseDataInfo = None

    def enter_data(self, train_set, test_set, target_value_col_name="y"):
        self.train_set = train_set
        self.train_set = train_set
        self.pairwise_data_info = PairwiseDataInfo(
            train_set, test_set, target_value_col_name=target_value_col_name
        )
        self.pairwise_train = pair_2d_pairwise_features(
            data=self.pairwise_data_info.train_test,
            pair_ids=self.pairwise_data_info.c1_test_pair_ids,
        )  # (X_train, Y_train)
        self.pairwise_test = pair_2d_pairwise_features(
            data=self.pairwise_data_info.train_test,
            pair_ids=self.pairwise_data_info.c2_test_pair_ids,
        )  # (X_test, Y_test)

    def run_pairwise_cnn_model(self):
        model = cnn_model()
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        n_epochs = 20
        batch_size = 50
        for epoch in range(n_epochs):
            X_train, Y_train = self.pairwise_train
            n_batches = int(len(Y_train) / batch_size)
            for batch_id in range(n_batches):
                batch_indice = range(batch_id * batch_size, (batch_id + 1) * batch_size)
                # forward, backward, and then weight update
                X = torch.tensor(X_train[batch_indice], dtype=torch.float32)
                Y = torch.tensor(Y_train[batch_indice], dtype=torch.float32)
                Y_pred = model(X)
                loss = loss_fn(Y_pred, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mae = 0
            X_test, Y_test = self.pairwise_test
            n_batches = int(len(Y_test) / batch_size)
            for batch_id in range(n_batches):
                batch_indice = range(batch_id * batch_size, (batch_id + 1) * batch_size)

                X = torch.tensor(X_test[batch_indice], dtype=torch.float32)
                Y = torch.tensor(Y_test[batch_indice], dtype=torch.float32)
                Y_pred = model(X)
                mae += (Y - Y_pred).sum()

            print("Epoch %d: model MAE %.2f%%" % (epoch, mae))
        self.model = model


class cnn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 2, kernel_size=12, stride=1, padding=1)
        self.act1 = nn.ReLU()  # return positive response
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(2, 2, kernel_size=4, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=(2))

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(1014, 169)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(169, 1)

    def forward(self, x):
        # input 2*1024, output 2*1015
        x = self.act1(self.conv1(x))
        x = self.drop1(x)  # drop 30% of nodes

        # input 2*1015, output 2*1014
        x = self.act2(self.conv2(x))

        # input 2*1014, output 2*507
        x = self.pool2(x)

        # input 2*507, output 1014
        x = self.flat(x)

        # input 1014, output 169
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 10
        # input 169, output 10
        x = self.fc4(x)
        return x





if __name__ == '__main__':
    input_dir = "./data/qsar_boolean_large_datasets/"
    output_dir = "./output/qsar_boolean_large_datasets/"
    list_filename = [
        "data_CHEMBL203.csv",
    ]

    sizes = [50]

    for filename in list_filename:
        whole_dataset = pd.read_csv(input_dir + filename)
        whole_dataset = whole_dataset.drop(columns=["molecule_id"])
        for size in sizes:
            train_test = whole_dataset.sample(n=size, random_state=0)
            train_test_splits_dict = kfold_splits(train_test=np.array(train_test), fold=4)

            train_set = train_test_splits_dict[0]["train_set"]
            test_set = train_test_splits_dict[0]["test_set"]

            pa_cnn = pairwise_cnn_model()
            pa_cnn.enter_data(train_set=train_set, test_set=test_set,)
            pa_cnn.run_pairwise_cnn_model()

