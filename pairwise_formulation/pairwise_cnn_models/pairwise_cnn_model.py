import numpy as np
import pandas as pd
from itertools import product
import logging

from pairwise_formulation.pa_basics.import_data import \
    filter_data, kfold_splits, transform_categorical_columns

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from pairwise_formulation.pairwise_data import PairwiseDataInfo, PairwiseValues
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
        pairwise_diff.append(data[id_i, 0] - data[id_j, 0])

    return np.array(pairwise_features), np.array(pairwise_diff)


class pairwise_cnn_model():
    def __init__(self):
        self.train_set = None
        self.test_set = None
        self.model = None
        self.train_test_pairwise_data_info: PairwiseDataInfo = None
        self.Y_values: PairwiseValues = PairwiseValues()
        self.target_value_col_name: str = "y"

    def enter_training_data(
            self,
            train_set: np.ndarray,
            pc_vld: float,
            target_value_col_name="y"
    ):
        self.train_set = train_set
        self.pc_vld = pc_vld
        self.target_value_col_name = target_value_col_name

    def generate_train_vld_sets(self):
        n_samples = len(self.train_set)
        pc_validation = 0.1
        train_set = shuffle(np.array(self.train_set))

        train_set = train_set[: (1 - int(n_samples * pc_validation))]
        validation_set = train_set[(1 - int(n_samples * pc_validation)):]

        train_vld_pairwise_data_info = PairwiseDataInfo(
            train_set, validation_set,
            target_value_col_name=self.target_value_col_name,
        )

        pairwise_train = pair_2d_pairwise_features(
            data=train_vld_pairwise_data_info.train_test,
            pair_ids=train_vld_pairwise_data_info.c1_test_pair_ids,
        )  # (X_train, Y_train)

        pairwise_validation = pair_2d_pairwise_features(
            data=train_vld_pairwise_data_info.train_test,
            pair_ids=train_vld_pairwise_data_info.c2_test_pair_ids,
        )  # (X_vld, Y_vld)
        return pairwise_train, pairwise_validation

    def run_pairwise_cnn_model(self):
        model = cnn_model()
        loss_fn = nn.MarginRankingLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        n_epochs = 10
        batch_size = 200
        for epoch in range(n_epochs):
            pairwise_train, pairwise_validation = self.generate_train_vld_sets()
            X_train, Y_train = pairwise_train
            n_batches = int(len(Y_train) / batch_size)
            for batch_id in range(n_batches):
                batch_indice = range(batch_id * batch_size, (batch_id + 1) * batch_size)
                # forward, backward, and then weight update
                X = torch.tensor(X_train[batch_indice], dtype=torch.float32)
                Y = torch.tensor(Y_train[batch_indice], dtype=torch.float32)
                Y_pred = model(X)
                loss = loss_fn(
                    Y_pred,
                    torch.zeros((len(Y), 1)),
                    torch.sign(Y.reshape((len(Y), 1)))
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            rank_loss = 0
            X_vld, Y_vld = pairwise_validation
            n_batches = int(len(Y_vld) / batch_size)
            for batch_id in range(n_batches):
                batch_indice = range(batch_id * batch_size, (batch_id + 1) * batch_size)
                X = torch.tensor(X_vld[batch_indice], dtype=torch.float32)
                Y = torch.tensor(Y_vld[batch_indice], dtype=torch.float32)
                Y_pred = model(X)
                rank_loss += accuracy_score(
                    torch.sign(Y).numpy(),
                    torch.sign(Y_pred.reshape(Y_pred.shape[0])).detach().numpy()
                )
            print(f"Epoch {epoch}: model signed accuracy is {rank_loss}")
        self.model = model

    def predict(
            self,
            test_set,
            ranking_method=rating_trueskill,
            ranking_input_type='c2',
            if_sbbr_dist=False,
        ):

        train_test_pairwise_data_info = PairwiseDataInfo(
            self.train_set, test_set, target_value_col_name=self.target_value_col_name
        )
        X_test_c2, Y_test_c2 = pair_2d_pairwise_features(
            data=train_test_pairwise_data_info.train_test,
            pair_ids=train_test_pairwise_data_info.c2_test_pair_ids,
        )

        if (self.train_test_pairwise_data_info is None) or \
           (self.Y_values.Y_pa_c2_sign_true != list(np.sign(Y_test_c2))):

            self.train_test_pairwise_data_info = train_test_pairwise_data_info

            X = torch.tensor(X_test_c2, dtype=torch.float32)
            Y_pred_c2 = self.model(X).reshape(X.shape[0])

            self.Y_values.Y_pa_c2_nume_true = list(Y_test_c2)
            self.Y_values.Y_pa_c2_nume = Y_pred_c2.tolist()
            self.Y_values.Y_pa_c2_sign_true = list(np.sign(Y_test_c2))
            self.Y_values.Y_pa_c2_sign = torch.sign(Y_pred_c2).tolist()

            X_test_c3, Y_test_c3 = pair_2d_pairwise_features(
                data=train_test_pairwise_data_info.train_test,
                pair_ids=train_test_pairwise_data_info.c3_test_pair_ids,
            )
            X = torch.tensor(X_test_c3, dtype=torch.float32)
            Y_pred_c3 = self.model(X).reshape(X.shape[0])

            self.Y_values.Y_pa_c3_nume_true = list(Y_test_c3)
            self.Y_values.Y_pa_c3_nume = Y_pred_c3.tolist()
            self.Y_values.Y_pa_c3_sign_true = list(np.sign(Y_test_c3))
            self.Y_values.Y_pa_c3_sign = torch.sign(Y_pred_c3).tolist()

        if ranking_method is not None:
            logging.info("Returning ranking score for y_test")
            y_ranking_score_test = self.rank(
                ranking_method=ranking_method,
                ranking_input_type=ranking_input_type,
                if_sbbr_dist=if_sbbr_dist
            )
            y = y_ranking_score_test
        else:
            logging.info("Returning estimated y_test")
            y_est = self.estimate_y_from_averaging(
                Y_pa_c2=self.Y_values.Y_pa_c2_nume,
                c2_test_pair_ids=train_test_pairwise_data_info.c2_test_pair_ids,
                test_ids=train_test_pairwise_data_info.test_ids,
                y_true=train_test_pairwise_data_info.y_true_all
            )
            y = y_est

        return y

    def rank(self, ranking_method, ranking_input_type, if_sbbr_dist=False):
        """ranking_inputs: sub-list of ['c2', 'c3', 'c2_c3',]"""
        combi_types = ranking_input_type.split("_")
        Y, test_pair_ids = [], []
        for pair_type in combi_types:

            if not if_sbbr_dist:
                Y += list(getattr(self.Y_values, f"Y_pa_{pair_type}_sign"))
            else:
                Y += list(
                    getattr(self.Y_values, f"Y_pa_{pair_type}_nume")
                )
            test_pair_ids += getattr(self.train_test_pairwise_data_info, f"{pair_type}_test_pair_ids")

        y_ranking_score_all = ranking_method(
            Y=Y,
            test_pair_ids=test_pair_ids,
            y_true=self.train_test_pairwise_data_info.y_true_all)
        y_ranking_score_test = y_ranking_score_all[self.train_test_pairwise_data_info.test_ids]

        setattr(self, f"y_rank_via_{ranking_input_type}", y_ranking_score_test)
        return y_ranking_score_all

    @staticmethod
    def estimate_y_from_averaging(Y_pa_c2, c2_test_pair_ids, test_ids, y_true, Y_weighted=None):
        """
        Estimate activity values from C2-type test pairs via arithmetic mean or weighted average, It is calculated by
        estimating y_test from [Y_(test, train)_pred + y_train_true] and [ - Y_(train, test)_pred + y_train_true]
        """
        if y_true is None:
            y_true = y_true
        if Y_weighted is None:  # linear arithmetic
            Y_weighted = np.ones((len(Y_pa_c2)))

        records = np.zeros((len(y_true)))
        weights = np.zeros((len(y_true)))

        for pair in range(len(Y_pa_c2)):
            ida, idb = c2_test_pair_ids[pair]
            delta_ab = Y_pa_c2[pair]
            weight = Y_weighted[pair]

            if ida in test_ids:
                # (test, train)
                weighted_estimate = (y_true[idb] + delta_ab) * weight
                records[ida] += weighted_estimate
                weights[ida] += weight

            elif idb in test_ids:
                # (train, test)
                weighted_estimate = (y_true[ida] - delta_ab) * weight
                records[idb] += weighted_estimate
                weights[idb] += weight

        return np.divide(records[test_ids], weights[test_ids])


class cnn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 2, kernel_size=32, stride=1, padding=1)
        self.act1 = nn.ReLU()  # return positive response

        self.conv2 = nn.Conv1d(2, 2, kernel_size=16, stride=1, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv1d(2, 2, kernel_size=6, stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=(2))

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(978, 169)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(169, 1)

    def forward(self, x):
        # input 2*1024, output 2*995
        x = self.act1(self.conv1(x))

        # input 2*995, output 2*982
        x = self.act2(self.conv2(x))

        # input 2*982, output 2*979
        x = self.act3(self.conv3(x))

        # input 2*979, output 2*489
        x = self.pool2(x)

        # input 2*489, output 978
        x = self.flat(x)

        # input 1014, output 169
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 10
        # input 169, output 10
        x = self.fc4(x)
        return x

