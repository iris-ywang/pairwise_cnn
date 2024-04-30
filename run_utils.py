import numpy as np

from pairwise_formulation.pairwise_data import PairwiseDataInfo
from pairwise_formulation.evaluations.extrapolation_evaluation import ExtrapolationEvaluation
from pairwise_formulation.pa_basics.rating import rating_trueskill, rating_sbbr

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, ndcg_score
from scipy.stats import spearmanr
from pairwise_formulation.pairwise_cnn_models.pairwise_cnn_model import pairwise_cnn_model


def estimate_y_from_averaging(Y_pa_c2, c2_test_pair_ids, test_ids, y_true, Y_weighted=None):
    """
    Estimate activity values from C2-type test pairs via arithmetic mean or weighted average, It is calculated by
    estimating y_test from [Y_(test, train)_pred + y_train_true] and [ - Y_(train, test)_pred + y_train_true]

    :param Y_pa_c2: np.array of (predicted) differences in activities for C2-type test pairsc
    :param c2_test_pair_ids: list of tuples, each specifying samples IDs for a c2-type pair.
            * Y_pa_c2 and c2_test_pair_ids should match in position; their length should be the same.
    :param test_ids: list of int for test sample IDs
    :param y_true: np.array of true activity values of all samples
    :param Y_weighted: np.array of weighting of each Y_pred (for example, from model prediction probability)
    :return: np.array of estimated activity values for test set
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


def metrics_evaluation(y_true, y_predict):
    rho = spearmanr(y_true, y_predict, nan_policy="omit")[0]
    mse = mean_squared_error(y_true, y_predict)
    mae = mean_absolute_error(y_true, y_predict)
    r2 = r2_score(y_true, y_predict)
    ndcg = ndcg_score([y_true], [y_predict])
    return [rho, mse, mae, r2, np.nan, np.nan]


def build_ml_model(model, train_data, search_model=None, test_data=None):
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]

    if search_model is not None:
        search_model.predict(x_train, y_train)
        model = search_model.best_estimator_

    fitted_model = model.fit(x_train, y_train)

    if type(test_data) == np.ndarray:
        x_test = test_data[:, 1:]
        y_test_pred = fitted_model.predict(x_test)
        return fitted_model, y_test_pred
    else:
        return fitted_model, None

def run_standard_approach(
        foldwise_data: dict,
        ML_reg,
        percentage_of_top_samples=0.1,
        target_value_col_name='y'
):

    train_set = foldwise_data['train_set']
    test_set = foldwise_data['test_set']

    pairwise_data = PairwiseDataInfo(
        train_set, test_set, target_value_col_name=target_value_col_name
    )

    # standard approach
    _, y_sa_pred = build_ml_model(
        model=ML_reg,
        train_data=pairwise_data.train_ary,
        test_data=pairwise_data.test_ary
    )

    y_sa_pred_w_train = np.array(pairwise_data.y_true_all)
    y_sa_pred_w_train[pairwise_data.test_ids] = y_sa_pred

    metrics_sa = ExtrapolationEvaluation(
        percentage_of_top_samples=percentage_of_top_samples,
        y_train_with_predicted_test=y_sa_pred_w_train,
        pairwise_data_info=pairwise_data,
    ).run_extrapolation_evaluation()  # list, 6

    metrics_est_sa = metrics_evaluation(
        pairwise_data.test_ary[:, 0],
        y_sa_pred
    )  # list, 6

    return metrics_sa, metrics_est_sa


def run_pairwise_cnn(
        foldwise_data: dict,
        percentage_of_top_samples=0.1,
):
    train_set = foldwise_data['train_set']
    test_set = foldwise_data['test_set']

    pa_cnn = pairwise_cnn_model()
    pa_cnn.enter_training_data(train_set=train_set, pc_vld=0.1)
    pa_cnn.run_pairwise_cnn_model()

    metrics_rank, metrics_est_pa = results_of_pairwise_combinations(
        pa_cnn, test_set, percentage_of_top_samples=percentage_of_top_samples
    )  # metrics_rank: list of list, 4 * 6; metrics_est_pa: list, 6.

    return metrics_rank, metrics_est_pa


def results_of_pairwise_combinations(
    pairwise_cnn_model,
    test_set: np.ndarray,
    percentage_of_top_samples=0.1,
):
    # Extrapolation performance evaluation:
    # Trueskill
    y_ranking_c2 = pairwise_cnn_model.predict(
        test_set=test_set,
        ranking_method=rating_trueskill,
        ranking_input_type="c2",
        if_sbbr_dist=False,
    )

    metrics_ts_c2 = ExtrapolationEvaluation(
        percentage_of_top_samples=percentage_of_top_samples,
        y_train_with_predicted_test=y_ranking_c2,
        pairwise_data_info=pairwise_cnn_model.train_test_pairwise_data_info,
    ).run_extrapolation_evaluation()

    y_ranking_c2_c3 = pairwise_cnn_model.predict(
        test_set=test_set,
        ranking_method=rating_trueskill,
        ranking_input_type="c2_c3",
        if_sbbr_dist=False,
    )

    metrics_ts_c2_c3 = ExtrapolationEvaluation(
        percentage_of_top_samples=percentage_of_top_samples,
        y_train_with_predicted_test=y_ranking_c2_c3,
        pairwise_data_info=pairwise_cnn_model.train_test_pairwise_data_info,
    ).run_extrapolation_evaluation()


    # SBBR
    y_ranking_c2 = pairwise_cnn_model.predict(
        test_set=test_set,
        ranking_method=rating_sbbr,
        ranking_input_type="c2",
        if_sbbr_dist=True,
    )

    metrics_sbbr_c2 = ExtrapolationEvaluation(
        percentage_of_top_samples=percentage_of_top_samples,
        y_train_with_predicted_test=y_ranking_c2,
        pairwise_data_info=pairwise_cnn_model.train_test_pairwise_data_info,
    ).run_extrapolation_evaluation()  # list of 6 items

    y_ranking_c2_c3 = pairwise_cnn_model.predict(
        test_set=test_set,
        ranking_method=rating_sbbr,
        ranking_input_type="c2_c3",
        if_sbbr_dist=True,
    )

    metrics_sbbr_c2_c3 = ExtrapolationEvaluation(
        percentage_of_top_samples=percentage_of_top_samples,
        y_train_with_predicted_test=y_ranking_c2_c3,
        pairwise_data_info=pairwise_cnn_model.train_test_pairwise_data_info,
    ).run_extrapolation_evaluation()  # list of 6 items


    # Regressive prediction performance evaluation:
    y_pred_est = pairwise_cnn_model.predict(
        test_set=test_set,
        ranking_method=None,
        ranking_input_type="c2",
        if_sbbr_dist=False,
    )

    metrics_est_pa = metrics_evaluation(
        pairwise_cnn_model.train_test_pairwise_data_info.test_ary[:, 0],
        y_pred_est
    )

    return [metrics_ts_c2, metrics_ts_c2_c3, metrics_sbbr_c2, metrics_sbbr_c2_c3], metrics_est_pa