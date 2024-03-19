"""
    Cross-validation pipeline for the adaptation methods
"""

# Author: Katherine Tsai <kt14@illinois.edu>
# License: MIT


import copy
from itertools import product
import numpy as np
import jax.numpy as jnp

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import normalize


def cv_evaluation(valdata_y, predict_y, task, thre):
    """Evaluation using cross-validation.

    Args:
      valdata_y: true Y, jax.numpy.array or numpy.darray
      predict_y:  prediected Y, jax.numpy.array or numpy.darray
      task:       "r" for regression task, "c" for classification task, str
      thre:       threshold, float
    """
    try:
        # try:
        if task == "r":
            mse_err = mean_squared_error(
                np.array(valdata_y), np.array(predict_y)
            )
            return mse_err

        elif task == "c":
            if len(valdata_y.shape) >= 2:
                # classification where Y is one-hot encoded
                testy_label = np.array(jnp.argmax(jnp.abs(valdata_y), axis=1))
                predicty_prob = normalize(np.array(jnp.abs(predict_y)), axis=1)
                predicty_label = np.array(
                    jnp.argmax(jnp.abs(predict_y), axis=1)
                )
                aucroc_err = roc_auc_score(testy_label, predicty_prob[:, -1])
                return aucroc_err
            else:
                # Y is either -1 or 1.
                testy_label = copy.copy(valdata_y)

                # correct -1 to 0
                idx = np.where(testy_label == -1)[0]

                testy_label[idx] = 0.0

                idx1 = np.where(predict_y >= thre)[0]
                predicty_label = np.zeros(predict_y.shape[0], dtype=np.int8)
                predicty_label[idx1] = 1

                acc_err = accuracy_score(testy_label, predicty_label)
                return acc_err
    except ValueError:
        print("Caught error on param selection")
    return 0


def tune_adapt_model_cv(
    source_train: dict,
    target_train: dict,
    source_test: dict,
    target_test: dict,
    method_set: dict,
    kernel_dict: dict,
    model,
    use_validation=False,
    val_data=None,
    task="r",
    fit_task="r",
    n_params=5,
    n_fold=5,
    min_log=-4,
    max_log=4,
    thre=0.0,
):
    """Model selection for adaptation with concepts and proxies.

    Args:
      source_train:   training data of source domain, dict
      target_train:   training data of target domain, dict
      source_test:    testing  data of source domain, dict
      target_test:    testing  data of target domain, dict
      method_set:     methods for optimization, dict
      kernel_dict:    selections of kernel, dict
      model:          estimator, object class
      use_validation: use held-out validation data, Boolean
      val_set:        validation data if use_validation set True, dict
      task:           evaluation method:
                      "c" for classification;
                      "r" for regression, str
      fit_task:       method when fitting model:
                      "c" for classification;
                      "r" for regression, str
      n_params:       number of parameters for searching, int
      n_fold:         number of folds for cross-validation, int
      min_log:        minimum value of the search log-scale, float
      max_log:        maximum value of the search log-scale, float
      thre:           threshold for classification task, float
    """

    best_estimator = None
    best_err = np.inf if task == "r" else -np.inf

    params = product(
        np.logspace(min_log, max_log, n_params).tolist(),  # alpha
        np.logspace(min_log, max_log, n_params).tolist(),  # alpha2
        np.logspace(min_log, max_log, n_params).tolist(),
    )

    best_params = {}

    for param, (alpha, alpha2, scale) in enumerate(params):
        lam_set = {
            "cme": alpha,
            "m0": alpha2,
            "h0": alpha2,
            "lam_min": -4,
            "lam_max": 0,
        }
        print(param, alpha, alpha2, scale)
        errs = []
        best_err_i = np.inf if task == "r" else -np.inf
        best_model_i = None
        split = False
        if use_validation:
            # Use held out validation set
            estimator = model(
                source_train,
                target_train,
                source_test,
                target_test,
                split,
                scale,
                lam_set,
                method_set,
                kernel_dict,
                thre,
            )
            estimator.fit(task=fit_task)
            # select parameters from source
            predict_y = estimator.predict(
                {"X": val_data["X"]}, "source", "source"
            )
            error = cv_evaluation(val_data["Y"], predict_y, task, thre)
            errs.append(error)
            best_model_i = estimator

        else:
            kf = KFold(n_splits=n_fold, random_state=None, shuffle=False)

            # use cross validation method
            for i, (train_idx, test_idx) in enumerate(
                kf.split(source_train["X"])
            ):
                print(f"({alpha}, {alpha2}, {scale}), Param {param} Fold {i}:")

                source_train_cv_train = {
                    k: v[train_idx] for k, v in source_train.items()
                }
                source_train_cv_val = {
                    k: v[test_idx] for k, v in source_train.items()
                }

                estimator = model(
                    source_train_cv_train,
                    target_train,
                    source_test,
                    target_test,
                    split,
                    scale,
                    lam_set,
                    method_set,
                    kernel_dict,
                    thre,
                )

                estimator.fit(task=fit_task)
                ##select parameters from source
                predict_y = estimator.predict(
                    {"X": source_train_cv_val["X"]}, "source", "source"
                )
                error = cv_evaluation(
                    source_train_cv_val["Y"], predict_y, task, thre
                )

                errs.append(error / len(source_train_cv_val))
                ## select parameters from target
                improve_r = (
                    error < best_err_i
                ) and task == "r"  # decrease mse
                improve_c = (
                    error > best_err_i
                ) and task == "c"  # improve accuracy or aucroc
                if improve_r or improve_c:
                    best_err_i = error
                    best_model_i = estimator

        # model selection of the parameter
        if len(errs) == 0:
            continue
        improve_r = (np.mean(errs) < best_err) and task == "r"
        improve_c = (np.mean(errs) > best_err) and task == "c"

        if improve_r or improve_c:
            best_err = np.mean(errs)
            best_estimator = best_model_i
            best_params = {"alpha": alpha, "alpha2": alpha2, "scale": scale}

            print(
                f"update best parameters alpha: {alpha}, \
            alpha2:{alpha2}, scale: {scale}, err: {np.mean(errs)}\n"
            )

    return best_estimator, best_params


def tune_multienv_adapt_model_cv(
    source_train_list: dict,
    target_train_list: dict,
    source_test_list: dict,
    target_test_list: dict,
    method_set: dict,
    kernel_dict: dict,
    model,
    use_validation=False,
    val_list=None,
    task="r",
    fit_task="r",
    n_params=5,
    n_fold=5,
    min_log=-4,
    max_log=4,
    thre=0,
    fix_scale=False,
):
    """Model selection for multi-env adaptation.

    Args:
      source_train_list:   training data of source domain, [dict]
      target_train_list:   training data of target domain, [dict]
      source_test_list:    testing  data of source domain, [dict]
      target_test_list:    testing  data of target domain, [dict]
      method_set:     methods for optimization, dict
      kernel_dict:    selections of kernel, dict
      model:          estimator, object class
      use_validation: use held-out validation data, Boolean
      val_list:       validation data if use_validation set True, dict
      task:           evaluation method:
                      "c" for classification;
                      "r" for regression, str
      fit_task:       method when fitting model:
                      "c" for classification;
                      "r" for regression, str
      n_params:       number of parameters for searching, int
      n_fold:         number of folds for cross-validation, int
      min_log:        minimum value of the search log-scale, float
      max_log:        maximum value of the search log-scale, float
      thre:           threshold for classification task, float
      fix_scale:      not tuning the scaling parameter, Boolean
    """

    best_estimator = None
    best_err = np.inf if task == "r" else -np.inf
    if fix_scale:
        # Do not tune the scale
        params = product(
            np.logspace(min_log, max_log, n_params).tolist(),  # alpha
            np.logspace(min_log, max_log, n_params).tolist(),  # alpha2
            [1.0],
        )
    else:
        params = product(
            np.logspace(min_log, max_log, n_params).tolist(),  # alpha
            np.logspace(min_log, max_log, n_params).tolist(),  # alpha2
            np.logspace(min_log, max_log, n_params).tolist(),
        )  # scale

    best_params = {}

    for _, (alpha, alpha2, scale) in enumerate(params):
        kf = KFold(n_splits=n_fold, random_state=None, shuffle=False)
        lam_set = {
            "cme": alpha,
            "m0": alpha2,
            "h0": alpha2,
            "lam_min": -4,
            "lam_max": 0,
        }
        errs = []
        best_err_i = np.inf if task == "r" else -np.inf
        best_model_i = None
        split = False
        if use_validation:
            estimator = model(
                source_train_cv_train,
                target_train_list,
                source_test_list,
                target_test_list,
                split,
                scale,
                lam_set,
                method_set,
                kernel_dict,
            )
            estimator.fit(task=fit_task)
            error = 0
            for idx, val in enumerate(val_list):
                predict_y = estimator.predict(
                    {"X": val["X"]}, "source", "source", idx
                )
                error += cv_evaluation(val["Y"], predict_y, task, thre)
            errs.append(error / len(val_list))
            best_model_i = estimator

        else:
            split_idx = kf.split(source_train_list[0]["X"])
            for i, (train_idx, test_idx) in enumerate(split_idx):
                print(f"({alpha}, {alpha2}, {scale}), Fold {i}:")

                def parse_by_id(par_idx, train_list):
                    return [
                        {k: v[par_idx] for k, v in st.items()}
                        for st in train_list
                    ]

                source_train_cv_train = parse_by_id(
                    train_idx, source_train_list
                )
                source_train_cv_val = parse_by_id(test_idx, source_train_list)

                estimator = model(
                    source_train_cv_train,
                    target_train_list,
                    source_test_list,
                    target_test_list,
                    split,
                    scale,
                    lam_set,
                    method_set,
                    kernel_dict,
                )

                estimator.fit(task=fit_task)

                # Select parameters from source
                error = 0
                for idx, source_val in enumerate(source_train_cv_val):
                    predict_y = estimator.predict(
                        {"X": source_val["X"]}, "source", "source", idx
                    )
                    error += cv_evaluation(
                        source_val["Y"], predict_y, task, thre
                    )
                errs.append(error / len(source_train_cv_val))

                # Select parameters from target
                improve_r = (error < best_err_i) and task == "r"
                improve_c = (error > best_err_i) and task == "c"
                if improve_r or improve_c:
                    best_err_i = error
                    best_model_i = estimator

        improve_r = (np.mean(errs) < best_err) and task == "r"
        improve_c = (np.mean(errs) > best_err) and task == "c"

        if improve_r or improve_c:
            best_err = np.mean(errs)
            best_estimator = best_model_i
            best_params = {"alpha": alpha, "alpha2": alpha2, "scale": scale}

            print(
                f"update best parameters alpha: {alpha}, alpha2:{alpha2} \
              , scale: {scale}, err: {np.mean(errs)}\n"
            )

    return best_estimator, best_params
