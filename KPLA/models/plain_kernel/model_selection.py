"""
    Cross-validation pipeline for the adaptation methods
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import normalize
import numpy as np
import jax.numpy as jnp
import copy
from itertools import product
from KPLA.models.plain_kernel.adaptation import FullAdapt
from KPLA.models.plain_kernel.multienv_adaptation import MultiEnvAdapt


def tune_adapt_model_cv(source_train:  dict,
                        target_train: dict,
                        source_test:  dict,
                        target_test:  dict,
                        method_set: dict,
                        kernel_dict: dict,
                        model, #function class object
                        task="c",
                        fit_task = "r",
                        n_params=5,
                        n_fold=5,
                        min_log=-4,
                        max_log=4,
                        thre = 0.):

  best_estimator = None
  best_err = np.inf if task=="r" else -np.inf

  params = product(np.logspace(min_log, max_log, n_params).tolist(), #alpha
                    np.logspace(min_log, max_log, n_params).tolist(), #alpha2
                    np.logspace(min_log, max_log, n_params).tolist()) #scale

  best_params = {}

  for param, (alpha, alpha2, scale) in enumerate(params):
    kf = KFold(n_splits=n_fold, random_state=None, shuffle=False)
    lam_set = {"cme": alpha,
               "m0": alpha2,
               "h0": alpha2,
               "lam_min":-4,
               "lam_max":0}
    errs = []
    best_err_i = np.inf if task=="r" else -np.inf
    best_model_i = None

    for i, (train_idx, test_idx) in enumerate(kf.split(source_train["X"])):
      print(f"({alpha}, {alpha2}, {scale}), Param {param} Fold {i}:")

      source_train_cv_train = {k: v[train_idx] for k, v in source_train.items()}
      source_train_cv_val   = {k: v[test_idx]  for k, v in source_train.items()}

      split = False

      estimator = model(source_train_cv_train,
                        target_train,
                        source_test,
                        target_test,
                        split,
                        scale,
                        lam_set,
                        method_set,
                        kernel_dict,
                        thre)

      estimator.fit(fit_task)
      ##select parameters from source
      predict_y = estimator.predict({"X": source_train_cv_val["X"]},
                                      "source", 
                                      "source")
      
      try:
        if task == "r":
          acc_err = mean_squared_error(np.array(source_train_cv_val["Y"]),
                                      np.array(predict_y))

        elif task == "c":
          testy_label = np.array(jnp.argmax(source_train_cv_val["Y"], axis=1))
          predicty_prob = normalize(np.array(predict_y), axis=1)

          acc_err = roc_auc_score(testy_label, predicty_prob[:,1])
      except ValueError as caught_err:
        print(f"Caught {caught_err} on param {param} fold {i}")
        continue

      errs.append(acc_err/len(source_train_cv_val))
      ## select parameters from target
      improve_r = (acc_err < best_err_i) and task == "r"
      improve_c = (acc_err > best_err_i) and task == "c"
      if improve_r or improve_c:
        best_err_i = acc_err
        best_model_i = estimator

    if len(errs) == 0:
      continue
    improve_r = (np.mean(errs) < best_err) and task == "r"
    improve_c = (np.mean(errs) > best_err) and task == "c"

    if improve_r or improve_c:
      best_err = np.mean(errs)
      best_estimator = best_model_i
      best_params = {"alpha":alpha, "alpha2":alpha2, "scale":scale}

      print(f"update best parameters alpha: {alpha}, \
            alpha2:{alpha2}, scale: {scale}, err: {np.mean(errs)}\n")

  return best_estimator, best_params



def tune_adapt_model(source_train: dict,
                     target_train: dict,
                     source_test: dict,
                     target_test: dict,
                     source_val:   dict,
                     method_set:   dict,
                     kernel_dict:   dict,
                     task="r",
                     n_params=5,
                     min_log=-4,
                     max_log=4,
                     ):

  best_estimator = None
  best_err = np.inf if task=="r" else -np.inf

  params = product(np.logspace(min_log, max_log, n_params).tolist(), #alpha
                    np.logspace(min_log, max_log, n_params).tolist()) #scale

  best_params = {}

  for alpha, scale in params:
    lam_set = {"cme": alpha,
               "m0": alpha,
               "h0": alpha,
               "lam_min":-4,
               "lam_max":0}



    print(f"({alpha}, {scale}):")

    split = False

    estimator = FullAdapt(source_train,
                          target_train,
                          source_test,
                          target_test,
                          split,
                          scale,
                          lam_set,
                          method_set,
                          kernel_dict)
    estimator.fit(task)
    ##select parameters from source
    predict_y = estimator.predict({"X": source_val["X"]},
                                  "source", 
                                  "source")
    if task == "r":
      acc_err = mean_squared_error(np.array(source_val["Y"]),
                                   np.array(predict_y))

    elif task == "c":
      testy_label = np.array(jnp.argmax(jnp.abs(source_val["Y"]), axis=1))
      predicty_prob = normalize(np.array(jnp.abs(predict_y)), axis=1)

      acc_err = roc_auc_score(testy_label, predicty_prob[:,1])

    improve_r = (acc_err < best_err) and task == "r"
    improve_c = (acc_err > best_err) and task == "c"

    if improve_r or improve_c:
      best_err = acc_err
      best_estimator = estimator
      best_params = {"alpha":alpha, "scale":scale}

      print(f"update best parameters alpha: {alpha}, \
            scale: {scale}, err: {acc_err}\n")

  return best_estimator, best_params


def tune_multienv_adapt_model(source_train_list: list,
                              target_train_list: list,
                              source_test_list:  list,
                              target_test_list:  list,
                              source_val_list:   list,
                              method_set:        dict,
                              kernel_dict:       dict,
                              models = MultiEnvAdapt,
                              task="r",
                              n_params=5,
                              min_log=-4,
                              max_log=4,
                            ):

  best_estimator = None
  best_err = np.inf if task=="r" else -np.inf

  params = product(np.logspace(min_log, max_log, n_params).tolist(), #alpha
                    np.logspace(min_log, max_log, n_params).tolist()) #scale

  best_params = {}

  for alpha, scale in params:
    lam_set = {"cme": alpha,
               "m0": alpha,
               "h0": alpha,
               "lam_min":-4,
               "lam_max":0}


    print(f"({alpha}, {scale}):")

    split = False

    estimator = models(source_train_list,
                      target_train_list,
                      source_test_list,
                      target_test_list,
                      split,
                      scale,
                      lam_set,
                      method_set,
                      kernel_dict)

    estimator.fit(task)
    ##select parameters from source
    acc_err = 0
    for source_val in source_val_list:
      predict_y = estimator.predict({"X": source_val["X"]},
                                  "source", 
                                  "source")

      if task == "r":
        acc_err += mean_squared_error(np.array(source_val["Y"]),
                                      np.array(predict_y))

      elif task == "c":
        testy_label = np.array(jnp.argmax(jnp.abs(source_val["Y"]), axis=1))
        predicty_prob = normalize(np.array(jnp.abs(predict_y)), axis=1)

        acc_err += roc_auc_score(testy_label, predicty_prob[:,1])

    improve_r = (acc_err < best_err) and task == "r"
    improve_c = (acc_err > best_err) and task == "c"

    if improve_r or improve_c:

      best_err = acc_err
      best_estimator = estimator
      best_params = {"alpha":alpha, "scale":scale}

      print(f"update best parameters alpha: {alpha}, \
            scale: {scale}, err: {acc_err}\n")

  return best_estimator, best_params



def tune_multienv_adapt_model_cv(source_train_list:  dict,
                                target_train_list: dict,
                                source_test_list:  dict,
                                target_test_list:  dict,
                                method_set: dict,
                                kernel_dict: dict,
                                model, #function class object
                                task="c",
                                n_params=5,
                                n_fold=5,
                                min_log=-4,
                                max_log=4,
                                fix_scale=False):

  best_estimator = None
  best_err = np.inf if task=="r" else -np.inf
  if fix_scale:
    params = product(np.logspace(min_log, max_log, n_params).tolist(), #alpha
                    np.logspace(min_log, max_log, n_params).tolist(), #alpha2
                    [1.]) #scale
  else:
    params = product(np.logspace(min_log, max_log, n_params).tolist(), #alpha
                      np.logspace(min_log, max_log, n_params).tolist(), #alpha2
                      np.logspace(min_log, max_log, n_params).tolist()) #scale

  best_params = {}

  #print(f"{len(params)} parameter combinations w/ {n_fold} folds.")

  for param, (alpha, alpha2, scale) in enumerate(params):
    kf = KFold(n_splits=n_fold, random_state=None, shuffle=False)
    lam_set = {"cme": alpha,
               "m0": alpha2,
               "h0": alpha2,
               "lam_min":-4,
               "lam_max":0}
    errs = []
    best_err_i = np.inf if task=="r" else -np.inf
    best_model_i = None

    split_idx = kf.split(source_train_list[0]["X"])
    for i, (train_idx, test_idx) in enumerate(split_idx):
      print(f"({alpha}, {alpha2}, {scale}), Fold {i}:")

      def parse_by_id(par_idx, train_list):
        return [{k: v[par_idx] for k, v in st.items()} for st in train_list]
      source_train_cv_train = parse_by_id(train_idx, source_train_list)
      source_train_cv_val   = parse_by_id(test_idx, source_train_list)

      split = False

      estimator = model(source_train_cv_train,
                        target_train_list,
                        source_test_list,
                        target_test_list,
                        split,
                        scale,
                        lam_set,
                        method_set,
                        kernel_dict)

      estimator.fit(task=task)
      acc_err = 0
      ##select parameters from source
      for idx, source_val in enumerate(source_train_cv_val):
        predict_y = estimator.predict({"X": source_val["X"]},
                                    "source", 
                                    "source", idx)
        #predict_y = estimator.predict({"X": source_val["X"], 
        #                               "Z": source_val["Z"]},
        #                            "source", 
        #                            "source", idx)
        try:
          if task == "r":
            acc_err = mean_squared_error(np.array(source_val["Y"]),
                                        np.array(predict_y))

          elif task == "c":
            testy_label = np.array(jnp.argmax(source_val["Y"], axis=1))
            predicty_prob = normalize(np.array(predict_y), axis=1)

            acc_err = roc_auc_score(testy_label, predicty_prob[:,1])
        except ValueError as caught_err:
          print(f"Caught {caught_err} on param {param} fold {i}")
          continue
      errs.append(acc_err/len(source_train_cv_val))
      ## select parameters from target
      improve_r = (acc_err < best_err_i) and task == "r"
      improve_c = (acc_err > best_err_i) and task == "c"
      if improve_r or improve_c:
        best_err_i = acc_err
        best_model_i = estimator

    improve_r = (np.mean(errs) < best_err) and task == "r"
    improve_c = (np.mean(errs) > best_err) and task == "c"

    if improve_r or improve_c:
      best_err = np.mean(errs)
      best_estimator = best_model_i
      best_params = {"alpha":alpha, "alpha2":alpha2, "scale":scale}

      print(f"update best parameters alpha: {alpha}, alpha2:{alpha2}\
              , scale: {scale}, err: {np.mean(errs)}\n")

  return best_estimator, best_params