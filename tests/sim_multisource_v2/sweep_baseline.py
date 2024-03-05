"""
Multi-source adaptation with baselines and sweep over target domains.
Using simulated regression task 1: data generation process (D.3).
"""

# Author: Katherine Tsai <kt14@illinois.edu>, Nicole Chiou <nicchiou@stanford.edu>
# MIT License

import argparse
import copy
import numpy as np
import pandas as pd

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

from KPLA.data.regression_task_1.gen_data import (
    gen_source_data,
    gen_target_data,
)

from KPLA.baselines.model_select import select_kernel_ridge_model


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=2000)
parser.add_argument("--n_env", type=int, default=2)
parser.add_argument("--var", type=float, default=1.0)
parser.add_argument("--mean", type=float, default=0)
parser.add_argument("--fixs", type=bool, default=False)
args = parser.parse_args()


out_fname = "sweep_baseline_v2.csv"

main_summary = pd.DataFrame()
for s1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    s2 = 1.0 - s1

    ####################
    # Generate data    #
    ####################

    seed_list = {}
    sd_lst = [
        5949,
        7422,
        4388,
        2807,
        5654,
        5518,
        1816,
        1102,
        9886,
        1656,
        4379,
        2029,
        8455,
        4987,
        4259,
        2533,
        9783,
        7987,
        1009,
        2297,
    ]

    # Generate data from source domain
    sd_train_list = sd_lst[: args.n_env]
    source_train = gen_source_data(
        args.n, s1, s2, args.var, args.mean, sd_train_list
    )
    source_cat_train = {}
    for k in ["U", "X", "W", "Z", "Y"]:
        temp = np.asarray(copy.deepcopy(source_train[0][k]))
        for data in source_train[1::]:
            temp = np.concatenate((temp, np.asarray(data[k])))
        source_cat_train[k] = temp

    # Test set only has 1000 samples
    sd_test_list = sd_lst[args.n_env : args.n_env * 2]
    source_test = gen_source_data(
        1000, s1, s2, args.var, args.mean, sd_test_list
    )
    source_cat_test = {}
    for k in ["U", "X", "W", "Z", "Y"]:
        temp = np.asarray(copy.deepcopy(source_test[0][k]))
        for data in source_test[1::]:
            temp = np.concatenate((temp, np.asarray(data[k])))
        source_cat_test[k] = temp

    # Generate data from target domain
    target_train = gen_target_data(
        args.n_env, args.n * 2, s1, s2, args.var, args.mean, [5949]
    )
    for k in ["U", "X", "W", "Z", "Y"]:
        target_train[0][k] = np.asarray(target_train[0][k])
    target_test = gen_target_data(
        args.n_env, 1000, s1, s2, args.var, args.mean, [5654]
    )
    for k in ["U", "X", "W", "Z", "Y"]:
        target_test[0][k] = np.asarray(target_test[0][k])

    print("Data generation complete")
    print("Number of source environments:", len(source_train))
    print(
        "Source_train number of samples: ",
        source_train[0]["X"].shape[0] * args.n_env,
    )
    print("Source_test  number of samples: ", source_test[0]["X"].shape[0])
    print("Number of target environments:", len(target_train))
    print("Target_train number of samples: ", target_train[0]["X"].shape[0])
    print("Target_test  number of samples: ", target_test[0]["X"].shape[0])

    RESULTS = {}
    metrics = []
    MODEL = KernelRidge
    kernel = "rbf"
    add_w = False
    # Regress with W
    if add_w:
        print(source_cat_train["X"].shape)
        print(source_cat_train["W"].shape)
        source_cat_train["covar"] = np.stack(
            (source_cat_train["X"].squeeze(), source_cat_train["W"])
        ).T
        source_cat_test["covar"] = np.stack(
            (source_cat_test["X"].squeeze(), source_cat_test["W"])
        ).T
        target_train[0]["covar"] = np.stack(
            (target_train[0]["X"].squeeze(), target_train[0]["W"])
        ).T
        target_test[0]["covar"] = np.stack(
            (target_test[0]["X"].squeeze(), target_test[0]["W"])
        ).T
    else:
        source_cat_train["covar"] = source_cat_train["X"]
        source_cat_test["covar"] = source_cat_test["X"]
        target_train[0]["covar"] = target_train[0]["X"]
        target_test[0]["covar"] = target_test[0]["X"]

    ####################
    # Source ERM       #
    ####################
    kr_model = MODEL(kernel=kernel)
    source_erm, source_erm_hparams = select_kernel_ridge_model(
        kr_model,
        source_cat_train["covar"],
        source_cat_train["Y"],
        n_params=10,
        n_fold=5,
        min_val=-4,
        max_val=3,
    )
    erm_metrics = {"approach": "ERM"}
    erm_metrics["source -> source"] = mean_squared_error(
        source_cat_test["Y"], source_erm.predict(source_cat_test["covar"])
    )
    erm_metrics["source -> target"] = mean_squared_error(
        target_test[0]["Y"], source_erm.predict(target_test[0]["covar"])
    )

    ####################
    # Target ERM       #
    ####################
    kr_model = MODEL(kernel=kernel)
    target_erm, target_erm_hparams = select_kernel_ridge_model(
        kr_model,
        target_train[0]["covar"],
        target_train[0]["Y"],
        n_params=10,
        n_fold=5,
        min_val=-4,
        max_val=3,
    )
    erm_metrics["target -> target"] = mean_squared_error(
        target_test[0]["Y"], target_erm.predict(target_test[0]["covar"])
    )
    erm_metrics["target -> source"] = mean_squared_error(
        source_cat_test["Y"], target_erm.predict(source_cat_test["covar"])
    )

    RESULTS["target_erm"] = target_erm

    metrics.append(erm_metrics)
    print(erm_metrics)
    summary = pd.DataFrame.from_records(metrics)
    summary["pU=0"] = s1
    main_summary = pd.concat([main_summary, summary])


main_summary.to_csv(out_fname, index=False)
