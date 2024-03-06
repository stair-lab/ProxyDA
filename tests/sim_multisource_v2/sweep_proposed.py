"""
Multi-source adaptation with the proposed method and sweep over target domains.
Using simulated regression task 1: data generation process (D.3).
"""

# Author: Katherine Tsai <kt14@illinois.edu>, Nicole Chiou <nicchiou@stanford.edu>
# MIT License

import argparse
import pandas as pd

from KPLA.data.regression_task_1.gen_data import (
    gen_source_data,
    gen_target_data,
)

from KPLA.models.plain_kernel.multienv_adaptation import MultiEnvAdapt


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=2000)
parser.add_argument("--n_env", type=int, default=2)
parser.add_argument("--var", type=float, default=1.0)
parser.add_argument("--mean", type=float, default=0)
parser.add_argument("--fixs", type=bool, default=False)
args = parser.parse_args()


out_fname = "sweep_proposed_v2"
if args.fixs <= 1:
    out_fname += "_fixscale"
out_fname += ".csv"

file_path = "./model_select/"
main_df = pd.DataFrame()
for s1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    fname = f"test_proposed_v2_onehot_{s1}_m_{args.mean}_var_{args.var}"
    if args.fixs <= 1:
        fname += "_fixscale"
    fname += ".csv"
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

    mean_z = [-0.5, 0.0, 0.5, 1.0]
    sigma_z = [1.0, 1.0, 1.0, 1.0]

    prob_list = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    # Generate data from source domain
    sd_train_list = sd_lst[: args.n_env]
    source_train = gen_source_data(
        args.n, s1, s2, args.var, args.mean, sd_train_list
    )
    # Test set only has 1000 samples
    sd_test_list = sd_lst[args.n_env : args.n_env * 2]
    source_test = gen_source_data(
        1000, s1, s2, args.var, args.mean, sd_test_list
    )

    # Generate data from target domain
    target_train = gen_target_data(
        args.n_env, args.n * 2, s1, s2, args.var, args.mean, [5949]
    )
    target_test = gen_target_data(
        args.n_env, 1000, s1, s2, args.var, args.mean, [5654]
    )

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

    ####################
    # Run adaptation   #
    ####################

    lam_set = {"cme": 1e-4, "m0": 1e-5, "lam_min": -4, "lam_max": -1}
    method_set = {"cme": "original", "m0": "original"}

    # Specity the kernel functions for each estimator
    kernel_dict = {}

    X_kernel = "rbf"
    W_kernel = "rbf"
    kernel_dict["cme_w_xz"] = {
        "X": X_kernel,
        "Y": W_kernel,
        "Z": "binary",
    }  # Y is W
    kernel_dict["cme_w_x"] = {"X": X_kernel, "Y": W_kernel}  # Y is W
    kernel_dict["m0"] = {"X": X_kernel}

    df = pd.read_csv(file_path + fname)

    best_lam_set = {
        "cme": df["alpha"].values[0],
        "m0": df["alpha2"].values[0],
        "lam_min": -4,
        "lam_max": -1,
    }
    best_scale = df["scale"].values[0]

    print("best lam:", best_lam_set)
    print("best scale:", best_scale)

    split = False
    scale = 1

    estimator_full = MultiEnvAdapt(
        source_train,
        target_train,
        source_test,
        target_test,
        split,
        best_scale,
        best_lam_set,
        method_set,
        kernel_dict,
    )
    estimator_full.fit(task="r")

    df = estimator_full.evaluation(task="r")
    df["pU=0"] = s1

    main_df = pd.concat([main_df, df])

main_df.to_csv(out_fname, index=False)
