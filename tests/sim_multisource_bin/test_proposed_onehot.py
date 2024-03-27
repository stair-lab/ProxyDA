"""
Multi-source adaptation with the proposed method for a single target domain.
Using simulated regression task 1: data generation process (D.3).
"""

# Author: Katherine Tsai <kt14@illinois.edu>, Nicole Chiou <nicchiou@stanford.edu>
# MIT License

import argparse
import os
import pandas as pd

from KPLA.data.regression_task_1.gen_data import (
    gen_source_data,
    gen_target_data,
)

from KPLA.models.plain_kernel.multienv_adaptation import MultiEnvAdapt
from KPLA.models.plain_kernel.model_selection import (
    tune_multienv_adapt_model_cv,
)


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=2000)
parser.add_argument("--n_env", type=int, default=2)
parser.add_argument("--s", type=float, default=0.9)
parser.add_argument("--var", type=float, default=1.0)
parser.add_argument("--mean", type=float, default=0.0)
parser.add_argument("--fixs", type=bool, default=False)
parser.add_argument("--fname", type=str, default="test_proporsed_onehot")
parser.add_argument("--outdir", type=str, default="./")
parser.add_argument("--verbose", type=bool, default=False)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
s1 = args.s
s2 = 1.0 - args.s
fname = args.fname


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
# Test set only has 1000 samples
sd_test_list = sd_lst[args.n_env : args.n_env * 2]
source_test = gen_source_data(1000, s1, s2, args.var, args.mean, sd_test_list)

# Generate data from target domain
target_train = gen_target_data(
    args.n_env, args.n * 2, s1, s2, args.var, args.mean, [5949]
)
target_test = gen_target_data(
    args.n_env, 1000, s1, s2, args.var, args.mean, [5654]
)


if args.verbose:
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

    # Check shape
    print("X shape", source_train[0]["X"].shape)
    print("Y shape", source_train[0]["Y"].shape)
    print("W shape", source_train[0]["W"].shape)
    print("U shape", source_train[0]["U"].shape)


####################
# Run adaptation   #
####################

lam_set = {"cme": 1e-4, "m0": 1e-4, "lam_min": -4, "lam_max": -1}
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

split = False
scale = 1


estimator_full = MultiEnvAdapt(
    source_train,
    target_train,
    source_test,
    target_test,
    split,
    scale,
    lam_set,
    method_set,
    kernel_dict,
    verbose=args.verbose,
)
estimator_full.fit(task="r")
estimator_full.evaluation(task="r")

best_estimator, best_parameter = tune_multienv_adapt_model_cv(
    source_train,
    target_train,
    source_test,
    target_test,
    method_set,
    kernel_dict,
    MultiEnvAdapt,
    task="r",
    n_params=10,
    n_fold=5,
    min_log=-4,
    max_log=3,
    fix_scale=args.fixs,
    verbose=args.verbose,
)

best_parameter["source_nsample"] = args.n
best_parameter["n_env"] = args.n_env

df = pd.DataFrame.from_records([best_parameter])
df.to_csv(os.path.join(args.outdir, f"{fname}.csv"), index=False)


# Evaluation of best model
best_estimator.evaluation(task="r")

best_params = best_estimator.get_params()
best_lam_set = best_params["lam_set"]
best_scale = best_params["scale"]
best_method_set = best_params["method_set"]
best_kernel_dict = best_params["kernel_dict"]

if args.verbose:
    print("Evaluation of best model")
    print("best lam:", best_lam_set)
    print("best scale:", best_scale)


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
    verbose=args.verbose,
)
estimator_full.fit(task="r")
estimator_full.evaluation(task="r")
