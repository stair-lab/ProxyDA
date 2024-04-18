"""
Implements model selection for the mdehod multi-source Marginal Kernel
Blanchard, G., Lee, G., & Scott, C. (2011). 
Generalizing from several related classification
tasks to a new unlabeled sample. 
Advances in neural information processing systems, 24.
"""

# Author: Katherine Tsai <kt14@illinois.edu>, Nicole Chiou <nicchiou@stanford.edu>
# MIT License


import argparse
import os
import time
from warnings import simplefilter
from multiprocessing import Pool, cpu_count
from itertools import product

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold

from KPLA.baselines.multi_source_mk import MultiSourceMK
from KPLA.data.data_generator import (
    gen_multienv_class_discrete_z,
    convert_to_numpy,
)
from KPLA.models.plain_kernel.method import soft_accuracy, log_loss64


simplefilter(action="ignore", category=FutureWarning)


def load_data(s_path, seed):

    source_train_list_mmd = []
    source_test_list_mmd = []

    for z_env in range(args.n_env):
        source_train = np.load(
            f"{s_path}/source_{z_env}_seed{seed}_train.npy",
            allow_pickle=True,
        ).item()
        source_test = np.load(
            f"{s_path}/source_{z_env}_seed{seed}_test.npy",
            allow_pickle=True,
        ).item()
        source_train_list_mmd.append(convert_to_numpy(source_train.copy()))
        source_test_list_mmd.append(convert_to_numpy(source_test.copy()))

    target_train_list_mmd = []
    target_test_list_mmd = []

    target_train = np.load(
        f"{s_path}/target_seed{seed}_train.npy",
        allow_pickle=True,
    ).item()
    target_test = np.load(
        f"{s_path}/target_seed{seed}_test.npy",
        allow_pickle=True,
    ).item()
    target_train_list_mmd.append(convert_to_numpy(target_train.copy()))
    target_test_list_mmd.append(convert_to_numpy(target_test.copy()))

    return (
        source_train_list_mmd,
        source_test_list_mmd,
        target_train_list_mmd,
        target_test_list_mmd,
    )


parser = argparse.ArgumentParser()
parser.add_argument("--source_n", type=int, default=4000)
parser.add_argument("--target_n", type=int, default=12000)
parser.add_argument("--task", type=int, default=1)
parser.add_argument("--n_env", type=int, default=3)
parser.add_argument("--load_data", type=int, default=1)
parser.add_argument("--source_path", type=str, default="../tmp_data")
parser.add_argument("--outdir", type=str, default="./results/")
parser.add_argument("--seed", type=int, default=192)
parser.add_argument("--verbose", type=int, default=0)
args = parser.parse_args()


outdir = os.path.join(args.outdir, f"task{args.task}")
fname = f"MultisourceMK_{args.seed}.csv"
os.makedirs(outdir, exist_ok=True)

best_params_set = []
result = {}

if args.load_data:
    (
        source_train_list_mmd,
        source_test_list_mmd,
        target_train_list_mmd,
        target_test_list_mmd,
    ) = load_data(
        os.path.join(
            args.source_path, f"task_{args.task}", f"seed{args.seed}"
        ),
        seed=args.seed,
    )

else:
    partition_dict = {"train": 0.8, "test": 0.2}

    # Generate source with 3 environments
    p_u_0 = 0.9
    p_u = [p_u_0, 1 - p_u_0]

    source_train_list_mmd = []
    source_test_list_mmd = []

    for z_env in range(args.n_env):
        source_train, source_test = gen_multienv_class_discrete_z(
            z_env, args.seed + z_env, args.source_n, args.task, partition_dict
        )
        source_train_list_mmd.append(convert_to_numpy(source_train.copy()))
        source_test_list_mmd.append(convert_to_numpy(source_test.copy()))

    target_train_list_mmd = []
    target_test_list_mmd = []

    target_train, target_test = gen_multienv_class_discrete_z(
        args.n_env + 1,
        args.seed + args.n_env + 1,
        args.target_n,
        args.task,
        partition_dict,
    )
    target_train_list_mmd.append(convert_to_numpy(target_train.copy()))
    target_test_list_mmd.append(convert_to_numpy(target_test.copy()))


def run_single_loop(params):
    kf = KFold(n_splits=n_fold, random_state=None, shuffle=False)
    errs = []
    best_err_i = -np.inf
    best_model_i = None
    split_idx = kf.split(source_train_list_mmd[0]["X"])
    for _, (train_idx, test_idx) in enumerate(split_idx):

        def split_data_cv(sidx, t_list):
            list_len = len(t_list)
            return [
                {k: v[sidx] for k, v in t_list[j].items()}
                for j in range(list_len)
            ]

        source_train_cv_train = split_data_cv(train_idx, source_train_list_mmd)
        source_train_cv_test = split_data_cv(test_idx, source_train_list_mmd)

        mssvm = MultiSourceMK(p_kernel=params[0], x_kernel=params[1])

        mssvm.fit(source_train_cv_train, target_train_list_mmd[0])
        # Select parameters from source
        acc_err = 0
        for sd in source_train_cv_test:
            acc_err += roc_auc_score(sd["Y"], mssvm.decision(sd["X"]))
        errs.append(acc_err)
        # Select parameters from target
        if acc_err > best_err_i:
            best_err_i = acc_err
            best_model_i = mssvm
            b_params = {"p_kernel": params[0], "x_kernel": params[1]}

    return best_model_i, np.mean(errs), b_params


n_fold = 5
n_params = 7
min_val = -4
max_val = 2


length_scale = np.logspace(min_val, max_val, n_params)
prams_batch = product(
    [RBF(length_scale=d) for d in length_scale],
    [RBF(length_scale=d) for d in length_scale],
)


pool = Pool(120)
if args.verbose:
    print(f"Start multiprocessing, number of cpu: {cpu_count()-1}")
start_time1 = time.time()
results = pool.map(run_single_loop, prams_batch)
end_time1 = time.time()
if args.verbose:
    print("Total Execution time:", end_time1 - start_time1)
pool.close()

model_batch = [r[0] for r in results]
err_batch = [r[1] for r in results]
params_batch = [r[2] for r in results]
idx = np.argmax(err_batch)
if args.verbose:
    print("Optimal parameters", params_batch[idx])

best_msa = model_batch[idx]
best_params = params_batch[idx]


best_msa.fit(source_train_list_mmd, target_train_list_mmd[0])

predictY = best_msa.predict(np.asarray(target_test_list_mmd[0]["X"]))
predictY_proba = best_msa.decision(np.asarray(target_test_list_mmd[0]["X"]))

err1 = log_loss64(target_test_list_mmd[0]["Y"], predictY_proba)
err2 = soft_accuracy(target_test_list_mmd[0]["Y"], predictY)
err3 = roc_auc_score(target_test_list_mmd[0]["Y"], predictY_proba.flatten())

best_params["approach"] = "MK"
best_params_set.append(best_params)


if args.verbose:
    print(
        "baseline Multisource Marginal Kernel: "
        + f"ll:{err1}, acc:{err2}, aucroc:{err3}"
    )


summary = pd.DataFrame.from_records(best_params_set)
summary.to_csv(os.path.join(outdir, fname), index=False)
