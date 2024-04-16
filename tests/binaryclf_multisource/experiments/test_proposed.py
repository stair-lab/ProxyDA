"""Test the proposed method on multi-source LSA classification dataset."""

# Author: Katherine Tsai <kt14@illinois.edu>, Nicole Chiou <nicchiou@stanford.edu>
# MIT License

import argparse
import os
import numpy as np
import pandas as pd

from KPLA.models.plain_kernel.multienv_adaptation import MultiEnvAdaptCAT
from KPLA.data.data_generator import gen_multienv_class_discrete_z


def load_data(s_path, seed):

    source_train_list = []
    source_test_list = []

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
        source_train_list_mmd.append(source_train.copy())
        source_test_list_mmd.append(source_test.copy())

        source_train["Y"] = source_train["Y_one_hot"]
        source_test["Y"] = source_test["Y_one_hot"]

        source_train_list.append(source_train)
        source_test_list.append(source_test)

    target_train_list = []
    target_test_list = []

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
    target_train_list_mmd.append(target_train.copy())
    target_test_list_mmd.append(target_test.copy())

    target_train["Y"] = target_train["Y_one_hot"]
    target_test["Y"] = target_test["Y_one_hot"]

    target_train_list.append(target_train)
    target_test_list.append(target_test)

    return (
        source_train_list,
        source_test_list,
        source_train_list_mmd,
        source_test_list_mmd,
        target_train_list,
        target_test_list,
        target_train_list_mmd,
        target_test_list_mmd,
    )


parser = argparse.ArgumentParser()
parser.add_argument("--source_n", type=int, default=4000)
parser.add_argument("--target_n", type=int, default=12000)
parser.add_argument("--task", type=int, default=1)
parser.add_argument("--n_env", type=int, default=3)
parser.add_argument("--load_data", type=bool, default=True)
parser.add_argument("--source_path", type=str, default="../tmp_data")
parser.add_argument("--seed", type=int, default=192)
parser.add_argument("--num_seeds", type=int, default=10)
parser.add_argument("--file_path", type=str, default="../model_selection/")
parser.add_argument("--ms_seed", type=int, default=200)
parser.add_argument("--outdir", type=str, default="./results/")
parser.add_argument("--verbose", type=bool, default=False)
args = parser.parse_args()


outdir = os.path.join(args.outdir, f"task{args.task}")
os.makedirs(outdir, exist_ok=True)

summary = pd.DataFrame()
result = {}


lam_set = {"cme": 1e-3, "m0": 1e-3, "lam_min": -4, "lam_max": -1}
method_set = {"cme": "original", "m0": "original"}

# Specify the kernel functions for each estimator
kernel_dict = {}

kernel_dict["cme_w_xz"] = {"X": "rbf", "Y": "rbf_column"}  # Y is W
kernel_dict["cme_w_x"] = {"X": "rbf", "Y": "rbf_column"}  # Y is W
kernel_dict["m0"] = {"X": "rbf"}


for seed in range(args.seed, args.seed + args.num_seeds):

    df = pd.read_csv(
        args.file_path + f"classification_model_select_{args.ms_seed}.csv"
    )

    best_lam_set = {
        "cme": df["alpha"].values[0],
        "m0": df["alpha2"].values[0],
        "lam_min": -4,
        "lam_max": -1,
    }
    best_scale = df["scale"].values[0]
    split = False

    if args.verbose:
        print(f"best lam: {best_lam_set}")
        print(f"best scale: {best_scale}")

    if args.load_data:
        (
            source_train_list,
            source_test_list,
            source_train_list_mmd,
            source_test_list_mmd,
            target_train_list,
            target_test_list,
            target_train_list_mmd,
            target_test_list_mmd,
        ) = load_data(
            os.path.join(args.source_path, f"task_{args.task}", f"seed{seed}"),
            seed=seed,
        )

    else:
        partition_dict = {"train": 0.8, "test": 0.2}

        # Generate source with 3 environments
        p_u_0 = 0.9
        p_u = [p_u_0, 1 - p_u_0]

        source_train_list = []
        source_test_list = []

        source_train_list_mmd = []
        source_test_list_mmd = []

        for z_env in range(args.n_env):
            source_train, source_test = gen_multienv_class_discrete_z(
                z_env, seed + z_env, args.source_n, args.task, partition_dict
            )
            source_train_list_mmd.append(source_train.copy())
            source_test_list_mmd.append(source_test.copy())

            source_train["Y"] = source_train["Y_one_hot"]
            source_test["Y"] = source_test["Y_one_hot"]

            source_train_list.append(source_train)
            source_test_list.append(source_test)

        # Generate target
        target_train_list = []
        target_test_list = []

        target_train_list_mmd = []
        target_test_list_mmd = []

        target_train, target_test = gen_multienv_class_discrete_z(
            args.n_env + 1,
            seed + args.n_env + 1,
            args.target_n,
            args.task,
            partition_dict,
        )

        target_train_list_mmd.append(target_train.copy())
        target_test_list_mmd.append(target_test.copy())

        target_train["Y"] = target_train["Y_one_hot"]
        target_test["Y"] = target_test["Y_one_hot"]

        target_train_list.append(target_train)
        target_test_list.append(target_test)

    estimator_full = MultiEnvAdaptCAT(
        source_train_list,
        target_train_list,
        source_test_list,
        target_test_list,
        split,
        best_scale,
        best_lam_set,
        method_set,
        kernel_dict,
    )
    estimator_full.fit(task="c")

    df = estimator_full.evaluation(task="c")
    df["seed"] = seed

    summary = pd.concat([summary, df])

if args.verbose:
    print(summary)

summary.to_csv(
    os.path.join(outdir, "multienv_classification.csv"), index=False
)
