"""Implements model selection for proposed method."""

# Author: Katherine Tsai <kt14@illinois.edu>, Nicole Chiou <nicchiou@stanford.edu>
# MIT License

import argparse
import os
import pandas as pd
import jax.numpy as jnp

from KPLA.models.plain_kernel.adaptation import FullAdapt
from KPLA.models.plain_kernel.model_selection import tune_adapt_model_cv
from KPLA.baselines.lsa_kernel import extract_from_df_nested


parser = argparse.ArgumentParser()
parser.add_argument("--source_path", type=str, default="./tmp_data")
parser.add_argument("--w_idx", type=int, default=1)
parser.add_argument("--seed", type=int, default=192)
parser.add_argument("--num_seeds", type=int, default=10)
parser.add_argument("--outdir", type=str, default="./")
parser.add_argument("--verbose", type=bool, default=False)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# Parameter selection
method_set = {
    "cme": "original",
    "h0": "original",
}

# Specify the kernel functions for each estimator
kernel_dict = {}

kernel_dict["cme_w_xc"] = {
    "X": "rbf",
    "C": "binary_column",
    "Y": "rbf",
}  # Y is W
kernel_dict["cme_wc_x"] = {
    "X": "rbf",
    "Y": [{"kernel": "rbf", "dim": 1}, {"kernel": "binary_column", "dim": 3}],
}  # Y is (W,C)
kernel_dict["h0"] = {"C": "binary_column"}


def map_data_to_jax(data_dict):
    train_data = {}
    val_data = {}
    test_data = {}
    keys = data_dict["train"].keys()
    maps = {"x": "X", "w": "W", "c": "C", "u": "U", "y_one_hot": "Y"}
    for key in keys:
        if key in maps.keys():
            train_data[maps[key]] = jnp.array(data_dict["train"][key])
            val_data[maps[key]] = jnp.array(data_dict["val"][key])
            test_data[maps[key]] = jnp.array(data_dict["test"][key])
    return train_data, val_data, test_data


def load_data(s_path, w_id, seed, qu=1, verbose=False):
    source_df = pd.read_csv(
        f"{s_path}/synthetic_multivariate_num_samples_10000_w_coeff_{w_id}_p_u_0_0.9_{seed}.csv"
    )
    source_data_dict = extract_from_df_nested(source_df)
    # Map data to jnp.array
    s_train, s_val, s_test = map_data_to_jax(source_data_dict)

    # Check label
    if verbose:
        print("source train number of Y", s_train["Y"].sum(axis=0))
        print("source test number of Y", s_test["Y"].sum(axis=0))

    # Load target data
    target_df = pd.read_csv(
        f"{s_path}/synthetic_multivariate_num_samples_10000_w_coeff_{w_id}_p_u_0_0.{qu}_{seed}.csv"
    )
    target_data_dict = extract_from_df_nested(target_df)

    t_train, t_val, t_test = map_data_to_jax(target_data_dict)

    # Check label
    if verbose:
        print("target train number of Y", t_train["Y"].sum(axis=0))
        print("target test number of Y", t_test["Y"].sum(axis=0))

    return s_train, s_val, s_test, t_train, t_val, t_test


# Load data
(
    source_train,
    source_val,
    source_test,
    target_train,
    target_val,
    target_test,
) = load_data(args.source_path, args.w_idx, args.seed, verbose=args.verbose)

# Perform model selection
best_estimator, best_params = tune_adapt_model_cv(
    source_train,
    target_train,
    source_test,
    target_test,
    method_set,
    kernel_dict,
    FullAdapt,
    task="c",
    fit_task="c",
    n_params=5,
    min_log=-3,
    max_log=1,
    verbose=args.verbose,
)

best_estimator.evaluation(task="c")
if args.verbose:
    print("best params", best_params)


lam_set = {
    "cme": best_params["alpha"],
    "h0": best_params["alpha"],
    "lam_min": -4,
    "lam_max": -1,
}
scale = best_params["scale"]
split = False

# Start training
for seed in range(args.seed, args.seed + args.num_seeds):
    for qu in range(9, 0, -1):
        (
            source_train,
            source_val,
            source_test,
            target_train,
            target_val,
            target_test,
        ) = load_data(
            args.source_path, args.w_idx, seed, qu, verbose=args.verbose
        )

        estimator_full = FullAdapt(
            source_train,
            target_train,
            source_test,
            target_test,
            split,
            scale,
            lam_set,
            method_set,
            kernel_dict,
        )

        estimator_full.fit(task="c")
        df = estimator_full.evaluation(task="c")

        if args.verbose:
            print("Saving results.")
        df.to_csv(
            f"{args.outdir}/kernel_result_w{args.w_idx}_seed_{seed}_qu{qu}.csv",
            sep=",",
            index=False,
            encoding="utf-8",
        )
