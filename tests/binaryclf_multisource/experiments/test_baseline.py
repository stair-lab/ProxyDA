"""Test the baseline models on multi-source LSA classification task."""

# Author: Katherine Tsai <kt14@illinois.edu>, Nicole Chiou <nicchiou@stanford.edu>
# MIT License

import argparse
import os
import pandas as pd
import numpy as np
import jax.numpy as jnp

from sklearn.metrics import roc_auc_score
from sklearn.gaussian_process.kernels import RBF

from KPLA.baselines.multi_source_wcsc import MuiltiSourceCombCLF
from KPLA.baselines.multi_source_mk import MultiSourceMK
from KPLA.baselines.multi_source_ccm import (
    MultiSouceSimpleAdapt,
    MultiSourceUniform,
)
from KPLA.baselines.multi_source_cat import MultiSourceCat
from KPLA.data.data_generator import gen_multienv_class_discrete_z
from KPLA.models.plain_kernel.method import soft_accuracy, log_loss64


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

    return (
        source_train_list,
        source_test_list,
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
parser.add_argument("--seed", type=int, default=192)
parser.add_argument("--num_seeds", type=int, default=10)
parser.add_argument("--file_path", type=str, default="../model_selection/")
parser.add_argument("--ms_seed", type=int, default=None)
parser.add_argument("--outdir", type=str, default="./results/")
parser.add_argument("--verbose", type=int, default=0)
args = parser.parse_args()


outdir = os.path.join(args.outdir, f"task{args.task}")
os.makedirs(outdir, exist_ok=True)

result = {}
metrics = []


for seed in range(args.seed, args.seed + args.num_seeds):

    if args.load_data:
        (
            source_train_list,
            source_test_list,
            source_train_list_mmd,
            source_test_list_mmd,
            target_train_list_mmd,
            target_test_list_mmd,
        ) = load_data(
            os.path.join(args.source_path, f"task_{args.task}", f"seed{seed}"),
            seed=seed,
        )

    else:
        partition_dict = {"train": 0.8, "test": 0.2}

        # Generate source with 3 environments
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

    model_select_seed = seed if args.ms_seed is None else args.ms_seed

    ##################
    # MS Uniform     #
    ##################
    msu = MultiSourceUniform(len(source_train_list_mmd), max_iter=300)
    msu.fit(source_train_list_mmd)
    predictY_prob = msu.predict_proba(target_test_list_mmd[0]["X"])
    predictY = msu.predict(target_test_list_mmd[0]["X"])

    err1 = log_loss64(target_test_list_mmd[0]["Y"], predictY_prob[:, 1])
    err2 = soft_accuracy(target_test_list_mmd[0]["Y"], predictY)
    err3 = roc_auc_score(
        target_test_list_mmd[0]["Y"], predictY_prob[:, 1].flatten()
    )

    if args.verbose:
        print(f"baseline MLP uniform: ll:{err1}, acc:{err2}, aucroc:{err3}")

    m = {"approach": "MLP uniform", "seed": seed, "acc": err2, "aucroc": err3}
    metrics.append(m)

    ##################
    # MS Cat         #
    ##################
    msc = MultiSourceCat(max_iter=300)
    msc.fit(source_train_list_mmd)
    predictY_prob = msc.predict_proba(target_test_list_mmd[0]["X"])
    predictY = msc.predict(target_test_list_mmd[0]["X"])

    err1 = log_loss64(target_test_list_mmd[0]["Y"], predictY_prob[:, 1])
    err2 = soft_accuracy(target_test_list_mmd[0]["Y"], predictY)
    err3 = roc_auc_score(
        target_test_list_mmd[0]["Y"], predictY_prob[:, 1].flatten()
    )

    if args.verbose:
        print(f"baseline MLP concat: ll:{err1}, acc:{err2}, aucroc:{err3}")

    m = {"approach": "MLP cat", "seed": seed, "acc": err2, "aucroc": err3}
    metrics.append(m)

    ###################
    # MS Simple Adapt #
    ###################
    df = pd.read_csv(
        os.path.join(
            args.file_path,
            f"task{args.task}",
            f"MultisourceSA_{model_select_seed}.csv",
        )
    )
    bandwidth = df["bandwidth"].values[0]

    msa = MultiSouceSimpleAdapt(
        n_env=len(source_train_list_mmd),
        kde_kernel="gaussian",
        bandwidth=bandwidth,
        max_iter=300,
    )

    msa.fit(source_train_list_mmd)
    predictY = msa.predict(np.asarray(target_test_list_mmd[0]["X"]))
    predictY_proba = msa.predict_proba(
        np.asarray(target_test_list_mmd[0]["X"])
    )

    err1 = log_loss64(target_test_list_mmd[0]["Y"], predictY_proba[:, 1])
    err2 = soft_accuracy(target_test_list_mmd[0]["Y"], predictY)
    err3 = roc_auc_score(
        target_test_list_mmd[0]["Y"], predictY_proba[:, 1].flatten()
    )

    if args.verbose:
        print(
            "baseline Multisource Simple Adaptation: "
            + f"ll:{err1}, acc:{err2}, aucroc:{err3}"
        )

    m = {
        "approach": "Multisource SA",
        "seed": seed,
        "acc": err2,
        "aucroc": err3,
    }
    metrics.append(m)

    ##################
    # MS WCSC        #
    ##################
    df = pd.read_csv(
        os.path.join(
            args.file_path,
            f"task{args.task}",
            f"MultisourceWCSC_{model_select_seed}.csv",
        )
    )

    bandwidth = df["bandwidth"].values[0]

    rbf_ker = eval(df["kernel"].values[0])
    msmmd = MuiltiSourceCombCLF(source_train_list_mmd, rbf_ker, "gaussian")

    msmmd.fit(np.asarray(target_train_list_mmd[0]["X"]))

    predictY_prob = msmmd.predict(np.asarray(target_test_list_mmd[0]["X"]))
    predictY_label = np.array(jnp.argmax(predictY_prob, axis=1))

    err1 = log_loss64(target_test_list_mmd[0]["Y"], predictY_prob[:, 1])
    err2 = soft_accuracy(target_test_list_mmd[0]["Y"], predictY_label)
    err3 = roc_auc_score(
        target_test_list_mmd[0]["Y"], predictY_prob[:, 1].flatten()
    )

    if args.verbose:
        print(
            "baseline Multisource WCSC classifier: "
            + f"ll:{err1}, acc:{err2}, aucroc:{err3}"
        )

    m = {
        "approach": "Multisource WCSC",
        "seed": seed,
        "acc": err2,
        "aucroc": err3,
    }
    metrics.append(m)

    ##################
    # MS SVM         #
    ##################
    df = pd.read_csv(
        os.path.join(
            args.file_path,
            f"task{args.task}",
            f"MultisourceMK_{model_select_seed}.csv",
        )
    )

    p_ker = eval(df["p_kernel"].values[0])
    x_ker = eval(df["x_kernel"].values[0])
    mssvm = MultiSourceMK(p_ker, x_ker)
    mssvm.fit(source_train_list_mmd, target_train_list_mmd[0])

    predictY_prob = mssvm.decision(np.asarray(target_test_list_mmd[0]["X"]))
    predictY = mssvm.predict(np.asarray(target_test_list_mmd[0]["X"]))

    err1 = log_loss64(target_test_list_mmd[0]["Y"], predictY_prob)
    err2 = soft_accuracy(target_test_list_mmd[0]["Y"], predictY)
    err3 = roc_auc_score(target_test_list_mmd[0]["Y"], predictY_prob)

    if args.verbose:
        print(
            "baseline Multisource mssvm: "
            + f"ll:{err1}, acc:{err2}, aucroc:{err3}"
        )

    m = {
        "approach": "Multisource SVM",
        "seed": seed,
        "acc": err2,
        "aucroc": err3,
    }
    metrics.append(m)

df = pd.DataFrame.from_records(metrics)
if args.verbose:
    print(df)

df.to_csv(
    os.path.join(outdir, "multienv_classification_baseline.csv"),
    index=False,
)
