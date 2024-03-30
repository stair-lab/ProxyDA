"""
Implements model selection for WCSC method
Zhang, K., Gong, M., & Schölkopf, B.
Multi-source domain adaptation: A causal view.
In Proceedings of the AAAI Conference on Artificial Intelligence.

"""

# Author: Katherine Tsai <kt14@illinois.edu>, Nicole Chiou <nicchiou@stanford.edu>
# MIT License

import argparse
from itertools import product
from multiprocessing import Pool, cpu_count
import time
import pandas as pd
import numpy as np
import jax.numpy as jnp

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.gaussian_process.kernels import RBF


from KPLA.baselines.multi_source_wcsc import MuiltiSourceCombCLF
from KPLA.data.data_generator import (
    gen_multienv_class_discrete_z,
    convert_to_numpy,
)
from KPLA.models.plain_kernel.method import soft_accuracy, log_loss64


parser = argparse.ArgumentParser()
parser.add_argument("--source_n", type=int, default=4000)
parser.add_argument("--target_n", type=int, default=12000)
parser.add_argument("--task", type=int, default=1)
parser.add_argument("--n_env", type=int, default=3)
parser.add_argument("--seed", type=int, default=192)
parser.add_argument("--verbose", type=bool, default=False)
args = parser.parse_args()

best_params_set = []
partition_dict = {"train": 0.8, "test": 0.2}
result = {}

# Generate source with 3 environments
p_u_0 = 0.9
p_u = [p_u_0, 1 - p_u_0]


source_train_list = []
source_test_list = []


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

        msmmd = MuiltiSourceCombCLF(
            source_data=source_train_cv_train,
            kernel=params[0],
            kde_kernel="gaussian",
            bandwidth=params[1],
        )

        msmmd.fit(target_train_list_mmd[0]["X"])
        ##select parameters from source
        acc_err = 0
        for sd in source_train_cv_test:
            acc_err += roc_auc_score(sd["Y"], msmmd.predict(sd["X"])[:, 1])

        errs.append(acc_err)
        ## select parameters from target
        if acc_err > best_err_i:
            best_err_i = acc_err
            best_model_i = msmmd
            b_params = {"kernel": params[0], "bandwidth": params[1]}

    return best_model_i, np.mean(errs), b_params


n_fold = 5
n_params = 50
min_val = -1
max_val = 4

length_scale = np.logspace(min_val, max_val, n_params)

prams_batch = product(
    [RBF(length_scale=d) for d in length_scale], length_scale
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

best_msmmd = model_batch[idx]
best_params = params_batch[idx]

best_msmmd.fit(np.asarray(target_train_list_mmd[0]["X"]))

predictY_prob = best_msmmd.predict(np.asarray(target_test_list_mmd[0]["X"]))
predictY_label = np.array(jnp.argmax(predictY_prob, axis=1))


err1 = log_loss64(target_test_list_mmd[0]["Y"], predictY_prob[:, 1])
err2 = soft_accuracy(target_test_list_mmd[0]["Y"], predictY_label)
err3 = roc_auc_score(
    target_test_list_mmd[0]["Y"], predictY_prob[:, 1].flatten()
)

best_params["approach"] = "WCSC"
best_params_set.append(best_params)

if args.verbose:
    print(
        "baseline Multisource MMD Classifier: "
        + f"ll:{err1}, acc:{err2}, aucroc:{err3}"
    )


summary = pd.DataFrame.from_records(best_params_set)
summary.to_csv("MultisourceWCSC.csv", index=False)
