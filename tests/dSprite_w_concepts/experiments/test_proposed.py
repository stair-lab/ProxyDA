"""Implements model selection for proposed method."""

import os
import pickle
import argparse
import json
import hashlib
import random
import datetime
from copy import deepcopy
import numpy as np
import pandas as pd


from sklearn.kernel_ridge import KernelRidge
from sklearn import random_projection


from KPLA.models.plain_kernel.adaptation import FullAdapt
from KPLA.models.plain_kernel.model_selection import tune_adapt_model_cv
from KPLA.data.dSprite.data_generator import generate_data


parser = argparse.ArgumentParser()
parser.add_argument("--alpha_1", type=float, default=2)
parser.add_argument("--beta_1", type=float, default=4)
parser.add_argument(
    "--dist_1", type=str, default="beta", choices=["beta", "uniform"]
)
parser.add_argument("--alpha_2", type=float, default=0.0)
parser.add_argument("--beta_2", type=float, default=1.0)
parser.add_argument(
    "--dist_2", type=str, default="uniform", choices=["beta", "uniform"]
)

parser.add_argument("--N", type=int, default=10000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dir", type=str, default="./")
parser.add_argument("--verbose", type=bool, default=False)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)


desired_length = 10
current_date = datetime.date.today()

day = current_date.day
month = current_date.month

date_string = f"{day:02d}-{month:02d}"

json_str = json.dumps(vars(args), sort_keys=True)
unique_hash = hashlib.sha256(json_str.encode()).hexdigest()
expr_name = f"{args.alpha_2}"

path = os.path.join(args.output_dir, expr_name)
os.makedirs(path, exist_ok=True)

############################################################################
RESULTS = {"args": vars(args), "exp_path": path, "hparams": {}}
############################################################################
# Load dataset
data_path = "../../../KPLA/data/dSprite/"
data = generate_data(
    data_path,
    args.N,
    args.alpha_1,
    args.beta_1,
    args.dist_1,
    args.alpha_2,
    args.beta_2,
    args.dist_2,
    verbose=args.verbose,
)
source_train = data["source_train"]
source_val = data["source_val"]
source_test = data["source_test"]
source_imgs_dict = data["source_imgs_dict"]
target_train = data["target_train"]
target_val = data["target_val"]
target_test = data["target_test"]
target_imgs_dict = data["target_imgs_dict"]
RESULTS.update(data)

DIM = 16

proj = random_projection.GaussianRandomProjection(n_components=DIM).fit(
    source_train["X"]
)

source_train["X"] = proj.transform(source_train["X"])
target_train["X"] = proj.transform(target_train["X"])
source_val["X"] = proj.transform(source_val["X"])
target_val["X"] = proj.transform(target_val["X"])
source_test["X"] = proj.transform(source_test["X"])
target_test["X"] = proj.transform(target_test["X"])


if args.verbose:
    print(f"\nsource train X shape: {source_train['X'].shape}")
    print(f"source val X shape: {source_val['X'].shape}")
    print(f"source test X shape: {source_test['X'].shape}")
    print(f"target train X shape: {target_train['X'].shape}")
    print(f"target val X shape: {target_val['X'].shape}")
    print(f"target test X shape: {target_test['X'].shape}\n")

RESULTS["kernel_source_train"] = source_train
RESULTS["kernel_source_val"] = source_val
RESULTS["kernel_source_test"] = source_test

RESULTS["kernel_target_train"] = target_train
RESULTS["kernel_target_val"] = target_val
RESULTS["kernel_target_test"] = target_test

####################
# Training         #
####################

kernel = "rbf"
alpha = 1e-3

N_FOLD = 3
N_PARAMS = 7

alpha_log_min, alpha_log_max = -4, 2
scale_log_min, scale_log_max = -4, 2

MODEL = KernelRidge

method_set = {"cme": "original", "h0": "original", "m0": "original"}
X_kernel = "rbf"
kernel_dict = {}
kernel_dict["cme_w_xc"] = {"X": X_kernel, "C": "rbf", "Y": "rbf"}  # Y is W
kernel_dict["cme_wc_x"] = {"X": X_kernel, "Y": "rbf"}  # Y is (W,C)

kernel_dict["h0"] = {"C": "rbf"}
kernel_dict["m0"] = {"C": "rbf", "X": X_kernel}

estimator_full, best_params = tune_adapt_model_cv(
    source_train,
    target_train,
    source_test,
    target_test,
    method_set,
    kernel_dict,
    FullAdapt,
    task="r",
    n_params=N_PARAMS,
    n_fold=N_FOLD,
    min_log=-4,
    max_log=0,
)
RESULTS["kernel_dict"] = kernel_dict
RESULTS["hparams"]["kernel"] = {
    "alpha": best_params["alpha"],
    "alpha2": best_params["alpha2"],
    "scale": best_params["scale"],
}

print("")
print("*" * 20 + "LSA" + "*" * 20)
res_full = estimator_full.evaluation()

RESULTS["estimator_full"] = estimator_full

# Summary
full_adapt_metrics = {
    k.replace("-", " -> "): v
    for k, v in zip(res_full.task.values, res_full["predict error.l2"].values)
}
full_adapt_metrics["approach"] = "PROXY (W, C) (baseline)"

full_adapt_metrics = deepcopy(full_adapt_metrics)
full_adapt_metrics["approach"] = "PROXY (W, C) "
full_adapt_metrics["source -> target"] = full_adapt_metrics["adaptation"]

del full_adapt_metrics["adaptation"]

print(full_adapt_metrics)

proxy_metrics = [full_adapt_metrics]

# Summary
summary = pd.DataFrame.from_records(proxy_metrics)

RESULTS["summary"] = summary

with open(
    os.path.join(path, f"dsprites_proposed_{date_string}.pkl"), "wb"
) as f:
    pickle.dump(RESULTS, f)

with open(os.path.join(path, f"hparams_proposed.json"), "w") as f:
    json.dump(RESULTS["hparams"], f, indent=4)
