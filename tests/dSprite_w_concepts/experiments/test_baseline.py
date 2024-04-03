"""Implements model selection for baseline methods."""

import os
import pickle
import argparse
import json
import hashlib
import random
import datetime
import numpy as np
import pandas as pd


from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KernelDensity
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn import random_projection
from sklearn.model_selection import GridSearchCV, KFold

from KPLA.baselines.model_select import select_kernel_ridge_model
from KPLA.data.dSprite.data_generator import generate_data

N_CPUS = os.cpu_count()


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
parser.add_argument("--alpha", type=float)
parser.add_argument("--scale", type=float)

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
kr_model = MODEL(kernel=kernel)

# Regression Baselines
metrics = []


def get_best(
    X, y, seed=args.seed, n_params=N_PARAMS, n_fold=N_FOLD, sample_weight=None
):
    kr = MODEL(kernel=kernel)

    param_grid = {
        "gamma": np.logspace(-4, 2, n_params)
        / DIM,  # Adjust the range as needed
        "alpha": np.logspace(-4, 2, n_params),  # Adjust the range as needed
    }

    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    grid_search = GridSearchCV(
        kr, param_grid, cv=kf, scoring=scorer, n_jobs=-1, verbose=4
    )

    grid_search.fit(X, y, sample_weight=sample_weight)

    print(f"Best Parameters: {grid_search.best_params_}")

    return grid_search.best_estimator_, grid_search.best_params_


# ERM
print("*" * 20 + "ERM" + "*" * 20)
erm_metrics = {"approach": "ERM"}

kr_model = MODEL(kernel=kernel)
source_erm, source_erm_hparams = select_kernel_ridge_model(
    kr_model, source_train["X"], source_train["Y"]
)
RESULTS["hparams"]["source_erm"] = source_erm_hparams

erm_metrics["source -> source"] = mean_squared_error(
    source_test["Y"], source_erm.predict(source_test["X"])
)
erm_metrics["source -> target"] = mean_squared_error(
    target_test["Y"], source_erm.predict(target_test["X"])
)

RESULTS["source_erm"] = source_erm

kr_model = MODEL(kernel=kernel)
target_erm, target_erm_hparams = select_kernel_ridge_model(
    kr_model, target_train["X"], target_train["Y"]
)
RESULTS["hparams"]["target_erm"] = target_erm_hparams

erm_metrics["target -> target"] = mean_squared_error(
    target_test["Y"], target_erm.predict(target_test["X"])
)
erm_metrics["target -> source"] = mean_squared_error(
    source_test["Y"], target_erm.predict(source_test["X"])
)

RESULTS["target_erm"] = target_erm

metrics.append(erm_metrics)
print(erm_metrics)

# Covariate shift
print("")
print("*" * 20 + "COVAR" + "*" * 20)


def convert_data_Y2D(source, target):
    source_Y = np.zeros_like(source["Y"])
    target_Y = np.ones_like(target["Y"])

    return {
        "X": np.concatenate([source["X"], target["X"]], axis=0),
        "Y": np.concatenate([source_Y, target_Y], axis=0).ravel(),
    }


covar_metrics = {"approach": "COVAR"}

# Train domain classifier
domain_D = convert_data_Y2D(source_train, target_train)
d_x = LogisticRegression(random_state=0)
d_x.fit(domain_D["X"], domain_D["Y"])

# Compute sample weights
q_x_train = d_x.predict_proba(source_train["X"])[:, 1]
source_sample_weight_train = q_x_train / (1.0 - q_x_train + 1e-3)

# Compute sample weights
q_x_train = d_x.predict_proba(source_train["X"])[:, 1]
source_sample_weight_train = q_x_train / (1.0 - q_x_train + 1e-3)

kr_model = MODEL(kernel=kernel)
source_covar_model, source_covar_hparams = select_kernel_ridge_model(
    kr_model, source_train["X"], source_train["Y"], source_sample_weight_train
)


RESULTS["hparams"]["source_covar"] = source_covar_hparams

covar_metrics["source -> source"] = mean_squared_error(
    source_test["Y"], source_covar_model.predict(source_test["X"])
)
covar_metrics["source -> target"] = mean_squared_error(
    target_test["Y"], source_covar_model.predict(target_test["X"])
)

# Compute sample weights
q_x_train = d_x.predict_proba(target_train["X"])[:, 0]
target_sample_weight_train = q_x_train / (1.0 - q_x_train + 1e-3)

kr_model = MODEL(kernel=kernel)
target_covar_model, target_covar_hparams = select_kernel_ridge_model(
    kr_model, target_train["X"], target_train["Y"], target_sample_weight_train
)


RESULTS["hparams"]["target_covar"] = target_covar_hparams

covar_metrics["target -> target"] = mean_squared_error(
    target_test["Y"], target_covar_model.predict(target_test["X"])
)
covar_metrics["target -> source"] = mean_squared_error(
    source_test["Y"], target_covar_model.predict(source_test["X"])
)

metrics.append(covar_metrics)
print(covar_metrics)

# Label shift
print("")
print("*" * 20 + "LABEL" + "*" * 20)
label_metrics = {"approach": "LABEL"}

# Fit source and target KDE on oracle Y
source_kde = KernelDensity(kernel="gaussian", bandwidth=1).fit(source_val["Y"])
target_kde = KernelDensity(kernel="gaussian", bandwidth=1).fit(
    target_train["Y"]
)

# Compute sample weights q(Y)/p(Y)
log_q_y = target_kde.score_samples(source_train["Y"])
log_p_y = source_kde.score_samples(source_train["Y"])

source_sample_weight_train = np.exp(log_q_y - log_p_y)

kr_model = MODEL(kernel=kernel)
source_label_model, source_label_hparams = select_kernel_ridge_model(
    kr_model,
    source_train["X"],
    source_train["Y"],
    sample_weight=source_sample_weight_train,
)
RESULTS["hparams"]["source_label"] = source_label_hparams

label_metrics["source -> source"] = mean_squared_error(
    source_test["Y"], source_label_model.predict(source_test["X"])
)
label_metrics["source -> target"] = mean_squared_error(
    target_test["Y"], source_label_model.predict(target_test["X"])
)


# Compute sample weights p(Y)/q(Y)
target_sample_weight_train = np.exp(log_p_y - log_q_y)

kr_model = MODEL(kernel=kernel)
target_label_model, target_label_hparams = select_kernel_ridge_model(
    kr_model,
    target_train["X"],
    target_train["Y"],
    sample_weight=target_sample_weight_train,
)
RESULTS["hparams"]["target_label"] = target_label_hparams

label_metrics["target -> target"] = mean_squared_error(
    target_test["Y"], target_label_model.predict(target_test["X"])
)
label_metrics["target -> source"] = mean_squared_error(
    source_test["Y"], target_label_model.predict(source_test["X"])
)


metrics.append(label_metrics)
print(label_metrics)

# Summary
summary = pd.DataFrame.from_records(metrics)

RESULTS["summary"] = summary

with open(
    os.path.join(path, f"dsprites_baseline_{date_string}.pkl"), "wb"
) as f:
    pickle.dump(RESULTS, f)

with open(os.path.join(path, f"hparams_baseline.json"), "w") as f:
    json.dump(RESULTS["hparams"], f, indent=4)
