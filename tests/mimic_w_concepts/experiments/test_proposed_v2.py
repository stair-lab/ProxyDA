from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys, pickle
from itertools import product
import argparse
import json
import hashlib
import datetime
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from multiprocessing import Pool

from sklearn.preprocessing import StandardScaler
from sklearn import random_projection

sys.path.append(os.path.join('/home/oes2', 'proxy_latent_shifts'))
# REMOVE
from KPLA.data.MIMIC.data_utils import load_data
from KPLA.data.MIMIC.dataset import MIMIC
from KPLA.data.MIMIC.utils.constants import (CXR_EMB_PATH, REPORT_EMB_PATH)

from KPLA.models.plain_kernel.adaptation import FullAdapt
from KPLA.models.plain_kernel.model_selection import tune_adapt_model_cv
# from KPLA.baselines.lsa_kernel import extract_from_df_nested

N_CPUS = os.cpu_count()

# from components.approaches import compute_label_shift_correction_weights

parser = argparse.ArgumentParser()

data_pkl = "age-old_condition_0.7-0.2-0.1_ps-0.1-0.4-0.4-0.1_pt-0.4-0.1-0.1-0.4_ns-5000_nt-5000.pkl"

parser.add_argument('--dicom_split_path', type=str,
  default=os.path.join("/shared/rsaas/oes2/physionet.org/dicom_data_splits/", data_pkl),
)
parser.add_argument('--n_folds', type=int, default=3)
parser.add_argument('--n_params', type=int, default=3)
parser.add_argument('--u_dim', type=int, default=4)
parser.add_argument('--u_val', type=str, default='80+')
parser.add_argument('--u_var', type=str, default='age_old_condition')
parser.add_argument('--x_dim', type=int, default=1376)
parser.add_argument('--y_dim', type=int, default=2)
parser.add_argument('--reduce', type=int, default=64)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--output_dir', type=str, default='/home/kt14/workbench/backup/')

args = parser.parse_args()

np.random.seed(args.seed)

####
desired_length = 10
current_date = datetime.date.today()

day = current_date.day
month = current_date.month

date_string = f"{day:02d}-{month:02d}"

json_str = json.dumps(vars(args), sort_keys=True)
unique_hash = hashlib.sha256(json_str.encode()).hexdigest()
expr_name = ''

path = os.path.join(args.output_dir, expr_name)
os.makedirs(path, exist_ok=True)

############################################################################
RESULTS = {'args': vars(args), 'exp_path': path, 'hparams': {}}
############################################################################
# Load dataset
# Read DICOM splits from file and load CXR metadata
print("metadata")
with open(args.dicom_split_path, "rb") as f:
    dicom_splits = pickle.load(f)["domain_splits"]
metadata = load_data()
metadata.study_id = metadata.study_id.values.astype(str)
if args.u_dim == 2:
    # Perform binary split on U based on DataFrame value
    metadata["U"] = (metadata[args.u_var] == args.u_val).astype(int)
else:
    # Otherwise use column values as label
    metadata["U"] = (metadata[args.u_var]).astype(int)

print("metadata shape:", metadata.shape)
with open(args.dicom_split_path, "rb") as f:
  dicom_splits = pickle.load(f)["domain_splits"]

# Load CXR embeddings
print("loading source")
source_train_D = MIMIC(dicom_splits[0]["train"], metadata,
                             CXR_EMB_PATH, REPORT_EMB_PATH,
                             batch_size=64, U=["U"],
                             verbose=True).generate_data()

source_val_D = MIMIC(dicom_splits[0]["val"], metadata,
                             CXR_EMB_PATH, REPORT_EMB_PATH,
                             batch_size=64, U=["U"],
                             verbose=True).generate_data()

source_train = {
    'X': np.concatenate([source_train_D[0], source_val_D[0]], axis=0),
    'Y': np.concatenate([source_train_D[1], source_val_D[1]], axis=0),
    'C': np.concatenate([source_train_D[2], source_val_D[2]], axis=0),
    'W': np.concatenate([source_train_D[3], source_val_D[3]], axis=0),
    'U': np.concatenate([source_train_D[4], source_val_D[4]], axis=0),
}

# source_X_scaler = StandardScaler().fit(source_train_D[0][source_train_idx, :])
# source_W_scaler = StandardScaler().fit(source_train_D[3][source_train_idx, :])

source_test_D = MIMIC(dicom_splits[0]["test"], metadata,
                             CXR_EMB_PATH, REPORT_EMB_PATH,
                             batch_size=64, U=["U"],
                             verbose=True).generate_data()
source_test = {
    'X': source_test_D[0],
    'Y': source_test_D[1],
    'C': source_test_D[2],
    'W': source_test_D[3],
    'U': source_test_D[4],
}
print("loading target")
target_train_D = MIMIC(dicom_splits[1]["train"], metadata,
                             CXR_EMB_PATH, REPORT_EMB_PATH,
                             batch_size=64, U=["U"],
                             verbose=True).generate_data()
target_val_D = MIMIC(dicom_splits[1]["val"], metadata,
                             CXR_EMB_PATH, REPORT_EMB_PATH,
                             batch_size=64, U=["U"],
                             verbose=True).generate_data()

target_train = {
    'X': np.concatenate([target_train_D[0], target_val_D[0]], axis=0),
    'Y': np.concatenate([target_train_D[1], target_val_D[1]], axis=0),
    'C': np.concatenate([target_train_D[2], target_val_D[2]], axis=0),
    'W': np.concatenate([target_train_D[3], target_val_D[3]], axis=0),
    'U': np.concatenate([target_train_D[4], target_val_D[4]], axis=0),
}

target_test_D = MIMIC(dicom_splits[1]["test"], metadata,
                             CXR_EMB_PATH, REPORT_EMB_PATH,
                             batch_size=64, U=["U"],
                             verbose=True).generate_data()
target_test = {
    'X': target_test_D[0],
    'Y': target_test_D[1],
    'C': target_test_D[2],
    'W': target_test_D[3],
    'U': target_test_D[4],
}

print(source_train['Y'].min(), source_train['Y'].max())
# ## Training
## Dim red
if args.reduce is not None:
  proj_X = random_projection.GaussianRandomProjection(n_components=args.reduce).fit(source_train['X'])
  proj_C = random_projection.GaussianRandomProjection(n_components=args.reduce).fit(source_train['C'])
  print(f"projecting X to {proj_X.n_components_}")
  print(f"projecting C to {proj_C.n_components_}")

  source_train['X'] = proj_X.transform(source_train['X'])
  target_train['X'] = proj_X.transform(target_train['X'])
  source_test['X'] = proj_X.transform(source_test['X'])
  target_test['X'] = proj_X.transform(target_test['X'])

  source_train['C'] = proj_C.transform(source_train['C'])
  target_train['C'] = proj_C.transform(target_train['C'])
  source_test['C'] = proj_C.transform(source_test['C'])
  target_test['C'] = proj_C.transform(target_test['C'])

source_train['Y'] = np.eye(2)[source_train['Y'].flatten()].reshape(-1, 2)[:,1]
source_test['Y'] = np.eye(2)[source_test['Y'].flatten()].reshape(-1, 2)[:,1]
target_train['Y'] = np.eye(2)[target_train['Y'].flatten()].reshape(-1, 2)[:,1]
target_test['Y'] = np.eye(2)[target_test['Y'].flatten()].reshape(-1, 2)[:,1]

print(f"\nsource train X shape: {source_train['X'].shape}")
print(f"source test X shape: {source_test['X'].shape}")
print(f"target train X shape: {target_train['X'].shape}")
print(f"target test X shape: {target_test['X'].shape}\n")

print(f"\nsource train C shape: {source_train['C'].shape}")
print(f"source test C shape: {source_test['C'].shape}")
print(f"target train C shape: {target_train['C'].shape}")
print(f"target test C shape: {target_test['C'].shape}\n")

print(f"\nsource train Y shape: {source_train['Y'].shape}")
print(f"source test Y shape: {source_test['Y'].shape}")
print(f"target train Y shape: {target_train['Y'].shape}")
print(f"target test Y shape: {target_test['Y'].shape}\n")

print(f"\nsource train W shape: {source_train['W'].shape}")
print(f"source test W shape: {source_test['W'].shape}")
print(f"target train W shape: {target_train['W'].shape}")
print(f"target test W shape: {target_test['W'].shape}\n")

print(f"\nsource train U shape: {source_train['U'].shape}")
print(f"source test U shape: {source_test['U'].shape}")
print(f"target train U shape: {target_train['U'].shape}")
print(f"target test U shape: {target_test['U'].shape}\n")

# ### Training Kernel Method
####################
# training...      #
####################

#parameter selection
method_set = {"cme": "original", "h0": "original",}

#specity the kernel functions for each estimator
kernel_dict = {}

N_PARAMS = args.n_params

# # Classification Baselines
kernel_dict["cme_w_xc"] = {"X": "rbf",
                           "C": "rbf",
                           "Y":"rbf"} #Y is W
kernel_dict["cme_wc_x"] = {"X": "rbf",
                           "Y": [{"kernel":"rbf", "dim":source_train['W'].shape[1]},
                                 {"kernel":"rbf",
                                 "dim":source_train['C'].shape[1]}]} # Y is (W,C)
kernel_dict["h0"]       = {"C": "rbf"}

print("Startig kernel adaptation tuning.")
best_estimator, best_params = tune_adapt_model_cv(source_train,
                                                target_train,
                                                source_test,
                                                target_test,
                                                method_set,
                                                kernel_dict,
                                                model=FullAdapt,
                                                task="c",
                                                fit_task = "r",
                                                n_params=args.n_params,
                                                n_fold=args.n_folds,
                                                min_log=-3,
                                                max_log=1,
                                                )

df = best_estimator.evaluation(task="c")
print("best params", best_params)
lam_set = {"cme": best_params["alpha"],
            "h0": best_params["alpha"], 
            "lam_min":-4, 
            "lam_max":-1}
# scale = best_params["scale"]
# split = False

# estimator_full = FullAdapt(source_train,
#                                target_train,
#                                source_test,
#                                target_test,
#                                split,
#                                scale,
#                                lam_set,
#                                method_set,
#                                kernel_dict)

# estimator_full.fit(task="c")
# df = estimator_full.evaluation(task="c")
df.to_csv(os.path.join(path, f'MIMIC_{date_string}_reg.csv'), sep=",",
          index=False,
          encoding="utf-8")

with open(os.path.join(path, f'MIMIC_{date_string}_reg.pkl'), 'wb') as f:
  pickle.dump(RESULTS, f)

with open(os.path.join(path, f'hparams_reg.json'), 'w') as f:
  json.dump(best_params, f, indent=4)