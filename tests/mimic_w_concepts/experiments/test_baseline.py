from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys, pickle
import argparse
import json
import hashlib
import datetime
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from multiprocessing import Pool

from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KernelDensity
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn import random_projection

from sklearn.model_selection import GridSearchCV, KFold

# REMOVE
from KPLA.data.MIMIC.data_utils import load_data

N_CPUS = os.cpu_count()

# from components.approaches import compute_label_shift_correction_weights

parser = argparse.ArgumentParser()

data_pkl = "age-old_condition_0.7-0.2-0.1_ps-0.1-0.4-0.4-0.1_pt-0.4-0.1-0.1-0.4_ns-5000_nt-5000.pkl"
parser.add_argument('--dicom_split_path', type='str',
  default=os.path.join("/shared/rsaas/oes2/physionet.org/dicom_data_splits/", data_pkl),
)

parser.add_argument('--N', type=int, default=10000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--output_dir', type=str, default=os.environ['HOME'])

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
# expr_name = f"{args.alpha_2}_{unique_hash[:desired_length].ljust(desired_length, '0')}"
expr_name = f"{args.alpha_2}"

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
    metadata["U"] = (metadata[args.U_var] == args.U_val).astype(int)
else:
    # Otherwise use column values as label
    metadata["U"] = (metadata[args.U_var]).astype(int)


exit()
# ## Training

# ## Kernel Method
# ### Reduce Dim
## Dim red
DIM = 16

proj = random_projection.GaussianRandomProjection(n_components=DIM).fit(source_train['X'])
print(proj.n_components_)

source_train['X'] = proj.transform(source_train['X'])
print(source_train['X'].shape)
target_train['X'] = proj.transform(target_train['X'])
print(target_train['X'].shape)
source_val['X'] = proj.transform(source_val['X'])
print(source_val['X'].shape)
target_val['X'] = proj.transform(target_val['X'])
print(target_val['X'].shape)
source_test['X'] = proj.transform(source_test['X'])
print(source_test['X'].shape)
target_test['X'] = proj.transform(target_test['X'])
print(target_test['X'].shape)

print(f"\nsource train X shape: {source_train['X'].shape}")
print(f"source val X shape: {source_val['X'].shape}")
print(f"source test X shape: {source_test['X'].shape}")
print(f"target train X shape: {target_train['X'].shape}")
print(f"target val X shape: {target_val['X'].shape}")
print(f"target test X shape: {target_test['X'].shape}\n")

RESULTS['kernel_source_train'] = source_train
RESULTS['kernel_source_val'] = source_val
RESULTS['kernel_source_test'] = source_test

RESULTS['kernel_target_train'] = target_train
RESULTS['kernel_target_val'] = target_val
RESULTS['kernel_target_test'] = target_test

# ### Training Kernel Method
####################
# training...      #
####################

kernel = 'rbf'
alpha = 1e-3

N_FOLD = 3
N_PARAMS = 7

alpha_log_min, alpha_log_max = -4, 2
scale_log_min, scale_log_max = -4, 2

MODEL = KernelRidge

# # Regression Baselines
metrics = []

def get_best(X, y, seed=args.seed, n_params=N_PARAMS, n_fold=N_FOLD, sample_weight=None):
  kr = MODEL(kernel=kernel)

  param_grid = { 
      'gamma': np.logspace(-4, 2, n_params)/DIM,  # Adjust the range as needed
      'alpha': np.logspace(-4, 2, n_params),   # Adjust the range as needed
  }

  scorer = make_scorer(mean_squared_error, greater_is_better=False)

  kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
  grid_search = GridSearchCV(kr, param_grid, cv=kf, scoring=scorer, n_jobs=-1, verbose=4)

  grid_search.fit(X, y, sample_weight=sample_weight)

  print("Best Parameters: ", grid_search.best_params_)

  return grid_search.best_estimator_, grid_search.best_params_


# ## ERM
print('')
print('*'*20+"ERM"+'*'*20)
erm_metrics = {'approach': 'ERM'}

source_erm, source_erm_hparams = get_best(source_train['X'], source_train['Y'])
RESULTS['hparams']['source_erm'] = source_erm_hparams
# source_erm.fit(source_train['X'], source_train['Y'])

erm_metrics['source -> source'] = mean_squared_error(source_test['Y'], source_erm.predict(source_test['X']))
erm_metrics['source -> target'] = mean_squared_error(target_test['Y'], source_erm.predict(target_test['X']))

RESULTS['source_erm'] = source_erm

target_erm, target_erm_hparams = get_best(target_train['X'], target_train['Y'])
RESULTS['hparams']['target_erm'] = target_erm_hparams
# target_erm.fit(target_train['X'], target_train['Y'])

erm_metrics['target -> target'] = mean_squared_error(target_test['Y'], target_erm.predict(target_test['X']))
erm_metrics['target -> source'] = mean_squared_error(source_test['Y'], target_erm.predict(source_test['X']))

RESULTS['target_erm'] = target_erm

metrics.append(erm_metrics)
print(erm_metrics)

# ## Covariate shift
print('')
print('*'*20+"COVAR"+'*'*20)
def convert_data_Y2D(source, target):    
    source_Y = np.zeros_like(source['Y'])
    target_Y = np.ones_like(target['Y'])
    
    return {
        'X': np.concatenate([source['X'], target['X']], axis=0),
        'Y': np.concatenate([source_Y, target_Y], axis=0).ravel(),
    }

covar_metrics = {'approach': 'COVAR'}

# Train domain classifier
domain_D = convert_data_Y2D(source_train, target_train)
d_x = LogisticRegression(random_state=0)
d_x.fit(domain_D['X'], domain_D['Y'])

# Compute sample weights
q_x_train = d_x.predict_proba(source_train['X'])[:, 1]
source_sample_weight_train = q_x_train / (1. - q_x_train + 1e-3)

# RESULTS['d_x'] = d_x

# Compute sample weights
q_x_train = d_x.predict_proba(source_train['X'])[:, 1]
source_sample_weight_train = q_x_train / (1. - q_x_train + 1e-3)

# source_covar_model = MODEL(kernel=kernel, alpha=alpha, gamma=source_gamma)
source_covar_model, source_covar_hparams = get_best(source_train['X'], source_train['Y'], sample_weight=source_sample_weight_train)
RESULTS['hparams']['source_covar'] = source_covar_hparams
# source_covar_model.fit(source_train['X'], source_train['Y'], sample_weight=source_sample_weight_train)

covar_metrics['source -> source'] = mean_squared_error(source_test['Y'], source_covar_model.predict(source_test['X']))
covar_metrics['source -> target'] = mean_squared_error(target_test['Y'], source_covar_model.predict(target_test['X']))

# RESULTS['source_covar_model'] = source_covar_model

# Compute sample weights
q_x_train = d_x.predict_proba(target_train['X'])[:, 0]
target_sample_weight_train = q_x_train / (1. - q_x_train + 1e-3)

# target_covar_model = MODEL(kernel=kernel, alpha=alpha, gamma=target_gamma)
# target_covar_model.fit(target_train['X'], target_train['Y'], sample_weight=target_sample_weight_train)
target_covar_model, target_covar_hparams = get_best(target_train['X'], target_train['Y'], sample_weight=target_sample_weight_train)
RESULTS['hparams']['target_covar'] = target_covar_hparams

covar_metrics['target -> target'] = mean_squared_error(target_test['Y'], target_covar_model.predict(target_test['X']))
covar_metrics['target -> source'] = mean_squared_error(source_test['Y'], target_covar_model.predict(source_test['X']))

# RESULTS['target_covar_model'] = target_covar_model

metrics.append(covar_metrics)
print(covar_metrics)

# ## Label shift
print('')
print('*'*20+"LABEL"+'*'*20)
label_metrics = {'approach': 'LABEL'}

# Fit source and target KDE on oracle Y
source_kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(source_val['Y'])
target_kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(target_train['Y'])

# RESULTS['LABEL_source_kde'] = source_kde
# RESULTS['LABEL_target_kde'] = target_kde

# Compute sample weights q(Y)/p(Y)
log_q_y = target_kde.score_samples(source_train['Y'])
log_p_y = source_kde.score_samples(source_train['Y'])

source_sample_weight_train = np.exp(log_q_y - log_p_y)

# source_label_model = MODEL(kernel=kernel, alpha=alpha, gamma=source_gamma)
# source_label_model.fit(source_train['X'], source_train['Y'], sample_weight=source_sample_weight_train)
source_label_model, source_label_hparams = get_best(source_train['X'], source_train['Y'], sample_weight=source_sample_weight_train)
RESULTS['hparams']['source_label'] = source_label_hparams

label_metrics['source -> source'] = mean_squared_error(source_test['Y'], source_label_model.predict(source_test['X']))
label_metrics['source -> target'] = mean_squared_error(target_test['Y'], source_label_model.predict(target_test['X']))

# RESULTS['source_label_model'] = source_label_model

# Compute sample weights p(Y)/q(Y)
target_sample_weight_train = np.exp(log_p_y - log_q_y)

# target_label_model = MODEL(kernel=kernel, alpha=alpha, gamma=target_gamma)
# target_label_model.fit(target_train['X'], target_train['Y'], sample_weight=target_sample_weight_train)
target_label_model, target_label_hparams = get_best(target_train['X'], target_train['Y'], sample_weight=target_sample_weight_train)
RESULTS['hparams']['target_label'] = target_label_hparams

label_metrics['target -> target'] = mean_squared_error(target_test['Y'], target_label_model.predict(target_test['X']))
label_metrics['target -> source'] = mean_squared_error(source_test['Y'], target_label_model.predict(source_test['X']))

# RESULTS['target_label_model'] = target_label_model

metrics.append(label_metrics)
print(label_metrics)

# ## Summary
summary = pd.DataFrame.from_records(metrics)

RESULTS['summary'] = summary

with open(os.path.join(path, f'dsprites_data_{date_string}.pkl'), 'wb') as f:
  pickle.dump(RESULTS, f)

with open(os.path.join(path, f'hparams.json'), 'w') as f:
  json.dump(RESULTS['hparams'], f, indent=4)

