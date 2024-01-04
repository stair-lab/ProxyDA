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

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
from sklearn import random_projection

from sklearn.model_selection import GridSearchCV, KFold

sys.path.append(os.path.join('/home/oes2', 'proxy_latent_shifts'))
# REMOVE
from KPLA.data.MIMIC.data_utils import load_data
from KPLA.data.MIMIC.dataset import MIMIC
from KPLA.data.MIMIC.utils.constants import (CXR_EMB_PATH, REPORT_EMB_PATH)

N_CPUS = os.cpu_count()

# from components.approaches import compute_label_shift_correction_weights

parser = argparse.ArgumentParser()

data_pkl = "age-old_condition_0.7-0.2-0.1_ps-0.1-0.4-0.4-0.1_pt-0.4-0.1-0.1-0.4_ns-5000_nt-5000.pkl"

parser.add_argument('--dicom_split_path', type=str,
  default=os.path.join("/shared/rsaas/oes2/physionet.org/dicom_data_splits/", data_pkl),
)
parser.add_argument('--n_folds', type=int, default=3)
parser.add_argument('--n_params', type=int, default=7)
parser.add_argument('--u_dim', type=int, default=4)
parser.add_argument('--u_val', type=str, default='80+')
parser.add_argument('--u_var', type=str, default='age_old_condition')
parser.add_argument('--x_dim', type=int, default=1376)
parser.add_argument('--y_dim', type=int, default=2)
parser.add_argument('--reduce', type=int)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--output_dir', type=str, default='/home/oes2')

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
source_train_ = {
    'X': source_train_D[0],
    'Y': source_train_D[1],
    'C': source_train_D[2],
    'W': source_train_D[3],
    'U': source_train_D[4],
}
source_val_D = MIMIC(dicom_splits[0]["val"], metadata,
                             CXR_EMB_PATH, REPORT_EMB_PATH,
                             batch_size=64, U=["U"],
                             verbose=True).generate_data()
source_val = {
    'X': source_val_D[0],
    'Y': source_val_D[1],
    'C': source_val_D[2],
    'W': source_val_D[3],
    'U': source_val_D[4],
}
source_train = {
    'X': np.concatenate([source_train_D[0], source_val_D[0]], axis=0),
    'Y': np.concatenate([source_train_D[1], source_val_D[1]], axis=0),
    'C': np.concatenate([source_train_D[2], source_val_D[2]], axis=0),
    'W': np.concatenate([source_train_D[3], source_val_D[3]], axis=0),
    'U': np.concatenate([source_train_D[4], source_val_D[4]], axis=0),
}
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
target_train_ = {
    'X': target_train_D[0],
    'Y': target_train_D[1],
    'C': target_train_D[2],
    'W': target_train_D[3],
    'U': target_train_D[4],
}
target_val_D = MIMIC(dicom_splits[1]["val"], metadata,
                             CXR_EMB_PATH, REPORT_EMB_PATH,
                             batch_size=64, U=["U"],
                             verbose=True).generate_data()
target_val = {
    'X': target_val_D[0],
    'Y': target_val_D[1],
    'C': target_val_D[2],
    'W': target_val_D[3],
    'U': target_val_D[4],
}
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

# ## Training
## Dim red
if args.reduce is not None:
  proj = random_projection.GaussianRandomProjection(n_components=args.reduce).fit(source_train['X'])
  print(f"projecting to {proj.n_components_}")

  source_train_['X'] = proj.transform(source_train_['X'])
  target_train_['X'] = proj.transform(target_train_['X'])
  source_train['X'] = proj.transform(source_train['X'])
  target_train['X'] = proj.transform(target_train['X'])
  source_val['X'] = proj.transform(source_val['X'])
  target_val['X'] = proj.transform(target_val['X'])
  source_test['X'] = proj.transform(source_test['X'])
  target_test['X'] = proj.transform(target_test['X'])

print(f"\nsource train X shape: {source_train['X'].shape}")
print(f"source test X shape: {source_test['X'].shape}")
print(f"target train X shape: {target_train['X'].shape}")
print(f"target test X shape: {target_test['X'].shape}\n")

# ### Training Kernel Method
####################
# training...      #
####################

kernel = 'rbf'
alpha = 1e-3

N_FOLD = args.n_folds
N_PARAMS = args.n_params

MODEL = SVC

# # Regression Baselines
metrics = []

LR_param_grid = { 
              'C': np.logspace(-6, 6, args.n_params),  # Adjust the range as needed
          }
SVM_param_grid = { 
      'gamma': np.logspace(-6, 6, args.n_params),  # Adjust the range as needed
      'C': np.logspace(-6, 6, args.n_params),   # Adjust the range as needed
    }
KDE_param_grid = { 
              'bandwidth': np.logspace(-6, 6, args.n_params),  # Adjust the range as needed
          }

def get_best(data, param_grid=SVM_param_grid, model=MODEL(kernel='rbf'),
            seed=args.seed, n_params=N_PARAMS,
            n_fold=N_FOLD, sample_weight=None,
            LR=True):

  scorer = make_scorer(accuracy_score, greater_is_better=True)

  kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
  if LR:
    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring=scorer, verbose=4)
  else:
    grid_search = GridSearchCV(model, param_grid, cv=kf, verbose=4)

  grid_search.fit(*data, sample_weight=sample_weight)

  print("Best Parameters: ", grid_search.best_params_)

  return grid_search.best_estimator_, grid_search.best_params_

# ## ERM
print('')
print('*'*20+"ERM"+'*'*20)
erm_metrics = {'approach': 'ERM'}

source_erm, source_erm_hparams = get_best([source_train['X'], source_train['Y']])
RESULTS['hparams']['source_erm'] = source_erm_hparams
# source_erm.fit(source_train['X'], source_train['Y'])

erm_metrics['source -> source acc'] = accuracy_score(source_test['Y'], source_erm.predict(source_test['X']))
erm_metrics['source -> target acc'] = accuracy_score(target_test['Y'], source_erm.predict(target_test['X']))
erm_metrics['source -> source auc'] = roc_auc_score(source_test['Y'], source_erm.predict(source_test['X']))
erm_metrics['source -> target auc'] = roc_auc_score(target_test['Y'], source_erm.predict(target_test['X']))

RESULTS['source_erm'] = source_erm

target_erm, target_erm_hparams = get_best([target_train['X'], target_train['Y']])
RESULTS['hparams']['target_erm'] = target_erm_hparams
# target_erm.fit(target_train['X'], target_train['Y'])

erm_metrics['target -> target acc'] = accuracy_score(target_test['Y'], target_erm.predict(target_test['X']))
erm_metrics['target -> source acc'] = accuracy_score(source_test['Y'], target_erm.predict(source_test['X']))
erm_metrics['target -> target auc'] = roc_auc_score(target_test['Y'], target_erm.predict(target_test['X']))
erm_metrics['target -> source auc'] = roc_auc_score(source_test['Y'], target_erm.predict(source_test['X']))

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
domain_D = convert_data_Y2D(source_val, target_val)
# d_x = LogisticRegression(random_state=args.seed)
d_x, d_x_hparams = get_best([target_train_['X'], target_train_['Y']],
                            model=LogisticRegression(random_state=args.seed),
                            param_grid=LR_param_grid,)
RESULTS['hparams']['d_x'] = d_x_hparams
d_x.fit(domain_D['X'], domain_D['Y'])

# Compute sample weights
q_x_train = d_x.predict_proba(source_train_['X'])[:, 1]
source_sample_weight_train = q_x_train / (1. - q_x_train + 1e-3)

# RESULTS['d_x'] = d_x

# Compute sample weights
q_x_train = d_x.predict_proba(source_train_['X'])[:, 1]
source_sample_weight_train = q_x_train / (1. - q_x_train + 1e-3)

# source_covar_model = MODEL(kernel=kernel, alpha=alpha, gamma=source_gamma)
source_covar_model, source_covar_hparams = get_best([source_train_['X'], source_train_['Y']], sample_weight=source_sample_weight_train)
RESULTS['hparams']['source_covar'] = source_covar_hparams
# source_covar_model.fit(source_train['X'], source_train['Y'], sample_weight=source_sample_weight_train)

covar_metrics['source -> source acc'] = accuracy_score(source_test['Y'], source_covar_model.predict(source_test['X']))
covar_metrics['source -> target acc'] = accuracy_score(target_test['Y'], source_covar_model.predict(target_test['X']))
covar_metrics['source -> source auc'] = roc_auc_score(source_test['Y'], source_covar_model.predict(source_test['X']))
covar_metrics['source -> target auc'] = roc_auc_score(target_test['Y'], source_covar_model.predict(target_test['X']))

# RESULTS['source_covar_model'] = source_covar_model

# Compute sample weights
q_x_train = d_x.predict_proba(target_train_['X'])[:, 0]
target_sample_weight_train = q_x_train / (1. - q_x_train + 1e-3)

# target_covar_model = MODEL(kernel=kernel, alpha=alpha, gamma=target_gamma)
# target_covar_model.fit(target_train['X'], target_train['Y'], sample_weight=target_sample_weight_train)
target_covar_model, target_covar_hparams = get_best([target_train_['X'], target_train_['Y']], sample_weight=target_sample_weight_train)
RESULTS['hparams']['target_covar'] = target_covar_hparams

covar_metrics['target -> target acc'] = accuracy_score(target_test['Y'], target_covar_model.predict(target_test['X']))
covar_metrics['target -> source acc'] = accuracy_score(source_test['Y'], target_covar_model.predict(source_test['X']))
covar_metrics['target -> target auc'] = roc_auc_score(target_test['Y'], target_covar_model.predict(target_test['X']))
covar_metrics['target -> source auc'] = roc_auc_score(source_test['Y'], target_covar_model.predict(source_test['X']))

# RESULTS['target_covar_model'] = target_covar_model

metrics.append(covar_metrics)
print(covar_metrics)

# ## Label shift
print('')
print('*'*20+"LABEL"+'*'*20)
label_metrics = {'approach': 'LABEL'}

# Fit source and target KDE on oracle Y
# source_kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(source_val['Y'])
# target_kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(target_val['Y'])
source_kde, source_kde_hparams = get_best([source_val['Y']], model=KernelDensity(kernel='gaussian'), LR=False, param_grid=KDE_param_grid)
target_kde, target_kde_hparams = get_best([target_val['Y']], model=KernelDensity(kernel='gaussian'), LR=False, param_grid=KDE_param_grid)
RESULTS['hparams']['source_kde'] = source_kde_hparams
RESULTS['hparams']['target_kde'] = target_kde_hparams

# RESULTS['LABEL_source_kde'] = source_kde
# RESULTS['LABEL_target_kde'] = target_kde

# Compute sample weights q(Y)/p(Y)
log_q_y = target_kde.score_samples(source_train_['Y'])
log_p_y = source_kde.score_samples(source_train_['Y'])

source_sample_weight_train = np.exp(log_q_y - log_p_y)

# source_label_model = MODEL(kernel=kernel, alpha=alpha, gamma=source_gamma)
# source_label_model.fit(source_train['X'], source_train['Y'], sample_weight=source_sample_weight_train)
source_label_model, source_label_hparams = get_best([source_train_['X'], source_train_['Y']], sample_weight=source_sample_weight_train)
RESULTS['hparams']['source_label'] = source_label_hparams

label_metrics['source -> source acc'] = accuracy_score(source_test['Y'], source_label_model.predict(source_test['X']))
label_metrics['source -> target acc'] = accuracy_score(target_test['Y'], source_label_model.predict(target_test['X']))
label_metrics['source -> source auc'] = roc_auc_score(source_test['Y'], source_label_model.predict(source_test['X']))
label_metrics['source -> target auc'] = roc_auc_score(target_test['Y'], source_label_model.predict(target_test['X']))

# RESULTS['source_label_model'] = source_label_model

# Compute sample weights p(Y)/q(Y)
target_sample_weight_train = np.exp(log_p_y - log_q_y)

# target_label_model = MODEL(kernel=kernel, alpha=alpha, gamma=target_gamma)
# target_label_model.fit(target_train['X'], target_train['Y'], sample_weight=target_sample_weight_train)
target_label_model, target_label_hparams = get_best([target_train_['X'], target_train_['Y']], sample_weight=target_sample_weight_train)
RESULTS['hparams']['target_label'] = target_label_hparams

label_metrics['target -> target acc'] = accuracy_score(target_test['Y'], target_label_model.predict(target_test['X']))
label_metrics['target -> source acc'] = accuracy_score(source_test['Y'], target_label_model.predict(source_test['X']))
label_metrics['target -> target auc'] = roc_auc_score(target_test['Y'], target_label_model.predict(target_test['X']))
label_metrics['target -> source auc'] = roc_auc_score(source_test['Y'], target_label_model.predict(source_test['X']))

# RESULTS['target_label_model'] = target_label_model

metrics.append(label_metrics)
print(label_metrics)

# ## Summary
summary = pd.DataFrame.from_records(metrics)

RESULTS['summary'] = summary

print("SAVING")
with open(os.path.join(path, f'MIMIC_{date_string}.pkl'), 'wb') as f:
  pickle.dump(RESULTS, f)

with open(os.path.join(path, f'hparams.json'), 'w') as f:
  json.dump(RESULTS['hparams'], f, indent=4)
