"""
implements model selection for baseline methods.
"""
import os, pickle
import argparse
import json
import hashlib
import datetime
import numpy as np
import pandas as pd


from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KernelDensity
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn import random_projection

from sklearn.model_selection import GridSearchCV, KFold

from KPLA.data.dSprite.gen_data_wpc import latent_to_index, generate_samples

N_CPUS = os.cpu_count()

# from components.approaches import compute_label_shift_correction_weights

parser = argparse.ArgumentParser()

parser.add_argument('--alpha_1', type=float, default=2)
parser.add_argument('--beta_1', type=float, default=4)
parser.add_argument('--dist_1', type=str, default='beta',
                    choices=['beta', 'uniform'])
parser.add_argument('--alpha_2', type=float, default=0.)
parser.add_argument('--beta_2', type=float, default=1.)
parser.add_argument('--dist_2', type=str, default='uniform',
                    choices=['beta', 'uniform'])
parser.add_argument('--alpha', type=float)
parser.add_argument('--scale', type=float)

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
expr_name = f"{args.alpha_2}"

path = os.path.join(args.output_dir, expr_name)
os.makedirs(path, exist_ok=True)

############################################################################
RESULTS = {'args': vars(args), 'exp_path': path, 'hparams': {}}
############################################################################
# Load dataset
data_file = '/homes/oes2/mimic_experiments/datasets/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
dataset_zip = np.load(data_file, allow_pickle=True, encoding='bytes')

print('Keys in the dataset:', dataset_zip.keys())
imgs = dataset_zip['imgs']
latents_values = dataset_zip['latents_values']
latents_classes = dataset_zip['latents_classes']
metadata = dataset_zip['metadata'][()]

print('Metadata: \n', metadata)

# Define number of values per latents and functions to convert to indices
latents_sizes = metadata[b'latents_sizes']
latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))


U_basis = np.zeros((3,6))
for i in range(3):
  U_basis[i,1] = i
  U_basis[i,2] = 5
  U_basis[i, -2] = 16
basis_images = None

pos_X_basis_idx, pos_Y_basis_idx = 16, 0

pos_X_basis = metadata[b'latents_possible_values'][b'posX'][pos_X_basis_idx] - 0.5
pos_Y_basis = metadata[b'latents_possible_values'][b'posY'][pos_Y_basis_idx] - 0.5

indices_basis = latent_to_index(U_basis, metadata)
imgs_basis = imgs[indices_basis]

# # Exploration
A = np.random.uniform(0, 1, size=(10, 4096))
N = args.N

# ## Generate Data
alpha_1, beta_1 = args.alpha_1, args.beta_1
alpha_2, beta_2 = args.alpha_2, args.beta_2

u_params = {
  'source_dist': args.dist_1,
  'source_dist_params': {'alpha': alpha_1, 'beta': beta_1},
  'target_dist': args.dist_2,
  'target_dist_params': {'min': alpha_2, 'max': beta_2},
}

RESULTS['u_params'] = u_params

if args.dist_1 == 'uniform':
  U_s = np.random.uniform(alpha_1, beta_1, size=(N,1)) * 2 * np.pi
elif args.dist_1 == 'beta':
  U_s = np.random.beta(alpha_1, beta_1, size=(N,1)) * 2 * np.pi
else:
  raise NotImplementedError()

if args.dist_2 == 'uniform':
  U_t = np.random.uniform(alpha_2, beta_2, size=(N,1)) * 2 * np.pi
elif args.dist_2 == 'beta':
  U_t = np.random.beta(alpha_2, beta_2, size=(N,1)) * 2 * np.pi
else:
  raise NotImplementedError()

print("SOURCE")
source_train, source_val, source_test, source_imgs_dict = generate_samples(U_s,
  A, metadata, pos_X_basis, pos_X_basis_idx,
  pos_Y_basis, pos_Y_basis_idx, imgs, imgs_basis)

print("TARGET")
target_train, target_val, target_test, target_imgs_dict = generate_samples(U_t,
A, metadata, pos_X_basis, pos_X_basis_idx,
  pos_Y_basis, pos_Y_basis_idx, imgs, imgs_basis)

RESULTS['source_train'] = source_train
RESULTS['source_val'] = source_val
RESULTS['source_test'] = source_test
RESULTS['source_imgs_dict'] = source_imgs_dict

RESULTS['target_train'] = target_train
RESULTS['target_val'] = target_val
RESULTS['target_test'] = target_test
RESULTS['target_imgs_dict'] = target_imgs_dict

# ## Training

# ## Kernel Method
# ### Reduce Dim
## Dim red
DIM = 16

proj = random_projection.GaussianRandomProjection(n_components=DIM).fit(source_train['X'])

source_train['X'] = proj.transform(source_train['X'])
target_train['X'] = proj.transform(target_train['X'])
source_val['X'] = proj.transform(source_val['X'])
target_val['X'] = proj.transform(target_val['X'])
source_test['X'] = proj.transform(source_test['X'])
target_test['X'] = proj.transform(target_test['X'])


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

def get_best(X,
             y,
             seed=args.seed,
             n_params=N_PARAMS,
             n_fold=N_FOLD,
             sample_weight=None):
  kr = MODEL(kernel=kernel)

  param_grid = {
      'gamma': np.logspace(-4, 2, n_params)/DIM,  # Adjust the range as needed
      'alpha': np.logspace(-4, 2, n_params),   # Adjust the range as needed
  }

  scorer = make_scorer(mean_squared_error, greater_is_better=False)

  kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
  grid_search = GridSearchCV(kr, param_grid, cv=kf,
                            scoring=scorer, n_jobs=-1, verbose=4)

  grid_search.fit(X, y, sample_weight=sample_weight)

  print(f'Best Parameters: {grid_search.best_params_}')

  return grid_search.best_estimator_, grid_search.best_params_


# ## ERM

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
