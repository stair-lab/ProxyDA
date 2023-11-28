"""
implements model selection for proposed method
"""
import os, pickle
import argparse
import json
import hashlib
import datetime
from copy import deepcopy
import numpy as np
import pandas as pd


from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import  mean_squared_error
from sklearn import random_projection

from sklearn.model_selection import KFold


from itertools import product
from KPLA.data.dSprite.gen_data_wpc import latent_to_index, generate_samples
from KPLA.models.plain_kernel.adaptation import FullAdapt
from KPLA.models.plain_kernel.model_selection import tune_adapt_model_cv
import random

# from components.approaches import compute_label_shift_correction_weights

parser = argparse.ArgumentParser()

parser.add_argument('--alpha_1', type=float, default=2)
parser.add_argument('--beta_1', type=float, default=4)
parser.add_argument('--dist_1', type=str, default='beta', choices=['beta', 'uniform'])
parser.add_argument('--alpha_2', type=float, default=0.)
parser.add_argument('--beta_2', type=float, default=1.)
parser.add_argument('--dist_2', type=str, default='uniform', choices=['beta', 'uniform'])

parser.add_argument('--N', type=int, default=10000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--output_dir', type=str, default='./')

args = parser.parse_args()

random.seed(args.seed)
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
data_path = '../../../KPLA/data/dSprite/'
dataset_zip = np.load(data_path+'/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
                      allow_pickle=True,
                      encoding='bytes')

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

print('SOURCE')
source_train, source_val, source_test, source_imgs_dict = generate_samples(U_s,
  A, metadata, pos_X_basis, pos_X_basis_idx,
  pos_Y_basis, pos_Y_basis_idx, imgs, imgs_basis)

print('TARGET')
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


method_set = {'cme': 'original', 'h0': 'original', 'm0': 'original'}
X_kernel = 'rbf'
kernel_dict = {}
kernel_dict['cme_w_xc'] = {'X': X_kernel, 'C': 'rbf', 'Y':'rbf'} #Y is W
kernel_dict['cme_wc_x'] = {'X': X_kernel, 'Y': 'rbf'} # Y is (W,C)

kernel_dict['h0']       = {'C': 'rbf'}
kernel_dict['m0']       = {'C': 'rbf', 'X': X_kernel}

estimator_full, best_params = tune_adapt_model_cv(source_train,
                                                  target_train,
                                                  source_test,
                                                  target_test,
                                                  method_set,
                                                  kernel_dict,
                                                  FullAdapt,
                                                  task='r',
                                                  n_params=N_PARAMS,
                                                  n_fold=N_FOLD,
                                                  min_log=-4,
                                                  max_log=0)
RESULTS['kernel_dict'] = kernel_dict
RESULTS['hparams']['kernel'] = {'alpha': best_params['alpha'],
                                'alpha2': best_params['alpha2'],
                                'scale': best_params['scale']}

print('')
print('*'*20+"LSA"+'*'*20)
res_full = estimator_full.evaluation()

RESULTS['estimator_full'] = estimator_full

# ## Summary
full_adapt_metrics = {
k.replace('-', ' -> '): v for k, v in zip(res_full.task.values, res_full['predict error.l2'].values)
}
full_adapt_metrics['approach'] = 'PROXY (W, C) (baseline)'

full_adapt_metrics = deepcopy(full_adapt_metrics)
full_adapt_metrics['approach'] = 'PROXY (W, C) '
full_adapt_metrics['source -> target'] = full_adapt_metrics['adaptation']

del full_adapt_metrics['adaptation']

print(full_adapt_metrics)

proxy_metrics = [full_adapt_metrics]

# summary = pd.DataFrame.from_records(metrics + proxy_metrics)
summary = pd.DataFrame.from_records(proxy_metrics)

RESULTS['summary'] = summary

with open(os.path.join(path, f'dsprites_data_{date_string}.pkl'), 'wb') as f:
  pickle.dump(RESULTS, f)

with open(os.path.join(path, f'hparams.json'), 'w') as f:
  json.dump(RESULTS['hparams'], f, indent=4)