"""
test the proposed model on dSprite multisource regression
"""
#Author: Katherine Tsai <kt14@illinois.edu>
#MIT License

import os, pickle
import pandas as pd
import numpy as np
from KPLA.models.plain_kernel.multienv_adaptation import MultiEnvAdaptCAT, MultiEnvAdapt
from KPLA.data.dSprite.gen_data_multi_source import generate_samples_Z2U_v2
from KPLA.data.dSprite.gen_data_wpc import latent_to_index, x_trans
import json
import hashlib
import datetime
from copy import deepcopy
import random
from sklearn import random_projection
import argparse

import matplotlib.pyplot as plt


#load data
#load data



# from components.approaches import compute_label_shift_correction_weights
def add_domain(d, d_add):
  if d == {}:
    return deepcopy(d_add)
  for k, v in d.items():
    d[k] = np.concatenate([v, deepcopy(d_add[k])], axis=0)
  return d

parser = argparse.ArgumentParser()

parser.add_argument('--alpha_1', type=float, default=2)
parser.add_argument('--beta_1', type=float, default=4)
parser.add_argument('--dist_1', type=str, default='beta', choices=['beta', 'uniform'])
parser.add_argument('--alpha_2', type=float, default=0.9)
parser.add_argument('--beta_2', type=float, default=1.)
parser.add_argument('--dist_2', type=str, default='uniform', choices=['beta', 'uniform'])

parser.add_argument('--N', type=int, default=2000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--output_dir', type=str, default='./v2/')

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
# expr_name = f"{args.alpha_2}_{unique_hash[:desired_length].ljust(desired_length, '0')}"
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

df_main = pd.DataFrame()

# # Exploration
A = np.random.uniform(0, 1, size=(10, 4096))
N = args.N

# ## Generate Data
alpha_1, beta_1 = args.alpha_1, args.beta_1
beta_2 = args.beta_2
alpha_2 = args.alpha_2

u_params = {
  'source_dist': args.dist_1,
  'source_dist_params': {'alpha': alpha_1, 'beta': beta_1},
  'target_dist': args.dist_2,
  'target_dist_params': {'min': alpha_2, 'max': beta_2},
}

RESULTS['u_params'] = u_params

N_ENVS = 4
RESULTS['N_ENVS'] = N_ENVS

# source_pzs = [np.sort(generate_n_simplex(N_ENVS)) for i in range(N_ENVS)]
source_pzs = [[1., 0., 0., 0.], [0., 1.0, 0., 0.], [0., 0., 1., 0.], [0., 0.0, 0.0, 1.]]
# target_pzs = [np.sort(generate_n_simplex(N_ENVS))[::-1]]
target_pzs = [[0., 0., 0., 1.]]

# U_dists = [(0.5, 5), (5, 0.5), (2, 2), (0.5, 0.5)]
U_dists = [(2, 4), (2, 5), (2, 6), (4, 2)]

RESULTS['SOURCE_PZS'] = source_pzs
RESULTS['TARGET_PZS'] = target_pzs
RESULTS['U_DISTS'] = U_dists

source_Zs = {i: np.random.choice(N_ENVS, size=(N,1), p=pz) for i, pz in enumerate(source_pzs)}
# target_Zs = {i: np.random.choice(N_ENVS, size=(N*N_ENVS,1), p=pz) for i, pz in enumerate(target_pzs)}
target_Zs = {0: -1*np.ones((N*N_ENVS, 1))}

RESULTS['SOURCE_ZS'] = source_Zs
RESULTS['TARGET_ZS'] = target_Zs

print("SOURCE")
source_train_data_list, source_val_data_list, source_test_data_list, source_imgs_dict_list = [], [], [], []

source_train, source_val, source_test, source_imgs_dict = {}, {}, {}, {}
for k, Z_s in source_Zs.items():
  print(f"source domain {k}")
  
  source_train_k, source_val_k, source_test_k, source_imgs_dict_k = generate_samples_Z2U_v2(Z_s,
    A, metadata, pos_X_basis, pos_X_basis_idx,
    pos_Y_basis, pos_Y_basis_idx, imgs, imgs_basis, dom=k, alpha_2=alpha_2, U_dists=U_dists)
  
  source_train_data_list.append(source_train_k)
  source_val_data_list.append(source_val_k)
  source_test_data_list.append(source_test_k)
  source_imgs_dict_list.append(source_imgs_dict_k)
  
  source_train = add_domain(source_train, source_train_k)
  source_val = add_domain(source_val, source_val_k)
  source_test = add_domain(source_test, source_test_k)
  source_imgs_dict = add_domain(source_imgs_dict, source_imgs_dict_k)
  
print("TARGET")
target_train_data_list, target_val_data_list, target_test_data_list, target_imgs_dict_list = [], [], [], []

target_train, target_val, target_test, target_imgs_dict = {}, {}, {}, {}
for k, Z_s in target_Zs.items():
  print(f"target domain {k}")
  target_train_k, target_val_k, target_test_k, target_imgs_dict_k = generate_samples_Z2U_v2(Z_s,
    A, metadata, pos_X_basis, pos_X_basis_idx,
    pos_Y_basis, pos_Y_basis_idx, imgs, imgs_basis, dom=k,
    target=True, alpha_2=alpha_2, U_dists=U_dists)
  
  target_train_data_list.append(target_train_k)
  target_val_data_list.append(target_val_k)
  target_test_data_list.append(target_test_k)
  target_imgs_dict_list.append(target_imgs_dict_k)
  
  target_train = add_domain(target_train, target_train_k)
  target_val = add_domain(target_val, target_val_k)
  target_test = add_domain(target_test, target_test_k)
  target_imgs_dict = add_domain(target_imgs_dict, target_imgs_dict_k)

RESULTS['source_train'] = source_train
RESULTS['source_val'] = source_val
RESULTS['source_test'] = source_test
RESULTS['source_imgs_dict'] = source_imgs_dict

RESULTS['source_train_data_list'] = source_train
RESULTS['source_val_data_list'] = source_val
RESULTS['source_test_data_list'] = source_test
RESULTS['source_imgs_dict_list'] = source_imgs_dict

RESULTS['target_train'] = target_train
RESULTS['target_val'] = target_val
RESULTS['target_test'] = target_test
RESULTS['target_imgs_dict'] = target_imgs_dict

RESULTS['target_train_data_list'] = target_train
RESULTS['target_val_data_list'] = target_val
RESULTS['target_test_data_list'] = target_test
RESULTS['target_imgs_dict_list'] = target_imgs_dict

# ## Training

# ## Kernel Method
# ### Reduce Dim
## Dim red
def do_random_projection(reference, x, proj, dim=16, dicts=[]):
  if proj is None:
    proj = random_projection.GaussianRandomProjection(n_components=dim).fit(reference)
#     proj = PCA(n_components=dim).fit(reference)
  out = []
  for d in deepcopy(dicts):
    assert isinstance(d, dict)
    d['X_orig'] = deepcopy(d['X'])
    d['X'] = proj.transform(d['X'])
    out.append(d)
  return proj, out, proj.transform(x)

DIM = 16

proj = random_projection.GaussianRandomProjection(n_components=DIM).fit(source_train['X'])
print(proj.n_components_)

DIM = 16
"""
source_train['X_orig'] = deepcopy(source_train['X'])
source_val['X_orig'] = deepcopy(source_val['X'])
source_test['X_orig'] = deepcopy(source_test['X'])

target_train['X_orig'] = deepcopy(target_train['X'])
target_val['X_orig'] = deepcopy(target_val['X'])
target_test['X_orig'] = deepcopy(target_test['X'])

ref = deepcopy(source_train['X'])
proj, source_train_data_list, source_train['X'] = do_random_projection(ref, source_train['X'], None,
                                                                dim=DIM, dicts=source_train_data_list)
_, source_val_data_list, source_val['X'] = do_random_projection(ref, source_val['X'], proj,
                                                                dim=DIM, dicts=source_val_data_list)
_, source_test_data_list, source_test['X'] = do_random_projection(ref, source_test['X'], proj,
                                                                  dim=DIM, dicts=source_test_data_list)

_, target_train_data_list, target_train['X'] = do_random_projection(ref, target_train['X'], proj,
                                                                    dim=DIM, dicts=target_train_data_list)
_, target_val_data_list, target_val['X'] = do_random_projection(ref, target_val['X'], proj,
                                                                dim=DIM, dicts=target_val_data_list)
_, target_test_data_list, target_test['X'] = do_random_projection(ref, target_test['X'], proj,
                                                                  dim=DIM, dicts=target_test_data_list)
"""
print(f"\nsource train X shape: {source_train['X'].shape}")
print(f"source val X shape: {source_val['X'].shape}")
print(f"source test X shape: {source_test['X'].shape}")
print(f"target train X shape: {target_train['X'].shape}")
print(f"target val X shape: {target_val['X'].shape}")
print(f"target test X shape: {target_test['X'].shape}\n")
#specify model

plt.figure(figsize=(30, 10))
for i in range(N_ENVS):
  train_data = source_train_data_list[i]

color_plate = ['g', 'b', 'r', 'k']
plt.subplot(1,3,1)
for i in range(N_ENVS):
  train_data = source_train_data_list[i]
  plt.title('U x proj X', fontsize=20)
  a = np.random.uniform(0, 1, (1, source_train['X'].shape[-1]))
  plt.scatter(source_train['U'].ravel(), x_trans(source_train['X'], A).ravel(), facecolors='None', edgecolor=color_plate[i], alpha=0.5, s=50, label=f'domain {i}')
  plt.xlabel('U', fontsize=20)
  plt.ylabel('proj X', fontsize=20)
  plt.legend()

plt.subplot(1,3,2)
for i in range(N_ENVS):
  train_data = source_train_data_list[i]
  plt.title('U x Y', fontsize=20)
  plt.scatter(source_train['U'].ravel(), source_train['Y'].ravel(), facecolors='None', edgecolor=color_plate[i], alpha=0.5, s=50, label=f'domain {i}')
  plt.xlabel('U', fontsize=20)
  plt.ylabel('Y', fontsize=20)
  plt.legend()


plt.subplot(1,3,3)
for i in range(N_ENVS):
  train_data = source_train_data_list[i]
  plt.title('proj X x Y', fontsize=20)
  a = np.random.uniform(0, 1, (1, source_train['X'].shape[-1]))
  plt.scatter(x_trans(source_train['X'], A).ravel(), source_train['Y'].ravel(), facecolors='None', edgecolor=color_plate[i], s=50, label=f'domain {i}')
  plt.xlabel('proj X', fontsize=20)
  plt.ylabel('Y', fontsize=20)
  plt.legend()

plt.tight_layout()
plt.savefig(f'plot_{i}.png')