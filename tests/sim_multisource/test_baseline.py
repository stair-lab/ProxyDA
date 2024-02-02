from jax import random
import numpy as np
import jax.numpy as jnp

import os, pickle
import pandas as pd
import numpy as np
from KPLA.models.plain_kernel.multienv_adaptation import MultiEnvAdaptCAT, MultiEnvAdapt
from KPLA.models.plain_kernel.model_selection import tune_multienv_adapt_model_cv
import copy
from KPLA.models.plain_kernel.kernel_utils import standardise

from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import roc_auc_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import make_scorer, mean_squared_error
from KPLA.baselines.model_select import select_kernel_ridge_model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--s", type=float, default=0.2)
parser.add_argument("--var", type=float, default=1.0)
parser.add_argument("--var1", type=float, default=1.0)
parser.add_argument("--mean", type=float, default=-0.5)
parser.add_argument("--mean1", type=float, default=0.5)
parser.add_argument("--fixs", type=bool, default=False)
parser.add_argument("--fname", type=str)
args = parser.parse_args()

#s1= args.s 
#s2 = 1. - args.s
main_summary = pd.DataFrame()
for s1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
  s2 = 1-s1
  def gen_U(Z, n, key):
    if Z == 0:
      U = random.choice(key[0], jnp.arange(2), (n,), p=np.array([.9, .1]))
    elif Z == 1:
      U = random.choice(key[0], jnp.arange(2), (n,), p=np.array([.1, .9]))
    else: # for target domain
      U = random.choice(key[0], jnp.arange(2), (n,), p=np.array([s1, s2]))
    
    return U


  def gen_X(U, n,  key):
    X1 = random.normal(key[0],(n,))*args.var + args.mean
    X2 = random.normal(key[1],(n,))*args.var1 + args.mean1
    X = (1-U)*X1 + U*X2
    return X


  def gen_W(U, n,  key):
    W1 = random.normal(key[0],(n,))*.1 - 1
    W2 = random.normal(key[0],(n,))*.1 + 1
    W  = (1-U)*W1 + U*W2

    return W


  def gen_Y(X, U, n):
    Y1 = -X.squeeze()
    Y2 = X.squeeze()
    Y = (1-U)*Y1 + U*Y2
    return Y



  ####################
  # generate data    #
  ####################

  n_env = 2
      


  seed_list = {}
  sd_lst= [5949, 7422, 4388, 2807, 5654, 5518, 1816, 1102, 9886, 1656, 4379,
        2029, 8455, 4987, 4259, 2533, 9783, 7987, 1009, 2297]


  #set number of samples to train
  n = 2000




  def gen_source_data(n_env, n, seed_list):
    data_list = []
    for env_id, sd in enumerate(seed_list):
      seed=[]
      seed1=sd+5446
      seed2=sd+3569
      seed3=sd+10
      seed4=sd+1572
      seed5=sd+42980
      seed6=sd+368641

      seed=[seed1,seed2,seed3,seed4,seed5,seed6]


      keyu = random.PRNGKey(seed1)
      keyu, *subkeysu = random.split(keyu, 4)

      keyx = random.PRNGKey(seed2)
      keyx, *subkeysx = random.split(keyx, 4)


      keyw = random.PRNGKey(seed3)
      keyw, *subkeysw = random.split(keyw, 4)

      keyz = random.PRNGKey(seed4)
      keyz, *subkeysz = random.split(keyz, 4)

      U = ((jnp.asarray(gen_U(env_id, n, key=subkeysu))).T)
      X = ((jnp.asarray(gen_X(U, n, key=subkeysx))).T).reshape(-1,1)
      W = ((jnp.asarray(gen_W(U, n, key=subkeysw))).T)
      Y = (gen_Y(X, U, n))
      Z = (jnp.ones(n).T*env_id)
      # Standardised sample
      #U = standardise(np.asarray(U)) [0]
      #Z = standardise(np.asarray(Z)) [0]
      #W = standardise(np.asarray(W)) [0]
      #X = standardise(np.asarray(X)) [0]
      #Y = standardise(np.asarray(Y)) [0]

      data = {}



      data['U'] = jnp.array(U)
      data['X'] = jnp.array(X).reshape(-1,1)
      data['W'] = jnp.array(W)
      data['Z'] = jnp.array(Z)
      data['Y'] = jnp.array(Y)
      data_list.append(data)
      print("source env id", env_id)

    return data_list


  def gen_target_data(n_env, n, seed_list):
    data_list = []
    for sd in (seed_list):
      seed=[]
      seed1=sd+5446
      seed2=sd+3569
      seed3=sd+10
      seed4=sd+1572
      seed5=sd+42980
      seed6=sd+368641

      seed=[seed1,seed2,seed3,seed4,seed5,seed6]


      keyu = random.PRNGKey(seed1)
      keyu, *subkeysu = random.split(keyu, 4)

      keyx = random.PRNGKey(seed2)
      keyx, *subkeysx = random.split(keyx, 4)


      keyw = random.PRNGKey(seed3)
      keyw, *subkeysw = random.split(keyw, 4)

      keyz = random.PRNGKey(seed4)
      keyz, *subkeysz = random.split(keyz, 4)

      U = ((jnp.asarray(gen_U(n_env, n, key=subkeysu))).T)
      X = ((jnp.asarray(gen_X(U, n, key=subkeysx))).T).reshape(-1,1)
      W = ((jnp.asarray(gen_W(U, n, key=subkeysw))).T)
      Y = (gen_Y(X, U, n))
      Z = (jnp.ones(n).T*n_env)
      # Standardised sample
      #U = standardise(np.asarray(U)) [0]
      #Z = standardise(np.asarray(Z)) [0]
      #W = standardise(np.asarray(W)) [0]
      #X = standardise(np.asarray(X)) [0]
      #Y = standardise(np.asarray(Y)) [0]

      data = {}



      data['U'] = jnp.array(U)
      data['X'] = jnp.array(X).reshape(-1,1)
      data['W'] = jnp.array(W)
      data['Z'] = jnp.array(Z)
      data['Y'] = jnp.array(Y)
      data_list.append(data)
      print("target env id", n_env)

    return data_list


  sd_train_list = sd_lst[:n_env]

  source_train = gen_source_data(n_env, n, sd_train_list)
  source_cat_train = {}
  for k in ['U', 'X', 'W', 'Z', 'Y']:
    temp = np.asarray(copy.deepcopy(source_train[0][k]))
    for data in source_train[1::]:
      print(data[k].shape, k)
      temp = np.concatenate((temp, np.asarray(data[k])))
    source_cat_train[k] = temp



  #test set only has 1000 samples
  sd_test_list = sd_lst[n_env:n_env*2]
  source_test = gen_source_data(n_env, 1000, sd_test_list)

  source_cat_test = {}
  for k in ['U', 'X', 'W', 'Z', 'Y']:
    temp = np.asarray(copy.deepcopy(source_test[0][k]))
    for data in source_test[1::]:
      temp = np.concatenate((temp, np.asarray(data[k])))
    source_cat_test[k] = temp


  #generate data from target domain


  target_train = gen_target_data(2, n*2, [5949])
  #convert to nd array
  for k in ['U', 'X', 'W', 'Z', 'Y']:
    target_train[0][k] = np.asarray(target_train[0][k])
  target_test  = gen_target_data(2, 1000, [5654])

  for k in ['U', 'X', 'W', 'Z', 'Y']:
    target_test[0][k] = np.asarray(target_test[0][k])


  print('data generation complete')
  print('number of source environments:', len(source_train))
  print('source_train number of samples: ', source_train[0]['X'].shape[0]*n_env)
  print('source_test  number of samples: ', source_test[0]['X'].shape[0])
  print('number of target environments:', len(target_train))
  print('target_train number of samples: ', target_train[0]['X'].shape[0])
  print('target_test  number of samples: ', target_test[0]['X'].shape[0])

  RESULTS = {}
  metrics = []
  MODEL = KernelRidge
  kernel = 'rbf'
  #source_erm = MODEL(kernel=kernel,
  #                    alpha=1e-3,
  #                    gamma=1e-3)
  kr_model = MODEL(kernel=kernel)
  add_w = False
  #regress with W
  if add_w:
    print(source_cat_train['X'].shape)
    print(source_cat_train['W'].shape)
    source_cat_train['covar'] = np.stack((source_cat_train['X'].squeeze(), source_cat_train['W'])).T
    source_cat_test['covar'] = np.stack((source_cat_test['X'].squeeze(), source_cat_test['W'])).T
    target_train[0]['covar'] = np.stack((target_train[0]['X'].squeeze(), target_train[0]['W'])).T
    target_test[0]['covar']  = np.stack((target_test[0]['X'].squeeze(), target_test[0]['W'])).T
  else:
    source_cat_train['covar'] = source_cat_train['X']
    source_cat_test['covar']  = source_cat_test['X']
    target_train[0]['covar']  = target_train[0]['X']
    target_test[0]['covar']   = target_test[0]['X']
  source_erm, source_erm_hparams = select_kernel_ridge_model(kr_model, 
                                                            source_cat_train['covar'],
                                                            source_cat_train['Y'],
                                                            n_params=10,
                                                              n_fold=5,
                                                              min_val=-4,
                                                              max_val=3)
  #source_erm.fit(source_cat_train['X'],
  #                source_cat_train['Y'])

  # source_erm.fit(source_train['X'], source_train['Y'])
  erm_metrics = {'approach': 'ERM'}
  erm_metrics['source -> source'] = mean_squared_error(source_cat_test['Y'], source_erm.predict(source_cat_test['covar']))
  erm_metrics['source -> target'] = mean_squared_error(target_test[0]['Y'], source_erm.predict(target_test[0]['covar']))



  #target_erm = MODEL(kernel=kernel,
  #                    alpha=1e-3,
  #                    gamma=1e-3)
  #target_erm.fit(target_train[0]['X'],
  #                target_train[0]['Y'])

  kr_model = MODEL(kernel=kernel)
  target_erm, target_erm_hparams = select_kernel_ridge_model(kr_model, 
                                                            target_train[0]['covar'],
                                                            target_train[0]['Y'],
                                                            n_params=10,
                                                              n_fold=5,
                                                              min_val=-4,
                                                              max_val=3)
  erm_metrics['target -> target'] = mean_squared_error(target_test[0]['Y'], target_erm.predict(target_test[0]['covar']))
  erm_metrics['target -> source'] = mean_squared_error(source_cat_test['Y'], target_erm.predict(source_cat_test['covar']))

  RESULTS['target_erm'] = target_erm

  metrics.append(erm_metrics)
  print(erm_metrics)
  summary = pd.DataFrame.from_records(metrics)
  summary['pU=0'] = s1
  main_summary = pd.concat([main_summary, summary])

main_summary.to_csv(f'sweep_baseline_v3_add_w{add_w}.csv', index=False)