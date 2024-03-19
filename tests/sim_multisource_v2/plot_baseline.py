from jax import random
import numpy as np
import jax.numpy as jnp

import os, pickle
import pandas as pd
import numpy as np
from KPLA.models.plain_kernel.multienv_adaptation import MultiEnvAdaptCAT, MultiEnvAdapt
from KPLA.models.plain_kernel.model_selection import tune_multienv_adapt_model_cv
from KPLA.baselines.multi_source_ccm import MultiSouceSimpleAdapt, MultiSourceUniformReg
from KPLA.baselines.multi_source_cat import MultiSourceCatReg
from KPLA.baselines.multi_source_ccm import MultiSouceSimpleAdapt, MultiSourceUniformReg
from multiprocessing import Pool, cpu_count

import copy
from KPLA.models.plain_kernel.kernel_utils import standardise

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import roc_auc_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import make_scorer, mean_squared_error
from KPLA.baselines.model_select import select_kernel_ridge_model
from sklearn.neighbors import KernelDensity

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--s", type=float, default=0.2)
parser.add_argument("--var", type=float, default=1.0)
parser.add_argument("--var1", type=float, default=1)
parser.add_argument("--mean", type=float, default=0)
parser.add_argument("--mean1", type=float, default=0)
parser.add_argument("--fixs", type=bool, default=False)
parser.add_argument("--fname", type=str)
args = parser.parse_args()


def convert_data_Y2D(source, target):
  source_Y = np.zeros_like(source['Y'])
  target_Y = np.ones_like(target['Y'])

  return {
      'covar': np.concatenate([source['covar'], target['covar']], axis=0),
      'Y': np.concatenate([source_Y, target_Y], axis=0).ravel()
  }
main_summary = pd.DataFrame()

for sdj in range(1):
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
      #X2 = random.normal(key[1],(n,))*args.var1 + args.mean1
      #X = (1-U)*X1 + U*X2
      return X1


    def gen_W(U, n,  key):
      W1 = random.normal(key[0],(n,))*.01 - 1
      W2 = random.normal(key[0],(n,))*.01 + 1
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
    sd_lst = sd_lst[::-1]

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


    sd_train_list = sd_lst[sdj*n_env:n_env*(sdj+1)]

    source_train = gen_source_data(n_env, n, sd_train_list)
    source_cat_train = {}
    for k in ['U', 'X', 'W', 'Z', 'Y']:
      temp = np.asarray(copy.deepcopy(source_train[0][k]))
      for data in source_train[1::]:
        print(data[k].shape, k)
        temp = np.concatenate((temp, np.asarray(data[k])))
      source_cat_train[k] = temp



    #test set only has 1000 samples
    sd_test_list =  sd_lst[(9-sdj)*n_env:n_env*(10-sdj)]
    source_test = gen_source_data(n_env, 1000, sd_test_list)

    source_cat_test = {}
    for k in ['U', 'X', 'W', 'Z', 'Y']:
      temp = np.asarray(copy.deepcopy(source_test[0][k]))
      for data in source_test[1::]:
        temp = np.concatenate((temp, np.asarray(data[k])))
      source_cat_test[k] = temp


    #generate data from target domain


    target_train = gen_target_data(2, n*2, [sd_lst[sdj]])
    #convert to nd array
    for k in ['U', 'X', 'W', 'Z', 'Y']:
      target_train[0][k] = np.asarray(target_train[0][k])
    target_test  = gen_target_data(2, 1000, [sd_lst[sdj]])

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

    N_PARAMS = 10
    N_FOLD = 5
    source_erm, source_erm_hparams = select_kernel_ridge_model(kr_model, 
                                                              source_cat_train['covar'],
                                                              source_cat_train['Y'],
                                                              n_params=N_PARAMS,
                                                                n_fold=N_FOLD,
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
                                                              n_params=N_PARAMS,
                                                              n_fold=N_FOLD,
                                                              min_val=-4,
                                                              max_val=3)
    erm_metrics['target -> target'] = mean_squared_error(target_test[0]['Y'], target_erm.predict(target_test[0]['covar']))
    erm_metrics['target -> source'] = mean_squared_error(source_cat_test['Y'], target_erm.predict(source_cat_test['covar']))

    RESULTS['target_erm'] = target_erm

    metrics.append(erm_metrics)
    print(erm_metrics)





    covar_metrics = {'approach': 'COVAR'}

    # Train domain classifier
    domain_D = convert_data_Y2D(source_cat_train, target_train[0])
    d_x = LogisticRegression(random_state=0)
    d_x.fit(domain_D['covar'], domain_D['Y'])

    # Compute sample weights
    q_x_train = d_x.predict_proba(source_cat_train['covar'])[:, 1]
    source_sample_weight_train = q_x_train / (1. - q_x_train + 1e-3)

    # RESULTS['d_x'] = d_x

    # Compute sample weights
    q_x_train = d_x.predict_proba(source_cat_train['covar'])[:, 1]
    source_sample_weight_train = q_x_train / (1. - q_x_train + 1e-3)

    # source_covar_model = MODEL(kernel=kernel, alpha=alpha, gamma=source_gamma)
    kr_model = MODEL(kernel=kernel)
    source_covar_model, source_covar_hparams = select_kernel_ridge_model(kr_model,
                                                                        source_cat_train['covar'],
                                                                        source_cat_train['Y'],
                                                                        sample_weight=source_sample_weight_train,
                                                                        n_params=N_PARAMS,
                                                                        n_fold=N_FOLD,
                                                                        min_val=-4,
                                                                        max_val=3)




    # source_covar_model.fit(source_train['X'], source_train['Y'], sample_weight=source_sample_weight_train)

    covar_metrics['source -> source'] = mean_squared_error(source_cat_test['Y'], source_covar_model.predict(source_cat_test['covar']))
    covar_metrics['source -> target'] = mean_squared_error(target_test[0]['Y'], source_covar_model.predict(target_test[0]['covar']))

    # RESULTS['source_covar_model'] = source_covar_model

    # Compute sample weights
    q_x_train = d_x.predict_proba(target_train[0]['covar'])[:, 0]
    target_sample_weight_train = q_x_train / (1. - q_x_train + 1e-3)

    # target_covar_model = MODEL(kernel=kernel, alpha=alpha, gamma=target_gamma)
    # target_covar_model.fit(target_train['X'], target_train['Y'], sample_weight=target_sample_weight_train)
    kr_model = MODEL(kernel=kernel)
    target_covar_model, target_covar_hparams = select_kernel_ridge_model(kr_model,
                                                                        target_train[0]['covar'],
                                                                        target_train[0]['Y'],
                                                                        target_sample_weight_train,
                                                                        n_params=N_PARAMS,
                                                                        n_fold=N_FOLD,
                                                                        min_val=-4,
                                                                        max_val=3)



    covar_metrics['target -> target'] = mean_squared_error(target_test[0]['Y'], target_covar_model.predict(target_test[0]['covar']))
    covar_metrics['target -> source'] = mean_squared_error(source_cat_test['Y'], target_covar_model.predict(source_cat_test['covar']))

    # RESULTS['target_covar_model'] = target_covar_model

    metrics.append(covar_metrics)
    print(covar_metrics)


    label_metrics = {'approach': 'LABEL'}

    # Fit source and target KDE on oracle Y
    source_kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(source_cat_train['Y'].reshape((-1,1)))
    target_kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(target_train[0]['Y'].reshape((-1,1)))

    # RESULTS['LABEL_source_kde'] = source_kde
    # RESULTS['LABEL_target_kde'] = target_kde

    # Compute sample weights q(Y)/p(Y)
    log_q_y = target_kde.score_samples(source_cat_train['Y'].reshape((-1,1)))
    log_p_y = source_kde.score_samples(source_cat_train['Y'].reshape((-1,1)))

    source_sample_weight_train = np.exp(log_q_y - log_p_y)

    # source_label_model = MODEL(kernel=kernel, alpha=alpha, gamma=source_gamma)
    # source_label_model.fit(source_train['X'], source_train['Y'], sample_weight=source_sample_weight_train)
    kr_model = MODEL(kernel=kernel)
    source_label_model, source_label_hparams = select_kernel_ridge_model(kr_model,
                                                                        source_cat_train['covar'],
                                                                        source_cat_train['Y'],
                                                                        sample_weight=source_sample_weight_train,
                                                                        n_params=N_PARAMS,
                                                                        n_fold=N_FOLD,
                                                                        min_val=-4,
                                                                        max_val=3)

    label_metrics['source -> source'] = mean_squared_error(source_cat_test['Y'], source_label_model.predict(source_cat_test['covar']))
    label_metrics['source -> target'] = mean_squared_error(target_test[0]['Y'], source_label_model.predict(target_test[0]['covar']))

    # RESULTS['source_label_model'] = source_label_model

    # Compute sample weights p(Y)/q(Y)
    target_sample_weight_train = np.exp(log_p_y - log_q_y)

    # target_label_model = MODEL(kernel=kernel, alpha=alpha, gamma=target_gamma)
    # target_label_model.fit(target_train['X'], target_train['Y'], sample_weight=target_sample_weight_train)
    kr_model = MODEL(kernel=kernel)
    target_label_model, target_label_hparams = select_kernel_ridge_model(kr_model,
                                                                        target_train[0]['covar'],
                                                                        target_train[0]['Y'],
                                                                        sample_weight=target_sample_weight_train,
                                                                        n_params=N_PARAMS,
                                                                        n_fold=N_FOLD,
                                                                        min_val=-4,
                                                                        max_val=3)

    label_metrics['target -> target'] = mean_squared_error(target_test[0]['Y'], target_label_model.predict(target_test[0]['covar']))
    label_metrics['target -> source'] = mean_squared_error(source_cat_test['Y'], target_label_model.predict(source_cat_test['covar']))
    metrics.append(label_metrics)
    print(label_metrics)
    # RESULTS['target_label_model'] = target_label_model


    msur_metrics = {'approach': 'multisource uniform'}

    source_msur = MultiSourceUniformReg(n_env=n_env, max_iter=3000)

    source_msur.fit(source_train)
    msur_metrics['source -> source'] = mean_squared_error(source_cat_test['Y'], source_msur.predict(source_cat_test['covar']))
    msur_metrics['source -> target'] = mean_squared_error(target_test[0]['Y'], source_msur.predict(target_test[0]['covar']))

    metrics.append(msur_metrics)
    print(msur_metrics)


    msca_metrics = {'approach': 'multisource cat'}

    source_msca = MultiSourceCatReg( max_iter=3000)
    source_msca.fit(source_train)
    msca_metrics['source -> source'] = mean_squared_error(source_cat_test['Y'], source_msca.predict(source_cat_test['covar']))
    msca_metrics['source -> target'] = mean_squared_error(target_test[0]['Y'], source_msca.predict(target_test[0]['covar']))

    metrics.append(msca_metrics)
    print(msca_metrics)


    mssa_metrics = {'approach': 'multisource simple aadapt'}



    n_fold = 5

    def run_single_loop(params):
      kf = KFold(n_splits=n_fold, random_state=None, shuffle=False)
      errs = []
      best_err_i = np.inf
      best_model_i = None
      split_idx = kf.split(source_train[0]["X"])

      for _, (train_idx, test_idx) in enumerate(split_idx):

        def split_data_cv(sidx, t_list):
          list_len = len(t_list)
          return [{k:v[sidx] for k,v in t_list[j].items()} for j in range(list_len)]
        source_train_cv_train = split_data_cv(train_idx, source_train)
        source_train_cv_test  = split_data_cv(test_idx,  source_train)

        mssa = MultiSouceSimpleAdapt(n_env = len(source_train_cv_train),
                                    bandwidth   = params, 
                                    max_iter= 3000,
                                    task='r')


        mssa.fit(source_train_cv_train)
        ##select parameters from source
        acc_err = 0
        for sd in source_train_cv_test:
          acc_err += mean_squared_error(sd["Y"].ravel(), mssa.predict(sd["X"]))


        errs.append(acc_err)
        ## select parameters from target
        if acc_err < best_err_i:
          best_err_i = acc_err
          best_model_i = mssa
          b_params = {"bandwidth": params}


      return best_model_i, np.mean(errs), b_params

    n_params = 20
    n_fold   = 5
    min_val  = -4
    max_val  = 2

    length_scale = np.logspace(min_val, max_val, n_params)

    params_batch = length_scale

    results = [0]*length_scale.shape[0]
    for i in range(length_scale.shape[0]):
      results[i] = run_single_loop(params_batch[i])

    #pool = Pool(10)
    #print(f"start multiprocessing, number of cpu: {cpu_count()-1}")

    #results = pool.map(run_single_loop, prams_batch)
    #pool.close()


    model_batch  = [r[0] for r in results]
    err_batch    = [r[1] for r in results]
    params_batch = [r[2] for r in results]
    idx = np.argmin(err_batch)

    best_msa = model_batch[idx]
    best_params = params_batch[idx]

    print("optimal parameters", params_batch[idx])

    best_msa.fit(source_train)



    source_msa = MultiSouceSimpleAdapt(n_env   = n_env,
                                    bandwidth = best_params['bandwidth'], 
                                    max_iter = 3000, 
                                    task      ='r')
    source_msa.fit(source_train)
    mssa_metrics['source -> source'] = mean_squared_error(source_cat_test['Y'], source_msa.predict(source_cat_test['X']))
    mssa_metrics['source -> target'] = mean_squared_error(target_test[0]['Y'], source_msa.predict(target_test[0]['X']))


    metrics.append(mssa_metrics)
    print(mssa_metrics)

    summary = pd.DataFrame.from_records(metrics)
    summary['pU=0'] = s1
    summary['seed'] = sdj
    main_summary = pd.concat([main_summary, summary])

  main_summary.to_csv(f'sweep_baseline_v2_v4_seeds{sdj}.csv', index=False)