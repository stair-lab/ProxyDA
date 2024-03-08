from jax import random
import numpy as np
import jax.numpy as jnp

import os, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from KPLA.models.plain_kernel.multienv_adaptation import MultiEnvAdaptCAT, MultiEnvAdapt
from KPLA.models.plain_kernel.model_selection import tune_multienv_adapt_model_cv

from KPLA.models.plain_kernel.kernel_utils import standardise
import itertools
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--a", type=float, default=2)
parser.add_argument("--b", type=float, default=2)
parser.add_argument("--var", type=float, default=0.9)
parser.add_argument("--var1", type=float, default=1)
parser.add_argument("--mean", type=float, default=-0.5)
parser.add_argument("--mean1", type=float, default=0.5)
parser.add_argument("--fixs", type=bool, default=False)
parser.add_argument("--fname", type=str, default='test.csv')
parser.add_argument("--outdir", type=str, default='./')
args = parser.parse_args()

out_dir = args.outdir
a = args.a
b = args.b

file_path = './model_select/'
main_df = pd.DataFrame()

a_list = [1,2,3,4,5]
b_list = [1,2,3,4,5]

for sdj in range(10):
  for a,b in itertools.product(a_list, b_list):
    fname = f'test_proposed_v2_onehot_a{a}b{b}_fixscale_{sdj}.csv'
    def gen_U(Z, n, key):
      print(Z)
      if Z == 0:
        #reversed setting (0.1, 0.9) #original setting (0.9, 0.1)
        print('Z is 0')
        U = random.beta(key[0], 2, 5, (n,))
      elif Z == 1:
        print('Z is 1')
        U = random.beta(key[0], 5, 2, (n,))
      else: # for target domain
        print('Z is 2')
        U = random.beta(key[0], a, b, (n,))
      
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
      #return U


    def gen_Y(X, U, n):
      #Y1 = -X
      #Y2 = X
      #Y = (1-U)*Y1 + U*Y2
      Y = (2*U-1)*X
      return Y



    ####################
    # generate data    #
    ####################

    n_env = 2
        


    seed_list = {}
    sd_lst= [5949, 7422, 4388, 2807, 5654, 5518, 1816, 1102, 9886, 1656, 4379,
          2029, 8455, 4987, 4259, 2533, 9783, 7987, 1009, 2297]
    #reversed the list
    #sd_lst= sd_lst[::-1]


    #set number of samples to train
    n = 2000

    mean_z =  [-0.5, 0., 0.5, 1.]
    sigma_z = [  1., 1.,  1., 1.]

    prob_list = [[1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]]


    def gen_source_data(n_env, n, seed_list):
      data_list = []
      for env_id, sd in enumerate(seed_list):
        print("source env id", env_id, sd)
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
        
        X = ((jnp.asarray(gen_X(U, n, key=subkeysx))).T)
        W = ((jnp.asarray(gen_W(U, n, key=subkeysw))).T)
        Y = (gen_Y(X, U, n))

        #encode one-hot
        Z = np.zeros((n, 3))
        Z[:, env_id] = 1
        Z = jnp.asarray(Z)
        print(Z)
        #Z = (jnp.ones(n).T*env_id)
        # Standardised sample

        #U = standardise(np.asarray(U)) [0]
        #Z = standardise(np.asarray(Z)) [0]
        #W = standardise(np.asarray(W)) [0]
        #X = standardise(np.asarray(X)) [0]
        #Y = standardise(np.asarray(Y)) [0]

        data = {}



        data['U'] = jnp.array(U)
        data['X'] = jnp.array(X)
        data['W'] = jnp.array(W)
        data['Z'] = jnp.array(Z)
        data['Y'] = jnp.array(Y)
        data_list.append(data)
        

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
        X = ((jnp.asarray(gen_X(U, n, key=subkeysx))).T)
        W = ((jnp.asarray(gen_W(U, n, key=subkeysw))).T)
        Y = (gen_Y(X, U, n))
        Z = np.zeros((n, 3))
        Z[:, -1] = 1
        Z = jnp.asarray(Z)
        print(Z)
        #Z = (jnp.ones(n).T*n_env)
        # Standardised sample
        #U = standardise(np.asarray(U)) [0]
        #Z = standardise(np.asarray(Z)) [0]
        #W = standardise(np.asarray(W)) [0]
        #X = standardise(np.asarray(X)) [0]
        #Y = standardise(np.asarray(Y)) [0]

        data = {}



        data['U'] = jnp.array(U)
        data['X'] = jnp.array(X)
        data['W'] = jnp.array(W)
        data['Z'] = jnp.array(Z)
        data['Y'] = jnp.array(Y)
        data_list.append(data)
        print("target env id", n_env)

      return data_list


    #sd_train_list = sd_lst[:n_env]
    sd_train_list = sd_lst[sdj*n_env:n_env*(sdj+1)]

    print('generate training data for source domain')
    source_train = gen_source_data(n_env, n, sd_train_list)

    #test set only has 1000 samples
    print('generate testing data for source domain')
    #sd_test_list = sd_lst[n_env:n_env*2]
    sd_test_list = sd_lst[(9-sdj)*n_env:n_env*(10-sdj)]
    source_test = gen_source_data(n_env, 1000, sd_test_list)

    #generate data from target domain


    #target_train = gen_target_data(2, n*2, [5949])
    #target_train = gen_target_data(2, n, [5949])
    #target_test  = gen_target_data(2, 1000, [5654])
    target_train = gen_target_data(2, n, [sd_lst[sdj]])
    target_test  = gen_target_data(2, 1000, [sd_lst[sdj+1]])


    print('data generation complete')
    print('number of source environments:', len(source_train))
    print('source_train number of samples: ', source_train[0]['X'].shape[0]*n_env)
    print('source_test  number of samples: ', source_test[0]['X'].shape[0])
    print('number of target environments:', len(target_train))
    print('target_train number of samples: ', target_train[0]['X'].shape[0])
    print('target_test  number of samples: ', target_test[0]['X'].shape[0])




    print('data generation complete')
    print('number of source environments:', len(source_train))
    print('source_train number of samples: ', source_train[0]['X'].shape[0]*n_env)
    print('source_test  number of samples: ', source_test[0]['X'].shape[0])
    print('number of target environments:', len(target_train))
    print('target_train number of samples: ', target_train[0]['X'].shape[0])
    print('target_test  number of samples: ', target_test[0]['X'].shape[0])

    #check shape
    print('X shape', source_train[0]['X'].shape)
    print('Y shape', source_train[0]['Y'].shape)
    print('W shape', source_train[0]['W'].shape)
    print('U shape', source_train[0]['U'].shape)


    lam_set = {'cme': 1e-4, 'm0': 1e-4, 'lam_min':-4, 'lam_max':-1}
    method_set = {'cme': 'original', 'm0': 'original', 'm0': 'original'}

    #specity the kernel functions for each estimator
    kernel_dict = {}

    X_kernel = 'rbf'
    W_kernel = 'rbf'
    kernel_dict['cme_w_xz'] = {'X': X_kernel, 'Y': W_kernel, 'Z': 'binary'} #Y is W
    kernel_dict['cme_w_x']  = {'X': X_kernel, 'Y': W_kernel} # Y is W
    kernel_dict['m0']       = {'X': X_kernel}
    scale =  1
    split=False


    df = pd.read_csv(file_path+fname)

    best_lam_set = {'cme': df['alpha'].values[0],
                    'm0':  df['alpha2'].values[0],
                    'lam_min':-4, 
                    'lam_max':-1}
    best_scale =  df['scale'].values[0]

    split = False


    print('best lam:', best_lam_set)
    print('best scale:', best_scale)


    split = False


    estimator_full = MultiEnvAdapt(source_train,
                                      target_train,
                                      source_test,
                                      target_test,
                                      split,
                                      best_scale,
                                      best_lam_set,
                                      method_set,
                                      kernel_dict)
    estimator_full.fit(task='r')


    df = estimator_full.evaluation(task='r')
    df['a'] = a
    df['b'] = b
    main_df = pd.concat([main_df, df])


  main_df.to_csv(f'sweep_proposed_v2_seed{sdj}.csv', index=False)



