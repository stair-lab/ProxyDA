from jax import random
import numpy as np
import jax.numpy as jnp

import os, pickle
import pandas as pd
import numpy as np
from KPLA.models.plain_kernel.multienv_adaptation import MultiEnvAdaptCAT, MultiEnvAdapt
from KPLA.models.plain_kernel.model_selection import tune_multienv_adapt_model_cv
from KPLA.models.plain_kernel.kernel_utils import standardise
import matplotlib.pyplot as plt

def gen_U(Z, n, key):
  if Z == 0:
    U = random.choice(key[0], jnp.arange(2), (n,), p=np.array([.9, .1]))
  if Z == 1:
    U = random.choice(key[0], jnp.arange(2), (n,), p=np.array([.1, .9]))
  else: # for target domain
    U = random.choice(key[0], jnp.arange(2), (n,), p=np.array([.8, .2]))
  
  return U


def gen_X(U, n,  key):
  X1 = random.normal(key[0],(n,))*1 - 0.5
  X2 = random.normal(key[1],(n,))*1 + 0.5
  X = (1-U)*X1 + U*X2
  return X


def gen_W(U, n,  key):
  W1 = random.normal(key[0],(n,))*.1 - 1
  W2 = random.normal(key[0],(n,))*.1 + 1
  W  = (1-U)*W1 + U*W2

  return W


def gen_Y(X, U, n):
  Y1 = -X
  Y2 = X
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
    X = ((jnp.asarray(gen_X(U, n, key=subkeysx))).T)
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
    data['X'] = jnp.array(X)
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
    X = ((jnp.asarray(gen_X(U, n, key=subkeysx))).T)
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
    data['X'] = jnp.array(X)
    data['W'] = jnp.array(W)
    data['Z'] = jnp.array(Z)
    data['Y'] = jnp.array(Y)
    data_list.append(data)
    print("target env id", n_env)

  return data_list


sd_train_list = sd_lst[:n_env]

source_train = gen_source_data(n_env, n, sd_train_list)

#test set only has 1000 samples
sd_test_list = sd_lst[n_env:n_env*2]
source_test = gen_source_data(n_env, 1000, sd_test_list)

#generate data from target domain


target_train = gen_target_data(2, n*2, [5949])
target_test  = gen_target_data(2, 1000, [5654])


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
plt.figure()
plt.scatter(source_train[0]['X'], source_train[0]['Y'], alpha=0.3, label='env 0')

plt.scatter(source_train[1]['X'], source_train[1]['Y'], alpha=0.3, label='env 1')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('plot_xy.png')

print('unique u', jnp.unique(source_train[0]['U']))
print('unique u', jnp.unique(source_train[1]['U']))
plt.figure()
plt.scatter(source_train[0]['U'], source_train[0]['X'], alpha=0.3, label='env 0')

plt.scatter(source_train[1]['U'], source_train[1]['X'], alpha=0.3, label='env 1')

plt.xlabel('U')
plt.ylabel('X')
plt.legend()
plt.savefig('plot_ux.png')

plt.figure()
plt.scatter(source_train[0]['U'], source_train[0]['W'], alpha=0.3, label='env 0')

plt.scatter(source_train[1]['U'], source_train[1]['W'], alpha=0.3, label='env 1')

plt.xlabel('U')
plt.ylabel('W')
plt.legend()
plt.savefig('plot_uw.png')

