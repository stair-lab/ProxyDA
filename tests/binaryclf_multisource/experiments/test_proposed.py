"""
test the proposed model on lsa classification dataset
"""
#Author: Katherine Tsai <kt14@illinois.edu>
#MIT License


import pandas as pd

from KPLA.models.plain_kernel.multienv_adaptation import MultiEnvAdaptCAT
from KPLA.data.data_generator import gen_multienv_class_discrete_z



source_nsample = 4000
target_nsample = source_nsample*3
n_env = 3
partition_dict = {'train': 0.8, 'test': 0.2}
seed = 192
result = {}
i = 1
#generate source with 3 environments
p_u_0 = 0.9
p_u = [p_u_0, 1-p_u_0]






summary = pd.DataFrame()


lam_set = {'cme': 1e-3, 'k0': 1e-3, 'lam_min':-4, 'lam_max':-1}
method_set = {'cme': 'original', 'k0': 'original', 'm0': 'original'}

#specity the kernel functions for each estimator
kernel_dict = {}

kernel_dict['cme_w_xz'] = {'X': 'rbf', 'Y':'rbf_column'} #Y is W
kernel_dict['cme_w_x']  = {'X': 'rbf', 'Y': 'rbf_column'} # Y is W
kernel_dict['k0']       = {'X': 'rbf'}

df = pd.read_csv('classification_model_select.csv')

best_lam_set = {'cme': df['alpha'].values[0],
                'k0':  df['alpha2'].values[0],
                'lam_min':-4, 
                'lam_max':-1}
best_scale =  df['scale'].values[0]
split = False

print(best_lam_set)
print(best_scale)


for seed in range(1922, 1932):

  source_train_list = []
  source_test_list = []


  source_train_list_mmd = []
  source_test_list_mmd = []

  for z_env in range(n_env):
    source_train, source_test = gen_multienv_class_discrete_z(z_env,
                                                              seed+z_env,
                                                              source_nsample,
                                                              partition_dict)
    source_train_list_mmd.append(source_train.copy())
    source_test_list_mmd.append(source_test.copy())

    source_train['Y'] = source_train['Y_one_hot']
    source_test['Y']  = source_test['Y_one_hot']

    # uncomment this if using multienv_adapt
    # comment this if using multienv_adapt_categorical
    # source_test['Z'] = source_test['Z_one_hot']
    # source_train['Z'] = source_train['Z_one_hot']

    source_train_list.append(source_train)
    source_test_list.append(source_test)


  #generate target
  target_train_list = []
  target_test_list = []

  target_train_list_mmd = []
  target_test_list_mmd = []

  target_train, target_test = gen_multienv_class_discrete_z(4,
                                                            seed+4,
                                                            target_nsample,
                                                            partition_dict)


  target_train_list_mmd.append(target_train.copy())
  target_test_list_mmd.append(target_test.copy())

  target_train['Y'] = target_train['Y_one_hot']
  target_test['Y']  = target_test['Y_one_hot']

  target_train_list.append(target_train)
  target_test_list.append(target_test)

  estimator_full = MultiEnvAdaptCAT(source_train_list,
                                    target_train_list,
                                    source_test_list,
                                    target_test_list,
                                    split,
                                    best_scale,
                                    best_lam_set,
                                    method_set,
                                    kernel_dict)
  estimator_full.fit(task='c')


  df = estimator_full.evaluation(task='c')
  df['seed'] = seed

  summary = pd.concat([summary, df])

print(summary)
summary.to_csv('multienv_classification.csv')


