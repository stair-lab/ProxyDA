"""
implements model selection method for the proposed model
"""
#Author: Katherine Tsai <kt14@illinois.edu>
#MIT License


import pandas as pd

from KPLA.models.plain_kernel.multienv_adaptation import MultiEnvAdaptCAT
from KPLA.models.plain_kernel.model_selection import tune_multienv_adapt_model_cv
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




lam_set = {'cme': 1e-3, 'm0': 1e-3, 'lam_min':-4, 'lam_max':-1}
method_set = {'cme': 'original', 'm0': 'original', 'm0': 'original'}

#specity the kernel functions for each estimator
kernel_dict = {}

kernel_dict['cme_w_xz'] = {'X': 'rbf', 'Y':'rbf_column'} #Y is W
kernel_dict['cme_w_x']  = {'X': 'rbf', 'Y': 'rbf_column'} # Y is W
kernel_dict['m0']       = {'X': 'rbf'}




best_estimator, best_parameter = tune_multienv_adapt_model_cv(source_train_list,
                                                        target_train_list,
                                                        source_test_list,
                                                        target_test_list,
                                                        method_set,
                                                        kernel_dict,
                                                        MultiEnvAdaptCAT,
                                                        task='c',
                                                        n_params=3,
                                                        n_fold=5,
                                                        min_log=-3,
                                                        max_log=0)


best_parameter['source_nsample'] = source_nsample
best_parameter['n_env'] = n_env

df = pd.DataFrame.from_records([best_parameter])
df.to_csv('classification_model_select.csv')


print('evaluation on best_model')
best_estimator.evaluation(task='c')


best_lam_set     = best_estimator.get_params()['lam_set']
best_scale       = best_estimator.get_params()['scale']
best_method_set  = best_estimator.get_params()['method_set']
best_kernel_dict = best_estimator.get_params()['kernel_dict']
split = False


print('best lam:', best_lam_set)
print('best scale:', best_scale)


split = False
scale = 1

estimator_full = MultiEnvAdaptCAT(source_train_list,
                                  target_train_list,
                                  source_test_list,
                                  target_test_list,
                                  False,
                                  1,
                                  lam_set,
                                  method_set,
                                  kernel_dict)
estimator_full.fit(task='c')


estimator_full.evaluation(task='c')


