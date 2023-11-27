"""
implementation of model selection for deep kernel method
"""
#Author: Katherine Tsai <kt14@illinois.edu>
#MIT LICENSE

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
import numpy as np
import jax.numpy as jnp
from itertools import product

from KPLA.models.deep_kernel.deep_adaptation import DeepFullAdapt
from KPLA.models.deep_kernel.multienv_deep_adaptation import DeepMultiEnvAdapt
import copy
import json
def tune_deep_adapt_model_cv(source_train,
                             target_train,
                             model,
                             task,
                             n_params,
                             n_fold,
                             sample_configs,
                             config_filepath,
                             lam_min_log,
                             lam_max_log,
                             iter_min_log,
                             iter_max_log,
                             decay_min_log,
                             decay_max_log,
                             split
                             ):
  
  #prepare param_dict
  
  params = product(np.logspace(lam_min_log, lam_max_log, n_params).tolist(),#lam
                   np.logspace(iter_min_log, iter_max_log, n_params).tolist(),#cme_iter
                   np.logspace(iter_min_log, iter_max_log, n_params).tolist(),#h0_iter                  
                   np.logspace(decay_min_log, decay_max_log, n_params).tolist(),#wxc_decay
                  ) #h0iter

  #wite config files
  def write_configs(config_params, sample_config, idx):
    new_configs = copy.deepcopy(sample_config)
    new_configs["model"] = config_params
    fname = new_configs["data"]["name"] + "_config_" + idx + ".json"
    with open(config_filepath+fname, "w") as fp:
      json.dump(new_configs, fp)
  
  count = 0
  for alpha, cme_iter, h0_iter, weight_decay in params:
    sample_params = copy.deepcopy(sample_configs["model"])
    lam_keys = list(sample_params["lam_set"].keys())
    new_dict = {dname:alpha for dname in lam_keys}
    #update the dict
    sample_params["lam_set"] = new_dict
    sample_params["cme_iter"] = cme_iter
    sample_params["h0_iter"] = h0_iter
    sample_params["w_weight_decay"] = weight_decay
    sample_params["x_weight_decay"] = weight_decay
    sample_params["c_weight_decay"] = weight_decay

    write_configs(sample_params, sample_configs, count)
    

    #split data
    kf = KFold(n_splits=n_fold, random_state=None, shuffle=False)
    for i, (train_idx, test_idx) in enumerate(kf.split(source_train.X)):
      source_train_cv_train = 
      source_train_cv_train.to_gpu()
      source_train_cv_val   = 
    #train model

    #update count
    count += 1


def tune_deep_multi_adapt_model_cv():
  pass
