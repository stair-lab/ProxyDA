"""
 A demo of deep feature net adaptation (full adaptation)
"""
#Author:Katherine Tsai<kt14@illinois.edu>
#MIT LISENCE

import torch
import logging
import numpy as np
import pandas as pd
import json

from KPLA.models.deep_kernel.multienv_deep_adaptation import DeepMultiEnvAdapt
from KPLA.data.data_demand import generate_multi_demand_dataset
from KPLA.data.data_class import mdfaDataSetTorch

logger = logging.getLogger()



#configs


config_files = "./configs/multi_demand_config.json"


with open(config_files, encoding="utf-8") as f:
  config = json.load(f)
  data_configs = config["data"]
  n_sample = data_configs["n_sample"]


gpu_flg = True
split = True
verbose = 10

#generate data


def gen_e_source(n_sample, rng):
  return rng.choice(np.arange(1, 5), n_sample)

def gen_e_target(n_sample, rng):
  return np.ones(n_sample)*5

def gen_e2u(esamples, rng):
  lookup = {1: [0,   2.5],
            2: [2.5,   5],
            3: [5,   7.5],
            4: [7.5,  10],
            5: [2,     2]}
  
  u, indices = np.unique(esamples, return_inverse=True)
  demand = np.zeros(esamples.size)
  for i, val in enumerate(u):
    select_id = np.where(indices==i)[0]
    if val != 5:
      sample_d = rng.uniform(lookup[val][0],
                            lookup[val][1],
                            select_id.size)
    else:
      sample_d = rng.beta(lookup[val][0],
                          lookup[val][1],
                          select_id.size)*10 #rescale to [0,10]
    demand[select_id] = sample_d
  
  return demand


source_traindata = generate_multi_demand_dataset(n_sample,
                                                 gen_e2u,
                                                 gen_e_source,
                                                 seed=42)
source_traindata = mdfaDataSetTorch.from_numpy(source_traindata)


target_traindata = generate_multi_demand_dataset(n_sample,
                                                 gen_e2u,
                                                 gen_e_target,
                                                 seed=42)

target_traindata = mdfaDataSetTorch.from_numpy(target_traindata)

target_testdata = generate_multi_demand_dataset(1000,
                                                gen_e2u,
                                                gen_e_target,
                                                seed=42)
target_testdata = mdfaDataSetTorch.from_numpy(target_testdata)

print(source_traindata.X.shape)



###############
# training    #
###############
logging.info("Started")

#train source model and adapted model
dfa_model = DeepMultiEnvAdapt(config_files, gpu_flg)
dfa_model.fit(source_traindata, target_traindata, split, verbose, plot=True)

#train target model

target_model = DeepMultiEnvAdapt(config_files, gpu_flg)
target_model.fit(target_traindata, target_traindata, split, verbose, plot=False)


logging.info("FINISHED")




if  gpu_flg:
  torch.cuda.empty_cache()
  target_testdata = target_testdata.to_gpu()




target_testx = target_testdata.X
target_testy = target_testdata.Y
#source on source error
eval_list = []


#target on target error
predy = target_model.predict(target_testx)
tt_error = target_model.score(predy, target_testy).item()
eval_list.append({"task": "target-target",
                  "predict error": tt_error})


#adaptation error
predy = dfa_model.predict(target_testx)
adapt_error = dfa_model.score(predy, target_testy).item()
eval_list.append({"task": "source adapt to target",
                  "predict error": adapt_error})


df = pd.DataFrame(eval_list)
print(df)
