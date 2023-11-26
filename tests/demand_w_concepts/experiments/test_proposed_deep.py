"""
 A demo of deep feature net adaptation (full adaptation)
"""
#Author:Katherine Tsai<kt14@illinois.edu>
#MIT LISENCE

import torch
import logging

import pandas as pd
import json

from KPLA.models.deep_kernel.deep_adaptation import DeepFullAdapt
from KPLA.data.data_demand import generate_demand_dataset
from KPLA.data.data_class import dfaDataSetTorch

logger = logging.getLogger()



#configs


config_files = "./configs/demand_config.json"


with open(config_files, encoding="utf-8") as f:
  config = json.load(f)
  data_configs = config["data"]
  n_sample = data_configs["n_sample"]


gpu_flg = True
split = True
verbose = 10

#generate data
def gen_u_source(n_sample: int, rng):
  demand = rng.uniform(0, 10, n_sample)     #U
  return demand

def gen_u_target(n_sample: int, rng):
  demand = rng.uniform(0, 10, n_sample)     #U
  return demand



source_traindata = generate_demand_dataset(gen_u_source, n_sample, seed=42)
source_traindata = dfaDataSetTorch.from_numpy(source_traindata)

source_testdata = generate_demand_dataset(gen_u_source, 1000, seed=1)
source_testdata = dfaDataSetTorch.from_numpy(source_testdata )

target_traindata = generate_demand_dataset(gen_u_target, n_sample, seed=42)
target_traindata = dfaDataSetTorch.from_numpy(target_traindata)

target_testdata = generate_demand_dataset(gen_u_target, 1000, seed=1)
target_testdata = dfaDataSetTorch.from_numpy(target_testdata)




###############
# training    #
###############
logging.info("Started")

#train source model and adapted model
dfa_model = DeepFullAdapt(config_files, gpu_flg)
dfa_model.fit(source_traindata, target_traindata, split, verbose)

#train target model

target_model = DeepFullAdapt(config_files, gpu_flg)
target_model.fit(target_traindata, source_traindata, split, verbose)


logging.info("FINISHED")




if  gpu_flg:
  torch.cuda.empty_cache()
  source_testdata = source_testdata.to_gpu()
  target_testdata = target_testdata.to_gpu()

#dfa_task.evaluation(source_testdata, target_testdata)

source_testx = source_testdata.X
source_testy = source_testdata.Y

target_testx = target_testdata.X
target_testy = target_testdata.Y
#source on source error
eval_list = []

predy = dfa_model.predict(source_testx, "original")
ss_error = dfa_model.score(predy, source_testy).item()
eval_list.append({"task": "source-source", "predict error": ss_error})


#target on source error
predy = target_model.predict(source_testx, "original")
ts_error = target_model.score(predy, source_testy).item()
eval_list.append({"task": "target-source",
                  "predict error": ts_error})

#target on target error
predy = target_model.predict(target_testx, "original")
tt_error = target_model.score(predy, target_testy).item()
eval_list.append({"task": "target-target",
                  "predict error": tt_error})

#source on target error
predy = dfa_model.predict(target_testx, "original")
st_error = dfa_model.score(predy, target_testy).item()
eval_list.append({"task": "source-target",
                  "predict error": st_error})

#adaptation error
predy = dfa_model.predict(target_testx, "adapt")
adapt_error = dfa_model.score(predy, target_testy).item()
eval_list.append({"task": "source adapt to target",
                  "predict error": adapt_error})


#adaptation error
predy = target_model.predict(source_testx, "adapt")
adapt_error = target_model.score(predy, source_testy).item()
eval_list.append({"task": "target adapt to source",
                  "predict error": adapt_error})



df = pd.DataFrame(eval_list)
print(df)
