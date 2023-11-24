"""
Test the baseline models on LSA classification task
"""
#Author: Katherine Tsai <kt14@illinois.edu>
#MIT License


import pandas as pd
import numpy as np
import jax.numpy as jnp

from KPLA.baselines.multi_source_wcsc import MuiltiSourceCombCLF
from KPLA.baselines.multi_source_mk import  MultiSourceMK
from KPLA.baselines.multi_source_ccm import MultiSouceSimpleAdapt, MultiSourceUniform
from KPLA.baselines.multi_source_cat import MultiSourceCat
from KPLA.data.data_generator import gen_multienv_class_discrete_z



from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss


# Define Sklearn evaluation functions
def soft_accuracy(y_true, y_pred, threshold=0.5, **kwargs):
  return accuracy_score(y_true, y_pred >= threshold, **kwargs)

def log_loss64(y_true, y_pred, **kwargs):
  return log_loss(y_true, y_pred.astype(np.float64), **kwargs)




best_params_set = []

source_nsample = 4000
target_nsample = source_nsample*3
partition_dict = {"train": 0.8, "test": 0.2}
seed = 192
result = {}
i = 1
#generate source with 3 environments
p_u_0 = 0.9
p_u = [p_u_0, 1-p_u_0]


metrics = []


for seed in range(1922,1932):

  source_train_list = []
  source_test_list = []

  source_train_list_mmd = []
  source_test_list_mmd = []

  for z_env in range(0, 3):
    source_train, source_test = gen_multienv_class_discrete_z(z_env,
                                                              seed+z_env,
                                                              source_nsample,
                                                              partition_dict)
    source_train_list_mmd.append(source_train.copy())
    source_test_list_mmd.append(source_test.copy())



  target_train_list_mmd = []
  target_test_list_mmd = []

  target_train, target_test = gen_multienv_class_discrete_z(4,
                                                            seed+4,
                                                            target_nsample,
                                                            partition_dict)


  target_train_list_mmd.append(target_train.copy())
  target_test_list_mmd.append(target_test.copy())


  msu = MultiSourceUniform(len(source_train_list_mmd), max_iter=300)
  msu.fit(source_train_list_mmd)
  predictY_prob = msu.predict_proba(target_test_list_mmd[0]["X"])
  predictY = msu.predict(target_test_list_mmd[0]["X"])

  err1 = log_loss64(target_test_list_mmd[0]["Y"], predictY_prob[:,1])
  err2 = soft_accuracy(target_test_list_mmd[0]["Y"], predictY)
  err3 = roc_auc_score(target_test_list_mmd[0]["Y"],
                       predictY_prob[:,1].flatten())

  print(f"baseline MLP uniform: ll:{err1}, acc:{err2}, aucroc:{err3}")

  m = {"approach":"MLP uniform",
        "seed": seed,
        "acc": err2,
        "aucroc": err3}
  metrics.append(m)



  msc = MultiSourceCat(max_iter=300)
  msc.fit(source_train_list_mmd)
  predictY_prob = msc.predict_proba(target_test_list_mmd[0]["X"])
  predictY = msc.predict(target_test_list_mmd[0]["X"])

  err1 = log_loss64(target_test_list_mmd[0]["Y"], predictY_prob[:,1])
  err2 = soft_accuracy(target_test_list_mmd[0]["Y"], predictY)
  err3 = roc_auc_score(target_test_list_mmd[0]["Y"],
                       predictY_prob[:,1].flatten())


  print(f"baseline MLP concate: ll:{err1}, acc:{err2}, aucroc:{err3}")

  m = {"approach":"MLP cat",
        "seed": seed,
        "acc": err2,
        "aucroc": err3}
  metrics.append(m)


  df = pd.read_csv("MultisourceSA.csv")
  bandwidth = df["bandwidth"].values[0]

  msa = MultiSouceSimpleAdapt(n_env=len(source_train_list_mmd),
                                kde_kernel="gaussian",
                                bandwidth=bandwidth ,
                                max_iter=300)


  msa.fit(source_train_list_mmd)
  predictY = msa.predict(np.asarray(target_test_list_mmd[0]["X"]))
  predictY_proba = msa.predict_proba(np.asarray(target_test_list_mmd[0]["X"]))

  err1 = log_loss64(target_test_list_mmd[0]["Y"], predictY_proba[:,1])
  err2 = soft_accuracy(target_test_list_mmd[0]["Y"], predictY)
  err3 = roc_auc_score(target_test_list_mmd[0]["Y"],
                       predictY_proba[:,1].flatten())


  print(f"baseline Multisource Simple Adaptation:\
         ll:{err1}, acc:{err2}, aucroc:{err3}")

  m = {"approach":"Multisource SA",
        "seed": seed,
        "acc": err2,
        "aucroc": err3}
  metrics.append(m)

  df = pd.read_csv("MultisourceWCSC.csv")

  bandwidth = df["bandwidth"].values[0]

  rbf_ker = eval(df["kernel"].values[0])
  msmmd = MuiltiSourceCombCLF(source_train_list_mmd,
                              rbf_ker,
                              "gaussian")

  msmmd.fit(np.asarray(target_train_list_mmd[0]["X"]))

  predictY_prob = msmmd.predict(np.asarray(target_test_list_mmd[0]["X"]))
  predictY_label = np.array(jnp.argmax(predictY_prob, axis=1))


  err1 = log_loss64(target_test_list_mmd[0]["Y"], predictY_prob[:,1])
  err2 = soft_accuracy(target_test_list_mmd[0]["Y"], predictY_label)
  err3 = roc_auc_score(target_test_list_mmd[0]["Y"],
                        predictY_prob[:,1].flatten())

  print(f"baseline Multisource WCSC classifier: \
        ll:{err1}, acc:{err2}, aucroc:{err3}")

  m = {"approach":"Multisource WCSC",
        "seed": seed,
        "acc": err2,
        "aucroc": err3}
  metrics.append(m)

  df = pd.read_csv("MultisourceMK.csv")



  p_ker = eval(df["p_kernel"].values[0])
  x_ker = eval(df["x_kernel"].values[0])
  mssvm = MultiSourceMK(p_ker, x_ker)
  mssvm.fit(source_train_list_mmd, target_train_list_mmd[0])

  predictY_prob = mssvm.decision(np.asarray(target_test_list_mmd[0]["X"]))
  predictY      = mssvm.predict(np.asarray(target_test_list_mmd[0]["X"]))


  err1 = log_loss64(target_test_list_mmd[0]["Y"], predictY_prob)
  err2 = soft_accuracy(target_test_list_mmd[0]["Y"], predictY)
  err3 = roc_auc_score(target_test_list_mmd[0]["Y"],  predictY_prob)

  print(f"baseline Multisource mssvm: ll:{err1}, acc:{err2}, aucroc:{err3}")

  m = {"approach":"Multisource SVM",
        "seed": seed,
        "acc": err2,
        "aucroc": err3}
  metrics.append(m)

df = pd.DataFrame.from_records(metrics)
print(df)

df.to_csv("multienv_classification_baseline.csv")
