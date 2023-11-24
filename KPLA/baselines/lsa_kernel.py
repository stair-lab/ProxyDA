"""
LSA-spectral approach
clone from: 
https://github.com/google-research/google-research/tree/master/latent_shift_adaptation/latent_shift_adaptation
"""
#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT

import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.stats as stats
import re

from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
from cosde.base import LSEigenBase
from cosde.utils import compute_inv_eigen_system

from latent_shift_adaptation.methods.continuous_spectral_method.library import multi_true_p_u_x, multi_true_p_w_x, multi_true_p_yw_x, multi_true_p_y_ux, multi_true_p_y_x, multi_true_p_x
from latent_shift_adaptation.methods.continuous_spectral_method.utils import  compute_adaggerb_multi, least_squares, multi_least_squares_scale
from latent_shift_adaptation.methods.continuous_spectral_method.create_basis import basis_from_centers
from latent_shift_adaptation.methods.continuous_spectral_method.multi_ls_conditional_de import MultiCDEBase
from latent_shift_adaptation.methods.continuous_spectral_method.multi_ls_de import MultiDEBase
from latent_shift_adaptation.methods.continuous_spectral_method.multi_ls_marginal_de import MultiMDEBase

colors =  plt.get_cmap("tab20c")



# Extract dataframe format back to dict format
def extract_from_df(samples_df,
                    cols=["u", "x", "w", "c", "c_logits", "y",
                          "y_logits", "y_one_hot", "w_binary",
                          "w_one_hot", "u_one_hot", "x_scaled"]):
  """
  Extracts dict of numpy arrays from dataframe
  """
  result = {}
  for col in cols:
    if col in samples_df.columns:
      result[col] = samples_df[col].values
    else:
      match_str = f"^{col}_\d$"
      r = re.compile(match_str, re.IGNORECASE)
      matching_columns = list(filter(r.match, samples_df.columns))
      if len(matching_columns) == 0:
        continue
      result[col] = samples_df[matching_columns].to_numpy()
  return result

def extract_from_df_nested(samples_df,
                           cols=["u", "x", "w", "c", "c_logits", "y",
                                 "y_logits", "y_one_hot", "w_binary",
                                 "w_one_hot", "u_one_hot", "x_scaled"]):
  """
  Extracts nested dict of numpy arrays from
  dataframe with structure {domain: {partition: data}}
  """
  result = {}
  for partition in samples_df["partition"].unique():
    partition_df = samples_df.query("partition == @partition")
    result[partition] = extract_from_df(partition_df, cols=cols)
  return result


def compute_pu_x(fw_u,fw_x,x0):
  """Estimate p(U=i|x0) for i=1,...,k

  Args:
    fw_u: list of LSEigenBase objects [f(W|U=1),...,f(W|U=k)]
    fw_x: conditional_density_estimator_base
    x0: the point to be evaluated, (1, number of features)

  Returns:
    pU_x0: probability simplex

  """
  # get the estimated conditional density function
  fw_x0 = fw_x.get_density_function(x0)

  #use least-squares estimator to estimate f(U|x0)
  pU_x0 = least_squares(fw_u,fw_x0, verbose=False, reuse_gram=False)
  #print("pU_x0 before normalization: ", pU_x0)
  #make sure that the probability is non-negative
  pU_x0 = np.array([max(i,0) for i in pU_x0])
  #normalize to 1
  pU_x0 = pU_x0/sum(pU_x0)
  #print("pU_x0 after normalization: ", pU_x0)

  return pU_x0


def train_process(source_data_sample,
                  target_data_sample,
                  p_u_source,
                  p_u_target,
                  params,
                  method="kmeans",
                  test_c=np.array([1,0,0]),
                  test_y = 1,
                  evaluate=False):
  """domain adaptation via spectral method"""

  # construct linear independent basis
  basisx = [basis_from_centers(params["mu_x_u_mat"].squeeze()[i], 1) for i in range(params["k_x"])]
  basisw = basis_from_centers(params["mu_w_u_mat"].squeeze(), 1)
  basis = []
  for x,w in zip(basisx, basisw):
    basis.append(x+[w])
  ##########################
  # step 1 Estimate f(W|U) #
  ##########################

  # Estimate f(W,X|c) and f(W,X,y|c)
  c_id = np.where(np.sum(source_data_sample["c"]==test_c,axis=1)==3)[0]
  sx_c = np.array(source_data_sample["x"][c_id])
  sw_c = np.array(source_data_sample["w"][c_id])[:,np.newaxis]

  cy_id = np.where((np.sum(source_data_sample["c"]==test_c,axis=1)==3) & (source_data_sample["y"] == test_y))[0]
  sx_cy0 = np.array(source_data_sample["x"][cy_id])
  sw_cy0 = np.array(source_data_sample["w"][cy_id])[:,np.newaxis]

  # estimate the density estimator
  # f(x,w,c=test_c)
  fxw_c = MultiDEBase([sx_c[:, 0][:, np.newaxis], sx_c[:, 1][:, np.newaxis], sw_c], basis, 1e-2)
  # f(x,w,c=test_c,y=test_y)
  fxwy0_c =  MultiDEBase([sx_cy0[:, 0][:, np.newaxis], sx_cy0[:, 1][:, np.newaxis], sw_cy0], basis, 1e-2)

  # compute $\mathfrak{A}^\dagger\mathfrak{B}$

  # ensure that sigular values of fxwy0_c is not too small
  fxwy0_c_df = fxwy0_c.density_function
  id = np.argsort(fxwy0_c_df.get_params()["coeff"])[::-1][0:2]
  #id = np.arange(4)

  new_coeff = fxwy0_c_df.get_params()["coeff"][id]

  base_list = []
  for i in id:
    base_list.append(fxwy0_c_df.get_params()["base_list"][i])
  fxwy0_c_df = LSEigenBase(base_list, new_coeff)

  fxw_c_df = fxw_c.density_function
  id = np.argsort(fxw_c_df.get_params()["coeff"])[::-1][0:2]

  #id = np.arange(4)

  new_coeff = fxw_c_df.get_params()["coeff"][id]
  base_list = []
  for i in id:
    base_list.append(fxw_c_df.get_params()["base_list"][i])
  fxw_c_df = LSEigenBase(base_list, new_coeff)

  D, x_coor, y_coor = compute_adaggerb_multi(fxw_c_df, fxwy0_c_df)


  # only consider taking the top 2 components
  w, eigen_func = compute_inv_eigen_system(D, y_coor)


  # plot eigen function

  fw_u = []

  for func in eigen_func:
    # get the parameters
    param_dict = func.get_params()

    baselist = param_dict["base_list"]
    vec = []
    # normalize the function
    for b in baselist:
      l = b.get_params()["kernel"].get_params()["length_scale"]
      vec.append(np.sqrt(2*np.pi)*np.sum(b.get_params()["weight"])*l)
    l1_sum = np.sum(param_dict["coeff"] * np.array(vec))
    # rescale the eigenfunction so that the density function is sum to 1
    weight = param_dict["coeff"]/l1_sum
    fw_u.append(LSEigenBase(baselist, weight))

  for j in range(len(fw_u)):
    new_w = np.linspace(-7,7,100)
    w_0 = new_w[0]
    l1_sum = 0
    p_w = np.zeros(new_w.shape)
    for i, w in enumerate(new_w):
      p_w[i] = fw_u[j].eval(w.reshape((1,1)))

      l1_sum += np.abs(p_w[i]) * (w-w_0)
      w_0 = w

    plt.plot(new_w, p_w, color=colors((j+2)*4+0),label="fw_u %d cdf: %.2f"%(j+1, l1_sum))
  plt.legend(bbox_to_anchor=(1.1, 0.6))
  plt.plot(new_w, stats.norm.pdf(new_w, params["mu_w_u_mat"][0], 1),"-.", color="b", label="true density function 1")
  plt.plot(new_w, stats.norm.pdf(new_w, params["mu_w_u_mat"][1], 1), "-.",color="r", label="true density function 2")
  plt.title("Estimated f(W|U)")
  plt.show()

  #############################
  # step 2 Estimate q(U)/p(U) #
  #############################


  # The first step is to estimate f(w|x)
  sw = np.array(source_data_sample["w"])[:, np.newaxis]
  sx = np.array(source_data_sample["x"])
  fw_x = MultiCDEBase( sx, sw, basisx, basisw, 1e-4)


  #compute the MSE of f(w|x)
  if evaluate:
    new_w = np.linspace(-7,7,20)
    new_x = np.linspace(-3,3,20)

    cosde_pdf = np.zeros((new_x.size,new_x.size, new_w.size))
    true_pdf = np.zeros(cosde_pdf.shape)

    for i, x in enumerate(new_x):
      for k, x2 in enumerate(new_x):
        for j, w in enumerate(new_w):
          n_x = np.array([x, x2])
          fw_x0 = fw_x.get_density_function(n_x)
          cosde_pdf[i, k, j] = fw_x0.eval(w.reshape(1,1))
          true_pdf[i, k, j] = multi_true_p_w_x(w, n_x, p_u_source, params)

    print("MSE of f(W|X)", np.mean((cosde_pdf-true_pdf)**2))


  # estimate g(x) and f(x) from data

  sx = np.array(source_data_sample["x"])
  tx = np.array(target_data_sample["x"])
  # kernel density estimator
  fx = MultiMDEBase(sx, basisx ,1e-4)
  gx = MultiMDEBase(tx, basisx, 1e-4)


  if evaluate:
    new_w = np.linspace(-7,7,20)
    new_x = np.linspace(-3,3,20)

    cosde_pdf = np.zeros((new_x.size,new_x.size))
    true_pdf = np.zeros(cosde_pdf.shape)

    for i, x in enumerate(new_x):
      for j, y in enumerate(new_x):
        n_x = np.array([x,y])
        cosde_pdf[i,j] = fx.get_pdf([x.reshape(1,1), y.reshape(1,1)])
        true_pdf[i,j] = multi_true_p_x(n_x, p_u_source, params["mu_x_u_mat"].squeeze()*params["mu_x_u_coeff"])
    print("MSE of f(x)", np.mean((cosde_pdf-true_pdf)**2))


    cosde_pdf = np.zeros((new_x.size,new_x.size))
    true_pdf = np.zeros(cosde_pdf.shape)

    for i, x in enumerate(new_x):
      for j, y in enumerate(new_x):
        n_x = np.array([x,y])
        cosde_pdf[i,j] = gx.get_pdf([x.reshape(1,1), y.reshape(1,1)])
        true_pdf[i,j] = multi_true_p_x(n_x,p_u_target, params["mu_x_u_mat"].squeeze()*params["mu_x_u_coeff"])
    print("MSE of g(x)", np.mean((cosde_pdf-true_pdf)**2))
  if method == "kmeans":
    # select samples
    kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(source_data_sample["x"]).reshape(-1,2))
    centers = kmeans.cluster_centers_.squeeze()
    x0 = centers[0]
    x1 = centers[1]

    # use least-squares estimator to estimate f(U|x1)

    pU_x0 = compute_pu_x(fw_u,fw_x,x0)

    pU_x1 = compute_pu_x(fw_u,fw_x,x1)

    # construct the confusion matrix
    C = np.array([pU_x0,pU_x1])

    # solve the linear system

    # get g(x0)/f(x0)
    f_x0 = fx.get_pdf([x0[0].reshape(1,1), x0[1].reshape(1,1)])
    g_x0 = gx.get_pdf([x0[0].reshape(1,1), x0[1].reshape(1,1)])
    gx0_fx0 = g_x0/f_x0

    # get g(x1)/f(x1)
    f_x1 = fx.get_pdf([x1[0].reshape(1,1), x1[1].reshape(1,1)])
    g_x1 = gx.get_pdf([x1[0].reshape(1,1), x1[1].reshape(1,1)])
    gx1_fx1 = g_x1/f_x1

    x_ratio = np.array([gx0_fx0, gx1_fx1]).squeeze()
    qu_pu = scipy.optimize.nnls(C, x_ratio)[0]
  elif method == "random":
    # randomly sample 100 points
    np.random.seed(1)
    random_id = np.random.choice(source_data_sample["x"].shape[0], 100, replace=False)

    select_x = np.array(source_data_sample["x"])[random_id,:]
    pU_x_mat = np.zeros((select_x.size,2))
    qx_px_mat = np.zeros(select_x.size)
    for i,x in enumerate(select_x):
      pU_x_mat[i] = compute_pu_x(fw_u,fw_x,x)
      qx = gx.get_pdf([x[0].reshape(1,1), x[1].reshape(1,1)])
      px = fx.get_pdf([x[0].reshape(1,1), x[1].reshape(1,1)])
      qx_px_mat[i] = qx/(px)
    qu_pu = scipy.optimize.nnls(pU_x_mat,qx_px_mat)[0]


  if evaluate:
    print("Estimated:", qu_pu)
    print("MSE of q(U)/p(U):", np.mean((qu_pu-np.array(p_u_target)/np.array(p_u_source))**2))

    random_id = np.random.choice(source_data_sample["x"].shape[0], 100, replace=False)
    select_x = np.array(source_data_sample["x"])[random_id,:]
    pU_x_mat = np.zeros((select_x.size,2))
    pU_x_mat_true = np.zeros((select_x.size,2))
    diff = 0
    for i,x in enumerate(select_x):
      pU_x_mat[i] = compute_pu_x(fw_u,fw_x,x)
      pU_x_mat_true[i] = np.array(multi_true_p_u_x(x,p_u_source,params["mu_x_u_coeff"]*params["mu_x_u_mat"].squeeze())).squeeze()
      diff += np.mean((pU_x_mat[i]-pU_x_mat_true[i])**2)

    print("MSE of p(U|x):", diff/select_x.size)

  #############################
  # step 3 Estimate f(W|U,x)  #
  #############################


  # Learn p(y|x) via MLP
  mlp_p_y_x = MLPClassifier(random_state=0, learning_rate="adaptive", max_iter=10000).fit(np.array(source_data_sample["x"]), np.array(source_data_sample["y"]))

  if evaluate:
    mse = []
    for x in source_data_sample["x"]:
      mse.append((multi_true_p_y_x(x, p_u_source , params)-mlp_p_y_x.predict_proba(x.reshape(-1,2))[:,1])**2)
    print("MSE of mlp p(y|x):",np.mean(np.array(mse)))
  # Estimate f(W|x, y=0)

  y0_id = np.where(source_data_sample["y"]==0)[0]
  sx_y0 = np.array(source_data_sample["x"][y0_id])
  sw_y0 = np.array(source_data_sample["w"][y0_id])[:, np.newaxis]

  fw_y0x = MultiCDEBase(sx_y0, sw_y0, basisx, basisw, 1e-4)

  # Estimate f(W|x, y=1)
  y1_id = np.where(source_data_sample["y"]==1)[0]
  sx_y1 = np.array(source_data_sample["x"][y1_id])
  sw_y1 = np.array(source_data_sample["w"][y1_id])[:, np.newaxis]

  fw_y1x = MultiCDEBase(sx_y1, sw_y1, basisx, basisw, 1e-4)

  if evaluate:

    new_w = np.linspace(-7,7,20)
    new_x = np.linspace(-4,4,20)

    fwy0_x0_pdf = np.zeros((new_x.size,new_x.size, new_w.size))
    fwy1_x0_pdf = np.zeros((new_x.size,new_x.size, new_w.size))

    true_fwy0_x0_pdf = np.zeros(fwy0_x0_pdf.shape)
    true_fwy1_x0_pdf = np.zeros(fwy1_x0_pdf.shape)

    for i, x1 in enumerate(new_x):
      for j, x2 in enumerate(new_x):
        for k, w in enumerate(new_w):
          n_x = np.array([x1, x2])
          fw_y0x0 = fw_y0x.get_density_function(n_x)
          fw_y1x0 = fw_y1x.get_density_function(n_x)


          fwy1_x0_pdf[i, j, k] = fw_y1x0.eval(w.reshape(1,1))*mlp_p_y_x.predict_proba(n_x.reshape(1,2))[0,1]
          true_fwy1_x0_pdf[i, j, k] =  multi_true_p_yw_x(1,w,n_x, p_u_source,params)



          fwy0_x0_pdf[i, j, k] = fw_y0x0.eval(w.reshape(1,1))*mlp_p_y_x.predict_proba(n_x.reshape(1,2))[0,0]
          true_fwy0_x0_pdf[i, j, k] = multi_true_p_yw_x(0,w,n_x, p_u_source,params)


    print("MSE of p(Y=1,w|x):", np.mean((fwy1_x0_pdf-true_fwy1_x0_pdf)**2))
    print("MSE of p(Y=0,w|x):", np.mean((fwy0_x0_pdf-true_fwy0_x0_pdf)**2))

  results = {
      "qu_pu": qu_pu,
      "fw_y0x": fw_y0x,
      "fw_y1x": fw_y1x,
      "py_x": mlp_p_y_x,
      "fw_u": fw_u,
      "fw_x": fw_x
  }
  return results

def inference(dataset,qu_pu, fw_y0x, fw_y1x, p_y_x, fw_u, fw_x, true_p_u, params):
  source_feature = np.array(dataset["x"])
  source_label = np.array(dataset["y"])

  acc = 0
  source_predict_score = []
  source_predict_label = []
  source_mse = []
  error = 0
  for x,y in zip(source_feature, source_label):
    #predicting the probability that q(Y=0|x)

    qy_x, yxu_err = predict(qu_pu, fw_y0x, fw_y1x, p_y_x, fw_u, fw_x, x, params, "standard")
    qy_x = qy_x[1]
    true_qy_x = multi_true_p_y_x(x, true_p_u, params)
    source_mse.append((true_qy_x - qy_x)**2)
    if(qy_x>=0.5):
      hat_label = 1
    else:
      hat_label = 0
    source_predict_score.append(qy_x)
    source_predict_label.append(hat_label)
    error += yxu_err
  print("MSE of f(y|x, u):",error/source_label.shape[0])
  results = {
    "acc": accuracy_score(source_label, np.array(source_predict_label)),
    "aucroc": roc_auc_score(source_label, source_predict_score),
    "log-loss": log_loss(source_label, source_predict_score),
    "mse": np.mean(source_mse),
    "brier": brier_score_loss(source_label, source_predict_score)
  }
  return results


def predict(qu_pu,fw_y0x,fw_y1x, py_x, fw_u, fw_x, x0, params, normalize="standard"):
  """Given fixed y, estimate f(y|x0, U=i) for i=1,...,k

  Args:
    qu_pu: density ratio,ndarray
    fw_y0x: f(W|Y=0, X)
    fw_y1x: f(W|Y=1, X)
    py_x: p(Y|x)
    fw_u: list of LSEigenBase objects, [f(W|U=1),...,f(W|U=k)]
    fw_x: f(W|X)
    x0: the point to be evaluated, (1, number of features)
  Returns:
    fy_x0u: probability simplex

  """
  #estimate p(U|x0)

  # get the estimated conditional density function
  # fw_x0 = fw_x.get_density_function(x0)

  #use least-squares estimator to estimate f(U|x0)

  pU_x0 = compute_pu_x(fw_u,fw_x,x0)

  #estimate f(w|u)p(u|x0)
  fwu_x0 = []
  for p, f in zip(pU_x0, fw_u):
    id_cut = np.where(np.abs(f.get_params()["coeff"])<1e-10)[0]
    new_coeff = f.get_params()["coeff"]*p
    new_coeff[id_cut] = 0.
    #truncate bases such that the coeff is too small
    fwu_x0.append(LSEigenBase(f.baselist, new_coeff))

  #fwy0_x evaluated at x0

  fw_y0x0 = fw_y0x.get_density_function(x0)
  fw_y1x0 = fw_y1x.get_density_function(x0)


  fw_y0x0_coeff = fw_y0x0.get_params()["coeff"]
  fwy0_x0 = LSEigenBase(fw_y0x0.get_params()["base_list"], fw_y0x0_coeff * py_x.predict_proba(x0.reshape(1,params["k_x"]))[0,0])

  fw_y1x0_coeff = fw_y1x0.get_params()["coeff"]
  fwy1_x0 = LSEigenBase(fw_y1x0.get_params()["base_list"], fw_y1x0_coeff * py_x.predict_proba(x0.reshape(1,params["k_x"]))[0,1])


  solution = multi_least_squares_scale(fwu_x0, fwy0_x0, fwy1_x0, pU_x0, reuse_gram = False)
  fy0_x0u = solution[0:2]
  fy1_x0u = solution[2::]
  sum_u0 = fy0_x0u[0]+fy1_x0u[0]
  sum_u1 = fy0_x0u[1]+fy1_x0u[1]
  sum_u = np.array([1./sum_u0, 1./sum_u1])
  fy0_x0u = fy0_x0u*sum_u
  fy1_x0u = fy1_x0u*sum_u

  true_fy0_x0u = np.array([1-multi_true_p_y_ux(x0, 0, params), 1-multi_true_p_y_ux(x0, 1, params)]).squeeze()

  true_fy1_x0u = np.array([multi_true_p_y_ux(x0, 0, params), multi_true_p_y_ux(x0, 1, params)]).squeeze()
  mse_fy_xu = np.mean(np.array([fy0_x0u-true_fy0_x0u, fy1_x0u-true_fy1_x0u])**2)
  qy0_x0 = max(0., sum(fy0_x0u*qu_pu*pU_x0))
  qy1_x0 = max(0., sum(fy1_x0u*qu_pu*pU_x0))
  out_prob = np.array([qy0_x0, qy1_x0])
  if normalize=="standard":
    out_prob /= np.sum(out_prob)
  else:
    out_prob = scipy.special.softmax(out_prob)
  return out_prob, mse_fy_xu



def inference_with_qux(dataset, fw_y0x, fw_y1x, p_y_x, fw_u, fw_x, gw_x, true_p_u, params):
  source_feature = np.array(dataset["x"])
  source_label = np.array(dataset["y"])


  source_predict_score = []
  source_predict_label = []
  source_mse = []
  error = 0
  for x, _ in zip(source_feature, source_label):
    #predicting the probability that q(Y=0|x)

    qy_x, yxu_err = predict_with_qux(fw_y0x, fw_y1x, p_y_x, fw_u, fw_x, gw_x, x, params, "standard")

    qy_x = qy_x[1]
    true_qy_x = multi_true_p_y_x(x, true_p_u, params)
    source_mse.append((true_qy_x - qy_x)**2)
    if(qy_x>=0.5):
      hat_label = 1
    else:
      hat_label = 0
    source_predict_score.append(qy_x)
    source_predict_label.append(hat_label)
    error += yxu_err
  print("MSE of f(y|x, u):",error/source_label.shape[0])
  results = {
    "acc": accuracy_score(source_label, np.array(source_predict_label)),
    "aucroc": roc_auc_score(source_label, source_predict_score),
    "log-loss": log_loss(source_label, source_predict_score),
    "mse": np.mean(source_mse),
    "brier": brier_score_loss(source_label, source_predict_score)
  }
  return results


def predict_with_qux(fw_y0x,fw_y1x, py_x, fw_u, fw_x, gw_x, x0, params, normalize="standard"):
  """Given fixed y, estimate f(y|x0, U=i) for i=1,...,k

  Args:
    qu_pu: density ratio,ndarray
    fw_y0x: f(W|Y=0, X)
    fw_y1x: f(W|Y=1, X)
    py_x: p(Y|x)
    fw_u: list of LSEigenBase objects, [f(W|U=1),...,f(W|U=k)]
    fw_x: f(W|X)
    gw_x: g(W|X)
    x0: the point to be evaluated, (1, number of features)
  Returns:
    fy_x0u: probability simplex

  """
  #estimate p(U|x0)

  # get the estimated conditional density function
  #fw_x0 = fw_x.get_density_function(x0)

  #use least-squares estimator to estimate f(U|x0)

  pU_x0 = compute_pu_x(fw_u, fw_x, x0)

  qU_x0 = compute_pu_x(fw_u, gw_x, x0)

  #estimate f(w|u)p(u|x0)
  fwu_x0 = []
  for p, f in zip(pU_x0, fw_u):
    id_cut = np.where(np.abs(f.get_params()["coeff"])<1e-10)[0]
    new_coeff = f.get_params()["coeff"]*p
    new_coeff[id_cut] = 0.
    #truncate bases such that the coeff is too small
    fwu_x0.append(LSEigenBase(f.baselist, new_coeff))

  #fwy0_x evaluated at x0

  fw_y0x0 = fw_y0x.get_density_function(x0)
  fw_y1x0 = fw_y1x.get_density_function(x0)


  fw_y0x0_coeff = fw_y0x0.get_params()["coeff"]
  fwy0_x0 = LSEigenBase(fw_y0x0.get_params()["base_list"], fw_y0x0_coeff * py_x.predict_proba(x0.reshape(1,params["k_x"]))[0,0])

  fw_y1x0_coeff = fw_y1x0.get_params()["coeff"]
  fwy1_x0 = LSEigenBase(fw_y1x0.get_params()["base_list"], fw_y1x0_coeff * py_x.predict_proba(x0.reshape(1,params["k_x"]))[0,1])


  solution = multi_least_squares_scale(fwu_x0, fwy0_x0, fwy1_x0, pU_x0, reuse_gram = False)
  fy0_x0u = solution[0:2]
  fy1_x0u = solution[2::]
  sum_u0 = fy0_x0u[0]+fy1_x0u[0]
  sum_u1 = fy0_x0u[1]+fy1_x0u[1]
  sum_u = np.array([1./sum_u0, 1./sum_u1])
  fy0_x0u = fy0_x0u*sum_u
  fy1_x0u = fy1_x0u*sum_u

  true_fy0_x0u = np.array([1-multi_true_p_y_ux(x0, 0, params), 1-multi_true_p_y_ux(x0, 1, params)]).squeeze()

  true_fy1_x0u = np.array([multi_true_p_y_ux(x0, 0, params), multi_true_p_y_ux(x0, 1, params)]).squeeze()
  mse_fy_xu = np.mean(np.array([fy0_x0u-true_fy0_x0u, fy1_x0u-true_fy1_x0u])**2)

  qy0_x0 = max(0., sum(fy0_x0u*qU_x0))
  qy1_x0 = max(0., sum(fy1_x0u*qU_x0))
  out_prob = np.array([qy0_x0, qy1_x0])
  if normalize=="standard":
    out_prob /= np.sum(out_prob)
  else:
    out_prob = scipy.special.softmax(out_prob)
  return out_prob, mse_fy_xu
