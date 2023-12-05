"""utility functions for kernel methods"""
#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT


import numpy as np
import functools
from typing import Callable

import operator
import collections

from sklearn.preprocessing import StandardScaler

import jax
import jax.numpy as jnp
from jax import vmap
import jax.scipy.linalg as jsla
import jax.numpy.linalg as jnla



# utility functions
# clone from:
# https://github.com/yuchen-zhu/kernel_proxies/blob/main/KPV/utils.py

@jax.jit
def modist(v):
  return jnp.median(v)

@jax.jit
def sum_jit(a,b):
  return jnp.sum(a,b)

@jax.jit
def linear_kern(x, y):
  return jnp.sum(x * y)

@jax.jit
def l2_dist(x,y):
  return jnp.array((x - y)**2)

@jax.jit
def binary_dist(x, y):
  return x*y + (1-x)*(1-y)

#@functools.partial(jax.jit, static_argnums=(0,1))
def identifier(x,y):
  if x!=y:
    b=0
  else:
    b=1
  return b

@functools.partial(jax.jit, static_argnums=0)
def dist_func(func1: Callable, x,y):
  return jax.vmap(lambda x1: jax.vmap(lambda y1: func1( x1, y1))(y))(x)

@jax.jit
def fill_diagonal(a, val):
  assert a.ndim >= 2
  i, j = jnp.diag_indices(min(a.shape[-2:]))
  return a.at[..., i, j].set(val)

@jax.jit
def rbf_ker(x,y,scale=1):
  dist_mat=dist_func(l2_dist, x, y)

  gamma=modist(jnp.sqrt(dist_mat))
  coef=1/(2*gamma**2)
  # coef = 1/x.shape[1] if len(x.shape) > 1 else 1
  coef *= scale
  coef *= 0.5
  ker = jnp.exp(-coef*dist_mat)

  return ker

#@jax.jit
def rbf_ker_equal(x,y,scale=1):
  ker = rbf_ker(x,y, scale)
  ker = fill_diagonal(ker, 1.)
  return ker

@jax.jit
def binary_ker(x,y):
  dist_mat = dist_func(binary_dist, x, y)
  return dist_mat

@jax.jit
def binary_column_ker(x,y):
  def fn(u, v):
    return binary_ker(u, v)
  v = vmap(fn, (1,1))
  dist_mat_stack = v(x,y)
  return jnp.prod(dist_mat_stack, axis=0)

@jax.jit
def rbf_column_ker(x, y, scale):

  def fn(u, v):
    return rbf_ker(u,v, scale)
  v = vmap(fn, (1,1))
  dist_mat_stack = v(x,y)
  return jnp.prod(dist_mat_stack, axis=0)

@jax.jit
def rbf_column_ker_equal(x, y, scale):

  def fn(u, v):
    return rbf_ker_equal(u,v, scale)
  v = vmap(fn, (1,1))
  dist_mat_stack = v(x,y)
  return jnp.prod(dist_mat_stack, axis=0)

@jax.jit
def mat_trans(a):
  return jnp.transpose(a)

@jax.jit
def identifier_ker(x,y):
  return dist_func(identifier,x,y)

@jax.jit
def hadamard_prod(a, b):
  return a*b

@jax.jit
def jsla_inv(a):
  return jsla.inv(a)

@jax.jit
def jnla_norm(a):
  return jnla.norm(a)

@jax.jit
def kron_prod(a,b):
  return jnp.kron(a,b)

@jax.jit
def modif_kron(x,y):
  if y.shape[1]!=x.shape[1]:
    print("Column_number error")
  else:
    #fn = vmap(lambda u, v: jnp.kron(u, v), (1, 1))
    return jnp.array([jnp.kron(x[:,i], y[:,i]).T for i in range(y.shape[1])])

@jax.jit
def mat_mul(a,b):
  return jnp.matmul(a,b)

@jax.jit
def jsla_solve(a,b):
  return jax.sp.linalg.solve(a, b, assume_a = "pos")


@jax.jit
def katri_rao_col(a,b):
  def fn(x,y):
    return kron_prod(x,y)
  v = vmap(fn, (1,1),1)
  return v(a,b)


def integral_rbf_ker(x,y, ori_scale):
  """
  compute new gram matrix such that each entry is 
  tilde{K}(x,y)=int K(z,x)K(z,y)dz, where K is the 
  original kernel function.
  """
  dist_mat=dist_func(l2_dist,x,y)
  gamma=modist(jnp.sqrt(dist_mat))
  new_l = ori_scale*2
  new_gram = rbf_ker(x,y,new_l)*jnp.sqrt(jnp.pi*ori_scale)*gamma
  return new_gram

def ker_mat(x1,x2, kernel="rbf", scale=1.):
  """
  compute the K_xx
  Args:
  X1: shape: (n1_samples, n1_features)
  X2: shape: (n2_samples, n2_features)
  kernel: kernel method, default: 'rbf', str
  scale: kernel_length scale
  """
  def compute_gram(x,y,kernel):
    if kernel == "rbf":
      equal= jnp.array_equal(x,y)
      #print('two array is equal:', equal)
      if equal:
        temp = rbf_ker_equal(jnp.array(x), jnp.array(y), scale)
      else:
        temp = rbf_ker(jnp.array(x), jnp.array(y), scale)

    if kernel == "binary":
      temp = binary_ker(jnp.array(x), jnp.array(y))

    if kernel == "binary_column":
      temp = binary_column_ker(jnp.array(x), jnp.array(y))

    if kernel == "rbf_column":
      equal= jnp.array_equal(x,y)
      #print('two array is equal:', equal)
      if equal:
        temp = rbf_column_ker_equal(jnp.array(x), jnp.array(y), scale)
      else:
        temp = rbf_column_ker(jnp.array(x), jnp.array(y), scale)

    return temp

  if isinstance(kernel, list):
    s_id = 0
    ker_x1x2 = jnp.ones((x1.shape[0], x2.shape[0]))
    for k_func in kernel:
      e_id = s_id + k_func["dim"]
      temp = jnp.squeeze(compute_gram(jnp.array(x1[:,s_id:e_id]),
                                      jnp.array(x2[:,s_id:e_id]),
                                      k_func["kernel"]))
      #print(type(temp))
      #print(temp.shape)
      ker_x1x2 = mat_mul(ker_x1x2, temp)
      #jnp.prod(K_x1x2,  jnp.squeeze(temp))
      s_id  += k_func["dim"]
  else:
    ker_x1x2 = compute_gram(x1,x2, kernel)

  if len(ker_x1x2.shape) == 3:
    #perform Hadmard product
    ker_x1x2 = jnp.prod(ker_x1x2, axis=2)

  return ker_x1x2


def stage2_weights(gamma_w, sigma_inv):
  n_row = gamma_w.shape[0]
  arr = [mat_mul(jnp.diag(gamma_w[i, :]), sigma_inv) for i in range(n_row)]
  return jnp.concatenate(arr, axis=0)


def standardise(x):
  scaler = StandardScaler()
  if x.ndim == 1:
    x_scaled = scaler.fit_transform(x.reshape(-1,1)).squeeze()
    return x_scaled, scaler
  else:
    x_scaled = scaler.fit_transform(x).squeeze()
    return x_scaled, scaler


def truncate_sqrtinv(x, thre=1e-5):
  """ truncate sqaured-root inverse
  """
  u,vh = jnp.linalg.eigh(x)
  #if jnp.isnan(u).any():
  #    print("encounter invalid eigenvalue")
  #if jnp.isnan(vh).any():
  #    print("encounter invalid eigenvector")
  select_id = jnp.where(u>thre)[0]

  new_u = u[select_id]
  new_vh = vh[:, select_id]

  inv_sqrt = mat_mul(new_vh/np.sqrt(new_u),new_vh.T)
  return inv_sqrt


def truncate_inv(x, thre=1e-5):
  """truncate inverse
  """
  u,vh = jnp.linalg.eigh(x)
  #if jnp.isnan(u).any():
  #    print("encounter invalid eigenvalue")
  #if jnp.isnan(vh).any():
  #    print("encounter invalid eigenvector")
  select_id = np.where(u>thre)[0]

  new_u = u[select_id]
  new_vh = vh[:, select_id]

  inv_sqrt = mat_mul(new_vh/(new_u),new_vh.T)
  return inv_sqrt


def truncate_sqrt(x, thre=1e-5):
  """ truncate sqaured-root
  """
  u,vh = jnp.linalg.eigh(x)
  select_id = jnp.where(u>thre)[0]

  new_u = u[select_id]
  new_vh = vh[:, select_id]
  inv_sqrt = mat_mul(new_vh*np.sqrt(new_u), new_vh.T)
  return inv_sqrt

def woodbury_identity(q_mat, lam, n):
  """ compute the inverse (lam*n*I+QQ^T) using woodbury lemma
  """
  q = q_mat.shape[1]
  inv_temp = jsla.solve(lam*n*jnp.eye(q)+mat_mul(q_mat.T, q_mat), jnp.eye(q))
  if jnp.isnan(inv_temp).any():
    print("inv_temp is nan")
  aprox_k = (jnp.eye(n)-mat_mul(mat_mul(q_mat,inv_temp), q_mat.T))/(lam*n)
  return aprox_k


#@jax.jit
def cal_loocv_emb(k_mat, kernel_y, lam):
  nd = k_mat.shape[0]
  i_mat = jnp.eye(nd)
  if nd <= 1000:
    #use linear solver
    q_mat = jsla.solve(k_mat + lam * nd * i_mat, i_mat)
  else:
    #Nystrom approximation
    q = 250
    select_x = np.random.choice(nd, q, replace=False)
    ker_q  = k_mat[select_x, :][:, select_x]
    ker_nq = k_mat[:, select_x]
    #if jnp.isnan(K_q).any():
    #    print('K_q is nan')
    inv_kerq_sqrt =  jnp.array(truncate_sqrtinv(ker_q))
    temp_q = mat_mul(ker_nq, inv_kerq_sqrt)
    q_mat = woodbury_identity(temp_q, lam, nd)

  h_mat = i_mat - mat_mul(k_mat, q_mat)
  h_inv = jnp.diag(1.0 / jnp.diag(h_mat))
  return jnp.trace(h_inv @ h_mat @ kernel_y @ h_mat @ h_inv)


def cal_l_w (k_mat, kernel_y, low=-4, high=0, n=10):

  lam_values = np.logspace(low, high, n)
  grid_search={}
  for lam in lam_values:
    grid_search[lam]=cal_loocv_emb(k_mat, kernel_y, lam)
  l,loo=min(grid_search.items(), key=operator.itemgetter(1))

  return l,loo

#@jax.jit
def cal_loocv_alpha(k_mat, sigma, gamma, y, lam):
  nd = k_mat.shape[0]
  i_mat = jnp.eye(nd)
  temp = jsla.solve(sigma + lam * nd* i_mat, i_mat)

  h_mat = i_mat - mat_mul(mat_mul(k_mat, gamma), temp)
  h_inv = jnp.diag(1.0 / jnp.diag(h_mat))

  return jnp.linalg.norm(mat_mul(h_inv, mat_mul(h_mat, y)))

def cal_l_yw(k_mat, sigma, gamma, y, low=-4, high=0, n=10):

  lam_values = np.logspace(low, high, num=n)
  grid_search={}
  for lam in lam_values:
    grid_search[lam]=cal_loocv_alpha(k_mat, sigma, gamma, y, lam)
  l,loo=min(grid_search.items(), key=operator.itemgetter(1))

  return l,loo

#@jax.jit
def cal_loocv_m0(d, dc, m, ker_cc, lam):
  nd = d.shape[0]
  i_mat = jnp.eye(nd)
  if nd<=1000:
    sigma = dc + lam*nd*m
    m1 = m.sum(axis=0)
    alpha = jsla.solve(sigma, m1)
  else:
    q = min(500, nd)
    select_x = np.random.choice(nd, q, replace=False)
    ker_q = m[select_x, :][:, select_x]
    ker_nq = m[:, select_x]

    inv_kerq_sqrt = truncate_sqrtinv(ker_q)
    q_mat = ker_nq.dot(inv_kerq_sqrt)

    aprox_m = q_mat.dot(q_mat.T)

    # nystrom M^{-1/2}GM^{-1/2}
    inv_m_sqrt = jnp.array(truncate_sqrtinv(aprox_m))

    m_sqrt = jnp.array(truncate_sqrt(aprox_m))

    mgm = inv_m_sqrt.dot(dc.dot(inv_m_sqrt))

    q = min(1000, nd)

    select_x2 = np.random.choice(nd, q, replace=False)
    ker_q2 = mgm[select_x2, :][:, select_x2]
    ker_nq2 = mgm[:, select_x2]

    inv_ker_q2_sqrt = truncate_sqrtinv(ker_q2)
    q2_mat = ker_nq2.dot(inv_ker_q2_sqrt)

    aprox_inv = woodbury_identity(q2_mat, lam, nd)

    temp_alpha = inv_m_sqrt.dot(aprox_inv.dot(m_sqrt.sum(axis=1)))
    alpha = temp_alpha/(lam*nd)

  h_mat = i_mat - mat_mul(d, jnp.diag(alpha))
  h_mat_inv = jnp.diag(1.0 / jnp.diag(h_mat))
  return jnp.trace(h_mat_inv @ h_mat @ ker_cc @ h_mat @ h_mat_inv)


def cal_l_m(d, dc, m, ker_cc, low=-4, high=0, n=10):
  lam_values = np.logspace(low, high, num=n)
  grid_search={}
  for lam in lam_values:
    grid_search[lam] = cal_loocv_m0(d, dc, m, ker_cc, lam)
  l,loo=min(grid_search.items(), key=operator.itemgetter(1))

  return l,loo


def flatten(nested_dict, seperator=".", name=None):
  flatten_dict = {}
  if (not nested_dict) and (nested_dict != 0):
    return flatten_dict

  if isinstance(nested_dict,
                collections.abc.MutableMapping,):
    for key, value in nested_dict.items():
      if name is not None:
        flatten_dict.update(
            flatten(
                nested_dict=value,
                seperator=seperator,
                name=f"{name}{seperator}{key}",
            ),
        )
      else:
        flatten_dict.update(
            flatten(
                nested_dict=value,
                seperator=seperator,
                name=key,
            ),
        )
  else:
    flatten_dict[name] = nested_dict

  return flatten_dict