import os,sys
import time
import numpy as np
import pandas as pd
import functools
from typing import Callable
import jax.scipy.linalg as jsla
import jax.scipy.sparse.linalg as jsla_sparse
import jax.numpy.linalg as jnla
import operator
from typing import Dict, Any, Iterator, Tuple
import collections

import scipy as sp
import scipy.sparse as sps
import scipy.linalg as la
from numpy.linalg import matrix_rank


from sklearn.kernel_approximation import (RBFSampler,Nystroem)
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, jit, vmap
from jax import random


# utility functions
# clone from: https://github.com/yuchen-zhu/kernel_proxies/blob/main/KPV/utils.py


@jax.jit
def modist(v):
    return jnp.median(v)

@jax.jit
def sum_jit(A,B):
    return jnp.sum(A,B)

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
    if (x!=y):
        b=0
    else:
        b=1
    return b


@functools.partial(jax.jit, static_argnums=(0))
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

    #gamma=modist(jnp.sqrt(dist_mat))
    gamma=1
    #coef=1/(2*gamma**2)
    coef = 1/x.shape[1] if len(x.shape) > 1 else 1
    coef *= scale 
    coef *= 0.5
    K = jnp.exp(-coef*dist_mat)

    return K

#@jax.jit
def rbf_ker_equal(x,y,scale=1):

    K = rbf_ker(x,y, scale)
    K=fill_diagonal(K, 1.)
    
    return K


@jax.jit
def binary_ker(x,y):
    dist_mat = dist_func(binary_dist, x, y)
    return dist_mat

@jax.jit
def binary_column_ker(x,y):
    fn = lambda u,v: binary_ker(u, v)
    v = vmap(fn, (1,1))
    dist_mat_stack = v(x,y)
    return jnp.prod(dist_mat_stack, axis=0)

@jax.jit
def rbf_column_ker(x, y, scale):
    fn = lambda u,v: rbf_ker(u,v, scale)
    v = vmap(fn, (1,1))
    dist_mat_stack = v(x,y)
    return jnp.prod(dist_mat_stack, axis=0)

@jax.jit
def rbf_column_ker_equal(x, y, scale):
    fn = lambda u,v: rbf_ker_equal(u,v, scale)
    v = vmap(fn, (1,1))
    dist_mat_stack = v(x,y)
    return jnp.prod(dist_mat_stack, axis=0)



		
@jax.jit	
def mat_trans(A):	
    return jnp.transpose(A)
    
@jax.jit
def identifier_ker(x,y):
    return dist_func(identifier,x,y)


@jax.jit
def Hadamard_prod(A,B):
    return A*B


@jax.jit
def jsla_inv(A):
    return jsla.inv(A)

@jax.jit
def jnla_norm(A):
  return jnla.norm(A)

@jax.jit
def kron_prod(a,b):
    return jnp.kron(a,b)


@jax.jit
def modif_kron(x,y):
    if (y.shape[1]!=x.shape[1]):
        print("Column_number error")
    else:
        
        #fn = vmap(lambda u, v: jnp.kron(u, v), (1, 1))
        return jnp.array([jnp.kron(x[:,i], y[:,i]).T for i in range(y.shape[1])])

@jax.jit
def mat_mul(A,B):
    return jnp.matmul(A,B)

@jax.jit
def jsla_solve(A,B):
    return jax.sp.linalg.solve(A, B, assume_a = 'pos')


@jax.jit
def katri_rao_col(a,b):
    fn = lambda x,y: kron_prod(x,y)
    v = vmap(fn, (1,1),1)
    return v(a,b)


def integral_rbf_ker(x,y, ori_scale):
    """
    compute new gram matrix such that each entry is \tilde{K}(x,y)=\int K(z,x)K(z,y)dz, where K is the original kernel function

    """
    dist_mat=dist_func(l2_dist,x,y)
    gamma=modist(jnp.sqrt(dist_mat))
    new_l = ori_scale*2
    new_gram = rbf_ker(x,y,new_l)*jnp.sqrt(jnp.pi*ori_scale)*gamma
    return new_gram

def ker_mat(X1,X2, kernel='rbf', scale=1.):
    """
    compute the K_xx
    Args:
    X1: shape: (n1_samples, n1_features)
    X2: shape: (n2_samples, n2_features)
    kernel: kernel method, default: 'rbf', str
    scale: kernel_length scale
    """
    def compute_gram(x,y,kernel):
        if kernel == 'rbf':
            equal= jnp.array_equal(x,y)
            #print('two array is equal:', equal)
            if equal:
                temp = rbf_ker_equal(jnp.array(x), jnp.array(y), scale)
            else:
                temp = rbf_ker(jnp.array(x), jnp.array(y), scale)
        if kernel == 'binary':
            temp = binary_ker(jnp.array(x), jnp.array(y))
        if kernel == 'binary_column':
            temp = binary_column_ker(jnp.array(x), jnp.array(y))
        if kernel == 'rbf_column':
            equal= jnp.array_equal(x,y)
            #print('two array is equal:', equal)
            if equal:
                temp = rbf_column_ker_equal(jnp.array(x), jnp.array(y), scale)
            else:
                temp = rbf_column_ker(jnp.array(x), jnp.array(y), scale)
        return temp

    if isinstance(kernel, list):
        s_id = 0
        K_x1x2 = jnp.ones((X1.shape[0], X2.shape[0]))
        for k_func in kernel:
            e_id = s_id + k_func['dim']
            temp = jnp.squeeze(compute_gram(jnp.array(X1[:,s_id:e_id]), jnp.array(X2[:,s_id:e_id]), k_func['kernel']))
            #print(type(temp))
            #print(temp.shape)
            K_x1x2 = mat_mul(K_x1x2, temp)
            #jnp.prod(K_x1x2,  jnp.squeeze(temp))
            s_id  += k_func['dim']


    else:
        K_x1x2 = compute_gram(X1,X2, kernel)

    if len(K_x1x2.shape) == 3:
        #perform Hadmard product
        K_x1x2 = jnp.prod(K_x1x2, axis=2)
   
    return K_x1x2


def stage2_weights(Gamma_w, Sigma_inv):
    n_row = Gamma_w.shape[0]
    arr = [mat_mul(jnp.diag(Gamma_w[i, :]), Sigma_inv) for i in range(n_row)]
    return jnp.concatenate(arr, axis=0)


def standardise(X):
    scaler = StandardScaler()
    if X.ndim == 1:
        X_scaled = scaler.fit_transform(X.reshape(-1,1)).squeeze()
        return X_scaled, scaler
    else:
        X_scaled = scaler.fit_transform(X).squeeze()
        return X_scaled, scaler


def truncate_sqrtinv(X, thre=1e-5):
    """
    """
    u,vh = jnp.linalg.eigh(X)
    #if jnp.isnan(u).any():
    #    print("encounter invalid eigenvalue")
    #if jnp.isnan(vh).any():
    #    print("encounter invalid eigenvector")
    select_id = jnp.where(u>thre)[0]

    new_u = u[select_id]
    new_vh = vh[:, select_id]
    temp = jnp.sqrt(new_u)
    #if jnp.isnan(temp).any():
    #    print("encounter invalid sqrt eigenvalue")
    inv_sqrt = mat_mul(new_vh/np.sqrt(new_u),new_vh.T)
    return inv_sqrt


def truncate_inv(X, thre=1e-5):
    """
    """
    u,vh = jnp.linalg.eigh(X)
    #if jnp.isnan(u).any():
    #    print("encounter invalid eigenvalue")
    #if jnp.isnan(vh).any():
    #    print("encounter invalid eigenvector")
    select_id = np.where(u>thre)[0]

    new_u = u[select_id]
    new_vh = vh[:, select_id]
    temp = jnp.sqrt(new_u)
    #if jnp.isnan(temp).any():
    #    print("encounter invalid sqrt eigenvalue")
    inv_sqrt = mat_mul(new_vh/(new_u),new_vh.T)
    return inv_sqrt


def truncate_sqrt(X, thre=1e-5):
    """
    """
    u,vh = jnp.linalg.eigh(X)
    select_id = jnp.where(u>thre)[0]

    new_u = u[select_id]
    new_vh = vh[:, select_id]
    inv_sqrt = mat_mul(new_vh*np.sqrt(new_u), new_vh.T)
    return inv_sqrt

def woodbury_identity(Q, lam, n):
    """ compute the inverse of (lam*n*I+QQ^T) using woodbury identitiy lemma
    """
    q = Q.shape[1]
    inv_temp = jsla.solve(lam*n*jnp.eye(q)+mat_mul(Q.T, Q), jnp.eye(q))
    if jnp.isnan(inv_temp).any():
        print("inv_temp is nan")         
    aprox_K = (jnp.eye(n)-mat_mul(mat_mul(Q,inv_temp), Q.T))/(lam*n)
    return aprox_K


#@jax.jit
def cal_loocv_emb(K, kernel_y, lam):
    nD = K.shape[0]
    I = jnp.eye(nD)
    if (nD <= 1000):
        #use linear solver
        Q = jsla.solve(K + lam * nD * I, I)
    else:
        #Nystrom approximation
        q = 250
        select_x = np.random.choice(nD, q, replace=False)
        K_q  = K[select_x, :][:, select_x]
        K_nq = K[:, select_x]
        #if jnp.isnan(K_q).any():
        #    print('K_q is nan')
        inv_Kq_sqrt =  jnp.array(truncate_sqrtinv(K_q))
        tempQ = mat_mul(K_nq, inv_Kq_sqrt)
        Q = woodbury_identity(tempQ, lam, nD)
        #Q = jsla.solve(lam*nD*jnp.eye(q)+tempQ.T.dot(tempQ), jnp.eye(q))
        #aprox_K_XX = (jnp.eye(nD)-(Q.dot(inv_temp)).dot(Q.T))/(self.lam*self.n_samples)
        #if jnp.isnan(Q).any():
        #    print("inv_temp is nan")   


    H = I - mat_mul(K, Q)
    tildeH_inv = jnp.diag(1.0 / jnp.diag(H))
    return jnp.trace(tildeH_inv @ H @ kernel_y @ H @ tildeH_inv)


def cal_l_w (K, kernel_y, low=-4, high=0, n=10):  

    lam_values = np.logspace(low, high, n)
    tolerance=lam_values[1]-lam_values[0]
    grid_search={}
    for lam in lam_values:
        grid_search[lam]=cal_loocv_emb(K, kernel_y, lam)    
    l,loo=min(grid_search.items(), key=operator.itemgetter(1))
    

    return l,loo   


#@jax.jit
def cal_loocv_alpha(K, Sigma, gamma, y, lam):
    nD = K.shape[0]
    I = jnp.eye(nD)
    #if (nD <= 1000):
    temp = jsla.solve(Sigma + lam * nD* I, I)
    """
    else:
        #Nystrom approximation
        select_x = np.random.choice(nD, 250, replace=False)
        K_q  = Sigma[select_x, :][:, select_x]
        K_nq = Sigma[:, select_x]
        inv_Kq_sqrt =  jnp.array(truncate_sqrtinv(K_q))
        tempQ = mat_mul(K_nq, inv_Kq_sqrt)
        temp = woodbury_identity(tempQ, lam, nD)
    """

    H = I - mat_mul(mat_mul(K, gamma), temp)
    tildeH_inv = jnp.diag(1.0 / jnp.diag(H))
    
    return jnp.linalg.norm(mat_mul(tildeH_inv,mat_mul(H,y)))

def cal_l_yw(K, sigma, gamma, y, low=-4, high=0, n=10):
 
    lam_values = np.logspace(low, high, num=n)
    tolerance=lam_values [1]-lam_values [0]
    grid_search={}
    for lam in lam_values:
        grid_search[lam]=cal_loocv_alpha(K, sigma, gamma, y, lam)
    l,loo=min(grid_search.items(), key=operator.itemgetter(1))

    return l,loo 

#@jax.jit
def cal_loocv_m0(D, DC, M, K_CC, lam):
    nD = D.shape[0]
    I = jnp.eye(nD)
    if (nD<=1000):
        Sigma = DC + lam*nD*M
        m1 = M.sum(axis=0)
        alpha = jsla.solve(Sigma, m1)
    else:
      q = min(500, nD)
      select_x = np.random.choice(nD, q, replace=False)
      K_q = M[select_x, :][:, select_x]
      K_nq = M[:, select_x]

      inv_Kq_sqrt = truncate_sqrtinv(K_q)
      Q = K_nq.dot(inv_Kq_sqrt)
      
      aprox_M = Q.dot(Q.T)
      
      # nystrom M^{-1/2}GM^{-1/2}
      inv_M_sqrt = jnp.array(truncate_sqrtinv(aprox_M))

      M_sqrt = jnp.array(truncate_sqrt(aprox_M))

      MGM = inv_M_sqrt.dot(DC.dot(inv_M_sqrt))
      
      q = min(1000, nD)
      
      select_x2 = np.random.choice(nD, q, replace=False)
      K_q2 = MGM[select_x2, :][:, select_x2]
      K_nq2 = MGM[:, select_x2]
      
      inv_Kq2_sqrt = truncate_sqrtinv(K_q2)
      Q2 = K_nq2.dot(inv_Kq2_sqrt)
      
      aprox_inv = woodbury_identity(Q2, lam, nD)

      temp_alpha = inv_M_sqrt.dot(aprox_inv.dot(M_sqrt.sum(axis=1)))
      alpha = temp_alpha/(lam*nD)

    H = I - mat_mul(D, jnp.diag(alpha))
    tildeH_inv = jnp.diag(1.0 / jnp.diag(H))
    return jnp.trace(tildeH_inv @ H @ K_CC @ H @ tildeH_inv)


def cal_l_m(D, DC, M, K_CC, low=-4, high=0, n=10):
    lam_values = np.logspace(low, high, num=n)
    grid_search={}
    for lam in lam_values:
        grid_search[lam] = cal_loocv_m0(D, DC, M, K_CC, lam)
    l,loo=min(grid_search.items(), key=operator.itemgetter(1))

    return l,loo 






def flatten(
    nested_dict,
    seperator='.',
    name=None,
):
    flatten_dict = {}
    if (not nested_dict) and (nested_dict != 0):
        return flatten_dict

    if isinstance(
        nested_dict,
        collections.abc.MutableMapping,
    ):
        for key, value in nested_dict.items():
            if name is not None:
                flatten_dict.update(
                    flatten(
                        nested_dict=value,
                        seperator=seperator,
                        name=f'{name}{seperator}{key}',
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

def _solve_cholesky_kernel(K, y, alpha, sample_weight=None, copy=False):
    # dual_coef = inv(X X^t + alpha*Id) y
    n_samples = K.shape[0]
    n_targets = y.shape[1]

    if copy:
        K = K.copy()

    alpha = np.atleast_1d(alpha)
    one_alpha = (alpha == alpha[0]).all()
    has_sw = isinstance(sample_weight, np.ndarray) or sample_weight not in [1.0, None]

    if has_sw:
        # Unlike other solvers, we need to support sample_weight directly
        # because K might be a pre-computed kernel.
        sw = np.sqrt(np.atleast_1d(sample_weight))
        y = y * sw[:, np.newaxis]
        K *= np.outer(sw, sw)

    if one_alpha:
        # Only one penalty, we can solve multi-target problems in one time.
        K.flat[:: n_samples + 1] += alpha[0]

        try:
            # Note: we must use overwrite_a=False in order to be able to
            #       use the fall-back solution below in case a LinAlgError
            #       is raised
            dual_coef = linalg.solve(K, y, assume_a="pos", overwrite_a=False)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Singular matrix in solving dual problem. Using "
                "least-squares solution instead."
            )
            dual_coef = linalg.lstsq(K, y)[0]

        # K is expensive to compute and store in memory so change it back in
        # case it was user-given.
        K.flat[:: n_samples + 1] -= alpha[0]

        if has_sw:
            dual_coef *= sw[:, np.newaxis]

        return dual_coef
    else:
        # One penalty per target. We need to solve each target separately.
        dual_coefs = np.empty([n_targets, n_samples], K.dtype)

        for dual_coef, target, current_alpha in zip(dual_coefs, y.T, alpha):
            K.flat[:: n_samples + 1] += current_alpha

            dual_coef[:] = linalg.solve(
                K, target, assume_a="pos", overwrite_a=False
            ).ravel()

            K.flat[:: n_samples + 1] -= current_alpha

        if has_sw:
            dual_coefs *= sw[np.newaxis, :]

        return dual_coefs.T



    
