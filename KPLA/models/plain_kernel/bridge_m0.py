from .kernel_utils import *
from .cme import ConditionalMeanEmbed
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import time
import scipy.sparse as ss
import scipy




class CME_m0_cme:
  """ Construct conditonal mean embedding that embeds the bridge function m0.
  Double conditional mean embedding.
  """
  def __init__(self, Cw_x, covars, lam, scale=1., q=None, method='original', lam_min = -4, lam_max=-1,  kernel_dict=None):
    """
    Args:
      Cw_x: ConditionalMeanEmbed, object
      covars: dictionary of covariates, dict
      lam: tuning parametier, float
      scale: kernel length-scale, float
      kernel_dict: dict
      q: rank of the matrix, when Nystrom approximation is used, int
      method: method, "original" or "nystrom"
      'original' for linear solver, 'nystrom' for Nystrom approximation
      lam_min: minimum of lambda (log space) for hyperparameter tuning, float
      lam_max: maximum of lambda (log space) for hyperparameter tuning, float
    """
    self.method = method
    self.sc = scale

    if kernel_dict == None:
      kernel_dict = {}
      kernel_dict['X']='rbf'
      kernel_dict['C']='rbf'
    
    self.Cw_x = Cw_x
    params = Cw_x.get_params()
    
    self.W = params['Y']
    self.w_sc = params['scale']
    kernel_dict['W'] = params['kernel_dict']['Y']
    self.kernel_dict = kernel_dict


    covarsx = {}
    covarsx['X'] = covars['X']
    K_ww = ker_mat(jnp.array(self.W), jnp.array(self.W), kernel=kernel_dict['W'], scale=params['scale'])
    self.Gamma_x = Cw_x.get_mean_embed(covarsx)["Gamma"]

    kx_g_kx = mat_mul(self.Gamma_x.T, mat_mul(K_ww, self.Gamma_x))

    self.X = covars['X']

    K_xx = ker_mat(jnp.array(self.X), jnp.array(self.X), kernel=kernel_dict['X'], scale=self.sc)
    #build the kernel matrix
    self.K_gram =  Hadamard_prod(K_xx, kx_g_kx)
    self.C = covars['C']
    self.n_samples = self.C.shape[0]


    K_CC = ker_mat(jnp.array(self.C), jnp.array(self.C), kernel=kernel_dict['C'], scale=self.sc)

    self.lam = lam

    if (self.lam == None):
      scale_dict = {}
      l_w, loo1 = cal_l_w(self.K_gram, K_CC, low=lam_min, high=lam_max, n=10)
      print('selected lam of m0:', l_w)
      self.lam = l_w

    if self.method=='nystrom':
      # set rank
      if q == None:
        q = min(250, self.n_samples)
      
      # set selected indices
      if q < self.n_samples: 
        select_x = np.random.choice(self.n_samples, q, replace=False)
      else:
        select_x = np.arange(self.n_samples)

      K_q = self.K_gram[select_x, :][:, select_x]
      K_nq = self.K_gram[:, select_x]

      inv_Kq_sqrt = jnp.array(truncate_sqrtinv(K_q))
      Q = mat_mul(K_nq, inv_Kq_sqrt)


      inv_temp = jsla.solve(self.lam*self.n_samples*jnp.eye(q)+Q.T.dot(Q), jnp.eye(q))
      if jnp.isnan(inv_temp).any():
        print("inv_temp is nan")         
      self.aprox_K_gram_inv = (jnp.eye(self.n_samples)-(Q.dot(inv_temp)).dot(Q.T))/(self.lam*self.n_samples)

    elif self.method=='original':
      self.K_gram_inv = jsla.solve(self.lam*self.n_samples*jnp.eye(self.n_samples)+self.K_gram, jnp.eye(self.n_samples))


  def get_mean_embed(self, Cw_x, new_x):
    """
    Args:
      Cw_x: ConditionalMeanEmbed, object
      new_x: shape (n2_samples, n2_features)
    """
    
    # compute the gram matrix
    K_Xnewx = ker_mat(jnp.array(self.X), jnp.array(new_x['X']), kernel=self.kernel_dict['X'], scale=self.sc)
    
    
    params1 = Cw_x.get_mean_embed(new_x)
    Gamma1_newx = params1["Gamma"] #(n_samples, n6_samples)
    W1 = params1["Y"]

    K_w1w2 = ker_mat(jnp.array(self.W), jnp.array(W1), kernel=self.kernel_dict['W'], scale=self.w_sc) #(n_samples, n'_samples)
    kx_g_knewx = mat_mul(mat_mul(self.Gamma_x.T, K_w1w2), Gamma1_newx) #(n5_samples, n6_samples)

    G_x = Hadamard_prod(K_Xnewx, kx_g_knewx)

    if self.method == 'nystrom':
      Gamma = mat_mul(self.aprox_K_gram_inv, G_x)
    elif self.method == 'original':
      Gamma = mat_mul(self.K_gram_inv, G_x)

    return Gamma



  def get_A_operator(self, Cw_x, new_x):
    """ return \sum_i beta_i(new_x)\phi(c_i)
    Args:
      Cw_x: ConditionalMeanEmbed object
      new_x: shape (n2_samples, n2_features)
    Returns:
    beta: shape (n2_samples, n_samples)
    """
    beta = self.get_mean_embed(Cw_x, new_x).T
    params = {}
    params["C"]=self.C
    params["scale"] = self.sc
    params["beta"] = beta
    return params



  def __call__(self, new_c, Cw_x, new_x):

    params = self.get_A_operator(Cw_x, new_x)
    K_Cnewc = ker_mat(jnp.array(self.C), jnp.array(new_c), kernel=self.kernel_dict['C'], scale=self.sc) #(n5_samples, n2_samples)
    return Hadamard_prod(params['beta'].T, K_Cnewc).sum(axis=0)

  def get_coefs(self, Cw_x, new_x):
    params = self.get_A_operator(Cw_x, new_x)
    return params['beta'].T #(n5_samples, n2_samples)
