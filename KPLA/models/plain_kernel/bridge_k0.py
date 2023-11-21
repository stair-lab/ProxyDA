"""
Implementation of the kernel bridge function k0
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT




from .kernel_utils import *
from .cme import ConditionalMeanEmbed
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import time
import scipy.sparse as ss


class Bridge_k0:
  """ Construct the bridge function k0 = \sum_i alpha_ij \phi(w_i)\otimes\phi(c_j)
      vec(alpha)=(Gamma_xc\odot I)(n2*lam I + \Sigma)^{-1}y, alpha shape=(n1_samples, n2_samples)
      Gamma_xc = mu_w_cx.get_mean_embed(x,c)['Gamma'] #(n1_samples, n2_samples)
      \Sigma = (Gamma_xc^T K_ww Gamma_xc)K_cc
  """


  def __init__(self, Cw_xz, covars, Y, lam, scale=1., method='original', lam_min=-4, lam_max=-1,  kernel_dict=None):
    """Initiate the parameters
    Args:
      Cw_xz: object, ConditionalMeanEmbed
      covars: covariates, dict {"C": ndarray shape=(n2_samples, n1_features), "X": ndarray shape=(n2_samples, n2_features)}
      Y: labels, (n2_samples,)
      lam: reuglarization parameter, lam
      scale: kernel length scale, float
      method: approximation method, str
      'original' for linear solver, 'nystrom' for Nystrom approximation
      lam_min: minimum of lambda (log space) for hyperparameter tuning, float
      lam_max: maximum of lambda (log space) for hyperparameter tuning, float
    """
    t1 = time.time()
    self.sc = scale
    self.lam_min = lam_min
    self.lam_max = lam_max
    n_sample = Y.shape[0]
    # construct A matrix
    X = covars["X"]

    if kernel_dict == None:
      kernel_dict = {}
      kernel_dict['X'] = 'rbf'

    K_XX = ker_mat(jnp.array(X), jnp.array(X), kernel=kernel_dict['X'], scale=self.sc)
    self.X = X
    params = Cw_xz.get_params()

    W = params["Y"]
    self.w_sc = params["scale"]
    self.W = W
    kernel_dict['W'] = params['kernel_dict']['Y']
    K_WW = ker_mat(jnp.array(W), jnp.array(W), kernel=kernel_dict['W'], scale=params["scale"])
    self.kernel_dict = kernel_dict


    assert(set(params["Xlist"]) == set(covars.keys()))
    # construct Gamma_xc matrix
    Gamma_xz = Cw_xz.get_mean_embed(covars)["Gamma"] #shape = (n1_samples, n2_samples)
    

    # construct sigma
    Sigma = Hadamard_prod(mat_mul(mat_mul(Gamma_xz.T, K_WW), Gamma_xz), K_XX)
    
    if lam == None:
      
      lam = self.model_select(n_sample, K_WW, K_XX, Gamma_xz, Y)


    #print("rank of sigma", jnp.linalg.matrix_rank(Sigma))
    F = Sigma + n_sample*lam*jnp.eye(n_sample)
    #print("F is pd", is_pos_def(F))
    
    t2 = time.time()


    
    #using linear solver
    
    
    if method == 'nystrom':
      print('use Nystrom method to estimate k0')
      #q = min(2*int(np.sqrt(n_sample)), int(n_sample/10))
      q = min(250, n_sample)
      select_id = np.random.choice(n_sample, q, replace=False)
      
      K_q = Sigma[select_id, :][:, select_id]
      K_nq = Sigma[:, select_id]
      

      inv_Kq_sqrt =  jnp.array(truncate_sqrtinv(K_q))
      Q = K_nq.dot(inv_Kq_sqrt)

      inv_temp = jsla.solve(lam*n_sample*jnp.eye(q)+Q.T.dot(Q), jnp.eye(q))
      if jnp.isnan(inv_temp).any():
        print("inv_temp is nan")         
      aprox_K = (jnp.eye(n_sample)-(Q.dot(inv_temp)).dot(Q.T))/(lam*n_sample)

      vec_alpha = mat_mul(aprox_K, Y)

    elif method == 'original':
      print('use linear solver to estimate k0')
      vec_alpha = jsla.solve(F, Y)
    
    t25 = time.time()

    
    vec_alpha = stage2_weights(Gamma_xz, vec_alpha)
    
    t3 = time.time()
    print("processing time: matrix preparation:%.4f solving inverse:%.4f, %.4f"%(t2-t1, t25-t2, t3-t25))
    self.alpha = vec_alpha.reshape((-1, n_sample)) #shape=(n1_sample, n2_sample)


  def model_select(self, n_sample, K_WW, K_XX, Gamma_zx, Y):
      
      if (n_sample >= 1000) or (K_WW.shape[0]>1000):
        select_id = np.random.choice(n_sample, min(n_sample, 1000), replace=False)
        select_id2 = np.random.choice(K_WW.shape[0], min(K_WW.shape[0], 1000), replace=False)
        
        K_sub_WW = K_WW[select_id2,:]
        K_sub_WW = K_sub_WW[:, select_id2]

        K_sub_XX = K_XX[select_id, :]
        K_sub_XX = K_sub_XX[:, select_id]

        Y_sub = Y[select_id,...]

        Gamma_sub_zx = Gamma_zx[select_id2, :]
        Gamma_sub_zx = Gamma_sub_zx[:, select_id]
      else:
        K_sub_WW = K_WW
        K_sub_XX = K_XX
        Y_sub = Y
        Gamma_sub_zx = Gamma_zx

      sub_Sigma =  Hadamard_prod(mat_mul(mat_mul(Gamma_sub_zx.T, K_sub_WW), Gamma_sub_zx), K_sub_XX)
      D_t = modif_kron(mat_mul(K_sub_WW, Gamma_sub_zx), K_sub_XX) 
      mk_gamma_I=mat_trans(modif_kron(Gamma_sub_zx, jnp.eye(1000)))
      lam, loo2 = cal_l_yw(D_t, sub_Sigma, mk_gamma_I , Y_sub, self.lam_min, self.lam_max)
      print('selected lam of h_0:', lam)

      return lam

  def __call__(self, new_w, new_x, Gamma_x):
    """return k0(w,c)
    Args:
        new_w: variable W, ndarray shape = (n3_samples, n1_features)
        new_x: variable X, ndarray shape = (n3_samples, n2_features)}
    Returns:
        k0(w,x): ndarray shape = (n3_samples)
    """
    # compute K_newWW
    
    K_WnewW = ker_mat(jnp.array(self.W), jnp.array(new_w), kernel=self.kernel_dict['W'], scale=self.w_sc) #(n1_sample, n3_sample)
    K_WnewW = mat_mul(K_WnewW, Gamma_x)

    # compute K_newCC
    K_XnewX = ker_mat(jnp.array(self.X), jnp.array(new_x), kernel=self.kernel_dict['X'], scale=self.sc) #(n2_sample, n3_sample)

    h_wx  = lambda kx, kw: jnp.dot(mat_mul(self.alpha, kx), kw)
    v = vmap(h_wx, (1,1))
    return v(K_XnewX, K_WnewW)

  def get_EYx(self, new_x, cme_W_x):
    """ when computing E[Y|c,x]=<k0, phi(c)\otimes mu_w|x,c>
    Args:
      new_x: ndarray shape=(n4_samples, n_features)
      cme_WC_x: ConditionalMeanEmbed
    """
    #print('alpha', self.alpha.shape)
    t1 = time.time()
    
    params = cme_W_x.get_mean_embed(new_x)
    #print('Gamma', params["Gamma"].shape)
    t2 = time.time()

    # params["Y"] shape=(n1_samples, w_features+c_features)
    new_w = params["Y"]
    # Gamma shape=(n1_samples, n4_samples)
    Gamma_x = params["Gamma"]
    kxTalphakw = self.__call__(new_w, new_x["X"], Gamma_x)
    t3 = time.time()
    #fn = lambda x: jnp.dot(kxTalphakw, x)
    #v = vmap(fn, (1))
    #print(kxTalphakw.shape)

    #result = v(params["Gamma"])
    t4 = time.time()

    print("inference time: %.4f/%.4f/%.4f"%(t2-t1, t3-t2, t4-t3))
    return kxTalphakw








class Bridge_k0_categorical(Bridge_k0):
  """ Construct the bridge function k0 = \sum_i alpha_ij \phi(w_i)\otimes\phi(c_j)
      vec(alpha)=(Gamma_xc\odot I)(n2*lam I + \Sigma)^{-1}y, alpha shape=(n1_samples, n2_samples)
      Gamma_xc = mu_w_cx.get_mean_embed(x,c)['Gamma'] #(n1_samples, n2_samples)
      \Sigma = (Gamma_xc^T K_ww Gamma_xc)K_cc
  """

  def __init__(self, Cw_xz_dict, covars, Y, lam, scale=1., method='original', lam_min=-4, lam_max=-1,  kernel_dict=None):
    """Initiate the parameters
    Args:
      Cw_xz_dict: dict, ConditionalMeanEmbed
      covars: covariates, dict {"C": ndarray shape=(n2_samples, n1_features), "X": ndarray shape=(n2_samples, n2_features)}
      Y: labels labels,  (n2_samples,)
      lam: reuglarization parameter, lam
      scale: kernel length scale, float
      method: approximation method, str
      'original' for linear solver, 'nystrom' for Nystrom approximation
      lam_min: minimum of lambda (log space) for hyperparameter tuning, float
      lam_max: maximum of lambda (log space) for hyperparameter tuning, float
    """
    self.sc = scale

    if kernel_dict == None:
      kernel_dict = {}
      kernel_dict['X'] = 'rbf'
    
    
    

    # concatenate Z
    z_label = Cw_xz_dict.keys()
    #print(z_label)
    #print(type(z_label))
    k0_lookup = {}
    for z in z_label:
      
      idx = jnp.where(covars['Z'] == z)[0]
      nz_sample = idx.shape[0]
      X_z = covars['X'][idx,...]
      Y_z = Y[idx,...]
      covarsz = {}
      #covarsz['Z'] = covars['Z'][idx,...]
      covarsz['X'] = covars['X'][idx,...]


      Cw_xz = Cw_xz_dict[z]
      params = Cw_xz.get_params()
      kernel_dict['W'] = params['kernel_dict']['Y']
      
      w_sc = params["scale"]
      W_z = params["Y"]

      k0 = Bridge_k0(Cw_xz, covarsz, Y_z, lam, 
                      kernel_dict = kernel_dict, scale = self.sc,  
                      method=method, lam_min=lam_min, lam_max=lam_max)

      k0_lookup[z] = k0

    self.kernel_dict = kernel_dict
    self.k0_lookup = k0_lookup
    self.z_label = z_label


  def __call__(self, new_w, new_x, Gamma_x):
    """return k0(w,c)
    Args:
        new_w: variable W, ndarray shape = (n3_samples, n1_features)
        new_x: variable X, ndarray shape = (n3_samples, n2_features)}
    Returns:
        k0(w,x): ndarray shape = (n3_samples)
    """
    output = []
    for z in self.z_label:
      output.append(self.k0_lookup[z](new_w, new_x, Gamma_x))
    output = jnp.array(output).sum(axis=0)

    return output





class Bridge_k0_classification(Bridge_k0):
  """ Construct the bridge function k0 = \sum_i alpha_ij \phi(w_i)\otimes\phi(c_j)
      vec(alpha)=(Gamma_xc\odot I)(n2*lam I + \Sigma)^{-1}y, alpha shape=(n1_samples, n2_samples)
      Gamma_xc = mu_w_cx.get_mean_embed(x,c)['Gamma'] #(n1_samples, n2_samples)
      \Sigma = (Gamma_xc^T K_ww Gamma_xc)K_cc
  """

  def __init__(self, Cw_xz, covars, Y, lam, scale=1., method='original', lam_min=-4, lam_max=-1,  kernel_dict=None):
    """Initiate the parameters
    Args:
      Cw_xz: object, ConditionalMeanEmbed
      covars: covariates, dict {"C": ndarray shape=(n2_samples, n1_features), "X": ndarray shape=(n2_samples, n2_features)}
      Y: labels, (n2_samples,)
      lam: reuglarization parameter, lam
      scale: kernel length scale, float
      method: approximation method, str
      'original' for linear solver, 'nystrom' for Nystrom approximation
      lam_min: minimum of lambda (log space) for hyperparameter tuning, float
      lam_max: maximum of lambda (log space) for hyperparameter tuning, float
    """
    t1 = time.time()
    self.lam_min = lam_min
    self.lam_max = lam_max
    self.sc = scale
    n_sample = Y.shape[0]
    # construct A matrix
    X = covars["X"]

    if kernel_dict == None:
      kernel_dict = {}
      kernel_dict['X'] = 'rbf'

    K_XX = ker_mat(jnp.array(X), jnp.array(X), kernel=kernel_dict['X'], scale=self.sc)
    self.X = X
    params = Cw_xz.get_params()

    W = params["Y"]
    self.w_sc = params["scale"]
    self.W = W
    kernel_dict['W'] = params['kernel_dict']['Y']
    K_WW = ker_mat(jnp.array(W), jnp.array(W), kernel=kernel_dict['W'], scale=params["scale"])
    self.kernel_dict = kernel_dict


    assert(set(params["Xlist"]) == set(covars.keys()))
    # construct Gamma_xc matrix
    Gamma_xz = Cw_xz.get_mean_embed(covars)["Gamma"] #shape = (n1_samples, n2_samples)
    

    # construct sigma
    Sigma = Hadamard_prod(mat_mul(mat_mul(Gamma_xz.T, K_WW), Gamma_xz), K_XX)
    

    if lam == None:
      lam  = self.model_select(n_sample, K_WW, K_XX, Gamma_xz, Y)

    #print("rank of sigma", jnp.linalg.matrix_rank(Sigma))
    F = Sigma + n_sample*lam*jnp.eye(n_sample)
    #print("F is pd", is_pos_def(F))
    
    t2 = time.time()


    
    #using linear solver
    
    

 
    print('use linear solver to estimate k0')
    vec_alpha = jsla.solve(F, Y)
    
    t25 = time.time()

    
    fn = lambda a: stage2_weights(Gamma_xz, a).reshape((-1, n_sample))
    parallel_stage2 = vmap(fn, (1))
    
    self.alpha = parallel_stage2(vec_alpha).transpose((1,2,0))

      
    t3 = time.time()
    print("processing time: matrix preparation:%.4f solving inverse:%.4f, %.4f"%(t2-t1, t25-t2, t3-t25))
    

  def __call__(self, new_w, new_x, Gamma_x):
    """return k0(w,c)
    Args:
        new_w: variable W, ndarray shape = (n3_samples, n1_features)
        new_x: variable X, ndarray shape = (n3_samples, n2_features)}
    Returns:
        k0(w,x): ndarray shape = (n3_samples)
    """
    # compute K_newWW
    
    K_WnewW = ker_mat(jnp.array(self.W), jnp.array(new_w), kernel=self.kernel_dict['W'], scale=self.w_sc) #(n1_sample, n3_sample)
    K_WnewW = mat_mul(K_WnewW, Gamma_x)

    # compute K_newCC
    K_XnewX = ker_mat(jnp.array(self.X), jnp.array(new_x), kernel=self.kernel_dict['X'], scale=self.sc) #(n2_sample, n3_sample)


    n_categories = self.alpha.shape[2]

    h_wx = lambda kx,kw : vmap(lambda d: jnp.dot(mat_mul(self.alpha[:,:, d], kx), kw))(jnp.arange(n_categories))
    #h_wc = lambda d: vmap(lambda kc, kw: jnp.dot(mat_mul(self.alpha[:,:, d], kc), kw), (1,1))

    outer_v = vmap(h_wx, (1,1))(K_XnewX, K_WnewW)
    return outer_v #(n3_samples, n_category)




class Bridge_k0_catclass(Bridge_k0_categorical):
  def __init__(self, Cw_xz_dict, covars, Y, lam, scale=1., method='original', lam_min=-4, lam_max=-1,  kernel_dict=None):
    """Initiate the parameters
    Args:
      Cw_xz_dict: dict, ConditionalMeanEmbed
      covars: covariates, dict {"C": ndarray shape=(n2_samples, n1_features), "X": ndarray shape=(n2_samples, n2_features)}
      Y: labels labels,  (n2_samples,)
      lam: reuglarization parameter, lam
      scale: kernel length scale, float
      method: approximation method, str
      'original' for linear solver, 'nystrom' for Nystrom approximation
      lam_min: minimum of lambda (log space) for hyperparameter tuning, float
      lam_max: maximum of lambda (log space) for hyperparameter tuning, float
    """
    self.sc = scale

    if kernel_dict == None:
      kernel_dict = {}
      kernel_dict['X'] = 'rbf'
    
    
    

    # concatenate Z
    z_label = Cw_xz_dict.keys()
    #print(z_label)
    #print(type(z_label))
    k0_lookup = {}
    for z in z_label:
      
      idx = jnp.where(covars['Z'] == z)[0]
      nz_sample = idx.shape[0]
      X_z = covars['X'][idx,...]
      Y_z = Y[idx,...]
      covarsz = {}
      #covarsz['Z'] = covars['Z'][idx,...]
      covarsz['X'] = covars['X'][idx,...]


      Cw_xz = Cw_xz_dict[z]
      params = Cw_xz.get_params()
      kernel_dict['W'] = params['kernel_dict']['Y']
      
      w_sc = params["scale"]
      W_z = params["Y"]

      k0 = Bridge_k0_classification(Cw_xz, covarsz, Y_z, lam, 
                      kernel_dict = kernel_dict, scale = self.sc,  
                      method=method, lam_min=lam_min, lam_max=lam_max)

      k0_lookup[z] = k0

    self.kernel_dict = kernel_dict
    self.k0_lookup = k0_lookup
    self.z_label = z_label


