from .kernel_utils import *
from .cme import ConditionalMeanEmbed
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import time
import scipy.sparse as ss


class Bridge_h0:
  """ Construct the bridge function h0 = \sum_i alpha_ij \phi(w_i)\otimes\phi(c_j)
      vec(alpha)=(Gamma_xc\odot I)(n2*lam I + \Sigma)^{-1}y, alpha shape=(n1_samples, n2_samples)
      Gamma_xc = mu_w_cx.get_mean_embed(x,c)['Gamma'] #(n1_samples, n2_samples)
      \Sigma = (Gamma_xc^T K_ww Gamma_xc)K_cc
  """
  def __init__(self, Cw_xc, covars, Y, lam, scale=1., method='original', lam_min=-4, lam_max=-1,  kernel_dict=None):
    """Initiate the parameters
    Args:
      Cw_xc: object, ConditionalMeanEmbed
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
    n_sample = Y.shape[0]
    # construct A matrix
    C = covars["C"]

    self.lam_max = lam_max
    self.lam_min = lam_min

    if kernel_dict == None:
      kernel_dict = {}
      kernel_dict['C'] = 'rbf'

    K_CC = ker_mat(jnp.array(C), jnp.array(C), kernel=kernel_dict['C'], scale=self.sc)
    self.C = C
    params = Cw_xc.get_params()
    W = params["Y"]
    self.w_sc = params["scale"]
    self.W = W
    kernel_dict['W'] = params['kernel_dict']['Y']
    K_WW = ker_mat(jnp.array(W), jnp.array(W), kernel=kernel_dict['W'], scale=params["scale"])
    self.kernel_dict = kernel_dict

    assert(set(params["Xlist"]) == set(covars.keys()))
    # construct Gamma_xc matrix
    Gamma_xc = Cw_xc.get_mean_embed(covars)["Gamma"] #shape = (n1_samples, n2_samples)
    

    # construct sigma
    Sigma = Hadamard_prod(mat_mul(mat_mul(Gamma_xc.T, K_WW), Gamma_xc), K_CC)
    
    if lam == None:
      #implement parameter selection
      #use random subsample
      lam = self.model_select(n_sample, K_WW, K_CC, Gamma_xc, Y)



    #print("rank of sigma", jnp.linalg.matrix_rank(Sigma))
    F = Sigma + n_sample*lam*jnp.eye(n_sample)
    #print("F is pd", is_pos_def(F))
    
    t2 = time.time()


    
    #using linear solver
    
    
    if method == 'nystrom':
      print('use Nystrom method to estimate h0')
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
      print('use linear solver to estimate h0')
      vec_alpha = jsla.solve(F, Y)

    
    t25 = time.time()

    
    vec_alpha = stage2_weights(Gamma_xc, vec_alpha)
    
    t3 = time.time()

    print("processing time: matrix preparation:%.4f solving inverse:%.4f, %.4f"%(t2-t1, t25-t2, t3-t25))
    self.alpha = vec_alpha.reshape((-1, n_sample)) #shape=(n1_sample, n2_sample)

  def model_select(self, n_sample, K_WW, K_CC, Gamma_xc, Y):
      
      if (n_sample >= 1000) or (K_WW.shape[0]>1000):
        select_id = np.random.choice(n_sample, min(n_sample, 1000), replace=False)
        select_id2 = np.random.choice(K_WW.shape[0], min(K_WW.shape[0], 1000), replace=False)
        
        K_sub_WW = K_WW[select_id2,:]
        K_sub_WW = K_sub_WW[:, select_id2]

        K_sub_CC = K_CC[select_id, :]
        K_sub_CC = K_sub_CC[:, select_id]

        Y_sub = Y[select_id,...]

        Gamma_sub_xc = Gamma_xc[select_id2, :]
        Gamma_sub_xc = Gamma_sub_xc[:, select_id]
      else:
        K_sub_WW = K_WW
        K_sub_CC = K_CC
        Y_sub = Y
        Gamma_sub_xc = Gamma_xc

      sub_Sigma =  Hadamard_prod(mat_mul(mat_mul(Gamma_sub_xc.T, K_sub_WW), Gamma_sub_xc), K_sub_CC)
      D_t = modif_kron(mat_mul(K_sub_WW, Gamma_sub_xc), K_sub_CC) 
      mk_gamma_I=mat_trans(modif_kron(Gamma_sub_xc, jnp.eye(1000)))
      lam, loo2 = cal_l_yw(D_t, sub_Sigma, mk_gamma_I , Y_sub, self.lam_min, self.lam_max)
      print('selected lam of h_0:', lam)

      return lam

  def __call__(self, new_w, new_c):
    """return h0(w,c)
    Args:
        new_w: variable W, ndarray shape = (n3_samples, n1_features)
        new_c: variable C, ndarray shape = (n3_samples, n2_features)}
    Returns:
        h0(w,c): ndarray shape = (n3_samples)
    """
    # compute K_newWW

    K_WnewW = ker_mat(jnp.array(self.W), jnp.array(new_w), kernel=self.kernel_dict['W'], scale=self.w_sc) #(n1_sample, n3_sample)


    # compute K_newCC
    K_CnewC = ker_mat(jnp.array(self.C), jnp.array(new_c), kernel=self.kernel_dict['C'], scale=self.sc) #(n2_sample, n3_sample)

    print(K_WnewW.shape, K_CnewC.shape, self.alpha.shape)
    h_wc = fn = lambda kc, kw: jnp.dot(mat_mul(self.alpha, kc), kw)
    v = vmap(h_wc, (1,1))
    return v(K_CnewC, K_WnewW)

  def get_EYx(self, new_x, cme_WC_x):
    """ when computing E[Y|c,x]=<h0, phi(c)\otimes mu_w|x,c>
    Args:
      new_x: ndarray shape=(n4_samples, n_features)
      cme_WC_x: ConditionalMeanEmbed
    """

    t1 = time.time()
    params = cme_WC_x.get_mean_embed(new_x)
    t2 = time.time()
    if len(self.W.shape) == 1:
      w_features = 1
    else:
      w_features = self.W.shape[1]

    if len(self.C.shape) == 1:
      c_features = 1
    else:
      c_features = self.C.shape[1]

    # params["Y"] shape=(n1_samples, w_features+c_features)
    new_w = params["Y"][:, 0:w_features]
    new_c = params["Y"][:, w_features:w_features+c_features]
    # Gamma shape=(n1_samples, n4_samples)
    kcTalphakw = self.__call__(new_w, new_c)
    t3 = time.time()
    fn = lambda x: jnp.dot(kcTalphakw, x)
    v = vmap(fn, (1))

    result = v(params["Gamma"])
    t4 = time.time()

    print("inference time: %.4f/%.4f/%.4f"%(t2-t1, t3-t2, t4-t3))
    return result

  def get_EYx_independent(self, new_x, cme_w_x, cme_c_x):
    """ E[Y | x] = <h0, cme_w_x \otimes cme_c_x>
    Args:
      new_x: ndarray shape=(n5_samples, n_features)
      cme_w_x: ConditionalMeanEmbed, object
      cme_c_x: CME_m0, object
    """
    t1 = time.time()
    #params_w = cme_w_x.get_params()
    new_w = cme_w_x.Y #params_w["Y"]
    Gamma_w = cme_w_x.get_mean_embed(new_x)['Gamma'] #(n3_samples, n5_samples)


    #params_c = cme_c_x.get_params()
    new_c = cme_c_x.C #params_c["C"]

    Gamma_c = cme_c_x.get_A_operator(cme_w_x, new_x)['beta'].T #(n4_sample, n5_sample)
    t2 = time.time()
    # compute K_newWW
    K_WnewW = ker_mat(jnp.array(self.W), jnp.array(new_w), kernel=self.kernel_dict['W'], scale=self.w_sc) #(n1_sample, n3_sample)
    # compute K_newCC
    K_CnewC = ker_mat(jnp.array(self.C), jnp.array(new_c), kernel=self.kernel_dict['C'], scale=self.sc) #(n2_sample, n4_sample)

    kcTalphakw = mat_mul(K_WnewW.T, mat_mul(self.alpha,K_CnewC)) #(n3_sample,  n4_sample)
    t3 = time.time()

    h_wc = fn = lambda b1, b2: jnp.dot(mat_mul(kcTalphakw, b1), b2)
    v = vmap(h_wc, (1,1))
    result = v(Gamma_c, Gamma_w)
    t4 = time.time()

    print("inference time: %.4f/%.4f/%.4f"%(t2-t1, t3-t2, t4-t3))
    return result

  def get_EYx_independent_cme(self, new_x, cme_w_x, cme_c_x):
    """ E[Y | x] = <h0, cme_w_x \otimes cme_c_x>
    Args:
      new_x: ndarray shape=(n5_samples, n_features)
      cme_w_x: ConditionalMeanEmbed, object
      cme_c_x: ConditionalMeanEmbed, object
    """
    t1 = time.time()
    #params_w = cme_w_x.get_params()
    new_w = cme_w_x.Y #params_w["Y"]
    Gamma_w = cme_w_x.get_mean_embed(new_x)['Gamma'] #(n3_samples, n5_samples)


    #params_c = cme_c_x.get_params()
    new_c = cme_c_x.Y #params_c["Y"]
    Gamma_c = cme_c_x.get_mean_embed(new_x)['Gamma'] #(n4_sample, n5_sample)
    t2 = time.time()
    # compute K_newWW
    K_WnewW = ker_mat(jnp.array(self.W), jnp.array(new_w), kernel=self.kernel_dict['W'], scale=self.w_sc) #(n1_sample, n3_sample)
    # compute K_newCC
    K_CnewC = ker_mat(jnp.array(self.C), jnp.array(new_c), kernel=self.kernel_dict['C'], scale=self.sc) #(n2_sample, n4_sample)

    kwTalphakc = mat_mul(K_WnewW.T, mat_mul(self.alpha,K_CnewC)) #(n3_sample,  n4_sample)
    t3 = time.time()

    h_wc = fn = lambda b1, b2: jnp.dot(mat_mul(kwTalphakc, b1), b2)
    v = vmap(h_wc, (1,1))
    result = v(Gamma_c, Gamma_w)

    t4 = time.time()

    print("inference time: %.4f/%.4f/%.4f"%(t2-t1, t3-t2, t4-t3))
    return result


class Bridge_h0_classification(Bridge_h0):


  def __init__(self, Cw_xc, covars, Y, lam, scale=1., method='original', lam_min=-4, lam_max=-1,  kernel_dict=None):
    """Initiate the parameters
    Args:
      Cw_xc: object, ConditionalMeanEmbed
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
    n_sample = Y.shape[0]
    # construct A matrix
    C = covars["C"]

    self.lam_max = lam_max
    self.lam_min = lam_min

    if kernel_dict == None:
      kernel_dict = {}
      kernel_dict['C'] = 'rbf'

    K_CC = ker_mat(jnp.array(C), jnp.array(C), kernel=kernel_dict['C'], scale=self.sc)
    self.C = C
    params = Cw_xc.get_params()
    W = params["Y"]
    self.w_sc = params["scale"]
    self.W = W
    kernel_dict['W'] = params['kernel_dict']['Y']
    K_WW = ker_mat(jnp.array(W), jnp.array(W), kernel=kernel_dict['W'], scale=params["scale"])
    self.kernel_dict = kernel_dict

    assert(set(params["Xlist"]) == set(covars.keys()))
    # construct Gamma_xc matrix
    Gamma_xc = Cw_xc.get_mean_embed(covars)["Gamma"] #shape = (n1_samples, n2_samples)
    

    # construct sigma
    Sigma = Hadamard_prod(mat_mul(mat_mul(Gamma_xc.T, K_WW), Gamma_xc), K_CC)
    

    if lam == None:
      #implement parameter selection
      #use random subsample
      lam = self.model_select(n_sample, K_WW, K_CC, Gamma_xc, Y)


    #print("rank of sigma", jnp.linalg.matrix_rank(Sigma))
    F = Sigma + n_sample*lam*jnp.eye(n_sample)
    #print("F is pd", is_pos_def(F))
    
    t2 = time.time()

    
    #using linear solver
    
    

    print('use linear solver to estimate h0')
    vec_alpha = jsla.solve(F, Y)
    t25 = time.time()
    fn = lambda a: stage2_weights(Gamma_xc, a).reshape((-1, n_sample))
    parallel_stage2 = vmap(fn, (1))
    self.alpha = parallel_stage2(vec_alpha).transpose((1,2,0))

    
    

    
    #vec_alpha = stage2_weights(Gamma_xc, vec_alpha)
    
    t3 = time.time()
    print("processing time: matrix preparation:%.4f solving inverse:%.4f, %.4f"%(t2-t1, t25-t2, t3-t25))
    #self.alpha = vec_alpha.reshape((-1, n_sample)) #shape=(n1_sample, n2_sample)


  def __call__(self, new_w, new_c):
    """return h0(w,c)
    Args:
        new_w: variable W, ndarray shape = (n3_samples, n1_features)
        new_c: variable C, ndarray shape = (n3_samples, n2_features)}
    Returns:
        h0(w,c): ndarray shape = (n3_samples)
    """
    # compute K_newWW
    K_WnewW = ker_mat(jnp.array(self.W), jnp.array(new_w), kernel=self.kernel_dict['W'], scale=self.w_sc) #(n1_sample, n3_sample)


    # compute K_newCC
    K_CnewC = ker_mat(jnp.array(self.C), jnp.array(new_c), kernel=self.kernel_dict['C'], scale=self.sc) #(n2_sample, n3_sample)

    n_categories = self.alpha.shape[2]

    h_wc = lambda kc,kw : vmap(lambda d: jnp.dot(mat_mul(self.alpha[:,:, d], kc), kw))(jnp.arange(n_categories))
    #h_wc = lambda d: vmap(lambda kc, kw: jnp.dot(mat_mul(self.alpha[:,:, d], kc), kw), (1,1))

    outer_v = vmap(h_wc, (1,1))(K_CnewC, K_WnewW)
    return outer_v #(n3_samples, outer)

  def get_EYx(self, new_x, cme_WC_x):
    """ when computing E[Y|c,x]=<h0, phi(c)\otimes mu_w|x,c>
    Args:
      new_x: ndarray shape=(n4_samples, n_features)
      cme_WC_x: ConditionalMeanEmbed
    """

    t1 = time.time()
    params = cme_WC_x.get_mean_embed(new_x)
    t2 = time.time()
    if len(self.W.shape) == 1:
      w_features = 1
    else:
      w_features = self.W.shape[1]

    if len(self.C.shape) == 1:
      c_features = 1
    else:
      c_features = self.C.shape[1]

    # params["Y"] shape=(n1_samples, w_features+c_features)
    new_w = params["Y"][:, 0:w_features]
    new_c = params["Y"][:, w_features:w_features+c_features]
    # Gamma shape=(n1_samples, n4_samples)
    kcTalphakw = self.__call__(new_w, new_c)
    t3 = time.time()
    #fn = lambda x: jnp.dot(kcTalphakw, x)
    #v = vmap(fn, (1))

    result = mat_mul(kcTalphakw.T, params["Gamma"]).T #(n4_samples, n_categories)
    t4 = time.time()

    print("inference time: %.4f/%.4f/%.4f"%(t2-t1, t3-t2, t4-t3))
    return result