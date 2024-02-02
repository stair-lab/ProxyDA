"""
Implementation of the kernel bridge function m0
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT




from KPLA.models.plain_kernel.kernel_utils import hadamard_prod,ker_mat, mat_mul, stage2_weights, modif_kron, mat_trans, cal_l_yw
import numpy as np
import jax.numpy as jnp
from jax import vmap
import jax.scipy.linalg as jsla
import time


class BridgeM0:
  """ Construct the bridge function m0
  """

  def __init__(self,
              cme_w_xz,
              covars,
              y,
              lam,
              scale=1.,
              method="original",
              lam_min=-4,
              lam_max=-1,
              kernel_dict=None):
    """Initiate the parameters
    Args:
      cme_w_xz: object, ConditionalMeanEmbed
      covars: covariates, dict {"C": ndarray shape=(n2_samples, n1_features), 
                                "X": ndarray shape=(n2_samples, n2_features)}
      y: labels, (n2_samples,)
      lam: reuglarization parameter, lam
      scale: kernel length scale, float
      method: approximation method, str
      'original' for linear solver, 'nystrom' for Nystrom approximation
      lam_min: minimum of lambda (log space) for hyperparameter tuning, float
      lam_max: maximum of lambda (log space) for hyperparameter tuning, float
      kernel_dict: specify kernel functions, dict
    """
    t1 = time.time()
    self.sc = scale
    self.lam_min = lam_min
    self.lam_max = lam_max
    n_sample = y.shape[0]
    # construct A matrix
    x = covars["X"]

    if kernel_dict is None:
      kernel_dict = {}
      kernel_dict["X"] = "rbf"

    ker_xx = ker_mat(jnp.array(x), jnp.array(x),
                    kernel=kernel_dict["X"],
                    scale=self.sc)
    self.x = x
    params = cme_w_xz.get_params()

    w = params["Y"]
    self.w_sc = params["scale"]
    self.w = w
    kernel_dict["W"] = params["kernel_dict"]["Y"]
    ker_ww = ker_mat(jnp.array(w), jnp.array(w),
                   kernel=kernel_dict["W"],
                   scale=params["scale"])
    self.kernel_dict = kernel_dict

    assert(set(params["Xlist"]) == set(covars.keys()))
    # construct Gamma_xc matrix
    gamma_xz = cme_w_xz.get_mean_embed(covars)["Gamma"]
    #shape = (n1_samples, n2_samples)

    # construct sigma
    sigma = hadamard_prod(mat_mul(mat_mul(gamma_xz.T, ker_ww),
                                          gamma_xz), ker_xx)

    if lam is None:
      lam = self.model_select(n_sample, ker_ww, ker_xx, gamma_xz, y)


    #print("rank of sigma", jnp.linalg.matrix_rank(sigma))
    f_mat = sigma + n_sample*lam*jnp.eye(n_sample)
    #print("f_mat is pd", is_pos_def(f_mat))

    t2 = time.time()

    #using linear solver

    if method == "original":
      print("use linear solver to estimate m0")
      vec_alpha = jsla.solve(f_mat, y)

    t25 = time.time()
    vec_alpha = stage2_weights(gamma_xz, vec_alpha)

    t3 = time.time()
    print(f"time: matrix preparation:{t2-t1} solve inverse:{t25-t2}, {t3-t25}")
    self.alpha = vec_alpha.reshape((-1, n_sample))
    #shape=(n1_sample, n2_sample)

  def model_select(self, n_sample, ker_ww, ker_xx, gamma_zx, y):

    if (n_sample >= 1000) or (ker_ww.shape[0]>1000):
      select_id = np.random.choice(n_sample,
                                  min(n_sample, 1000),
                                  replace=False)
      select_id2 = np.random.choice(ker_ww.shape[0],
                                    min(ker_ww.shape[0], 1000),
                                    replace=False)

      ker_sub_ww = ker_ww[select_id2,:]
      ker_sub_ww = ker_sub_ww[:, select_id2]

      ker_sub_xx = ker_xx[select_id, :]
      ker_sub_xx = ker_sub_xx[:, select_id]

      y_sub = y[select_id,...]

      gamma_sub_zx = gamma_zx[select_id2, :]
      gamma_sub_zx =gamma_sub_zx[:, select_id]

    else:
      ker_sub_ww = ker_ww
      ker_sub_xx = ker_xx
      y_sub = y
      gamma_sub_zx = gamma_zx

    sub_sigma =  hadamard_prod(mat_mul(mat_mul(gamma_sub_zx.T, ker_sub_ww),
                                               gamma_sub_zx), ker_sub_xx)
    d_t = modif_kron(mat_mul(ker_sub_ww, gamma_sub_zx), ker_sub_xx)
    mk_gamma_i=mat_trans(modif_kron(gamma_sub_zx, jnp.eye(1000)))
    lam, _ = cal_l_yw(d_t,
                      sub_sigma,
                      mk_gamma_i,
                      y_sub,
                      self.lam_min,
                      self.lam_max)
    print("selected lam of m0", lam)

    return lam

  def __call__(self, new_w, new_x, gamma_x):
    """return m0(w,c)
    Args:
        new_w: variable W, ndarray shape = (n3_samples, n1_features)
        new_x: variable X, ndarray shape = (n3_samples, n2_features)}
    Returns:
        m0(w,x): ndarray shape = (n3_samples)
    """
    # compute K_newWW

    ker_wneww = ker_mat(jnp.array(self.w), jnp.array(new_w),
                        kernel=self.kernel_dict["W"],
                        scale=self.w_sc) #(n1_sample, n3_sample)
    ker_wneww = mat_mul(ker_wneww, gamma_x)

    # compute K_newCC
    ker_xnewx = ker_mat(jnp.array(self.x), jnp.array(new_x),
                        kernel=self.kernel_dict["X"],
                        scale=self.sc) #(n2_sample, n3_sample)

    def h_wx(kx, kw):
      return jnp.dot(mat_mul(self.alpha, kx), kw)

    v = vmap(h_wx, (1,1))
    return v(ker_xnewx, ker_wneww)
    
  def get_exp_y_xz(self, covar, cme_w_xz):
    params = cme_w_xz.get_mean_embed(covar)
    gamma_xz = params["Gamma"]
    new_w = params["Y"]

    kxtalphakw = self(new_w, covar["X"], gamma_xz)
    return kxtalphakw



  def get_exp_y_x(self, new_x, cme_w_x):
    """ when computing E[Y|c,x]=<m0, phi(c) otimes mu_w|x,c>
    Args:
      new_x: ndarray shape=(n4_samples, n_features)
      cme_WC_x: ConditionalMeanEmbed
    """
    #print('alpha', self.alpha.shape)
    t1 = time.time()

    params = cme_w_x.get_mean_embed(new_x)
    #print('Gamma', params["Gamma"].shape)
    t2 = time.time()

    # params["Y"] shape=(n1_samples, w_features+c_features)
    new_w = params["Y"]
    # Gamma shape=(n1_samples, n4_samples)
    gamma_x = params["Gamma"]
    kxtalphakw = self(new_w, new_x["X"], gamma_x)
    t3 = time.time()
    #fn = lambda x: jnp.dot(kxTalphakw, x)
    #v = vmap(fn, (1))
    #print(kxTalphakw.shape)

    #result = v(params["Gamma"])
    t4 = time.time()

    print(f"inference time: {t2-t1}/{t3-t2}/{t4-t3}")
    return kxtalphakw


class BridgeM0CAT(BridgeM0):
  """ Construct the bridge function m0 for categorical z
  """

  def __init__(self,
               cme_w_xz_dict,
               covars,
               y,
               lam,
               scale=1.,
               method="original",
               lam_min=-4,
               lam_max=-1,
               kernel_dict=None):
    """Initiate the parameters
    Args:
      cme_w_xz_dict: dict, ConditionalMeanEmbed
      covars: covariates, dict {"C": ndarray shape=(n2_samples, n1_features), 
                                "X": ndarray shape=(n2_samples, n2_features)}
      Y: labels labels,  (n2_samples,)
      lam: reuglarization parameter, lam
      scale: kernel length scale, float
      method: approximation method, str
      'original' for linear solver, 'nystrom' for Nystrom approximation
      lam_min: minimum of lambda (log space) for hyperparameter tuning, float
      lam_max: maximum of lambda (log space) for hyperparameter tuning, float
      kernel_dict: specify kernel functions, dict
    """
    self.sc = scale

    if kernel_dict is None:
      kernel_dict = {}
      kernel_dict["X"] = "rbf"

    # concatenate Z
    z_label = cme_w_xz_dict.keys()
    #print(z_label)
    #print(type(z_label))
    m0_lookup = {}
    for z in z_label:

      idx = jnp.where(covars["Z"] == z)[0]
      y_z = y[idx,...]
      covarsz = {}
      covarsz["X"] = covars["X"][idx,...]


      cme_w_xz = cme_w_xz_dict[z]
      params = cme_w_xz.get_params()
      kernel_dict["W"] = params["kernel_dict"]["Y"]

      m0 = BridgeM0(cme_w_xz,
                    covarsz,
                    y_z,
                    lam,
                    kernel_dict = kernel_dict,
                    scale = self.sc,
                    method=method,
                    lam_min=lam_min,
                    lam_max=lam_max)

      m0_lookup[z] = m0

    self.kernel_dict = kernel_dict
    self.m0_lookup = m0_lookup
    self.z_label = z_label


  def __call__(self, new_w, new_x, gamma_x):
    """return m0(w,c)
    Args:
        new_w: variable W, ndarray shape = (n3_samples, n1_features)
        new_x: variable X, ndarray shape = (n3_samples, n2_features)}
        gamma_x: coefficients, ndarray
    Returns:
        m0(w,x): ndarray shape = (n3_samples)
    """
    output = []
    for z in self.z_label:
      output.append(self.m0_lookup[z](new_w, new_x, gamma_x))
    output = jnp.array(output).sum(axis=0)

    return output





class BridgeM0CLF(BridgeM0):
  """ Construct the bridge function m0 for classification
  """
  def __init__(self,
               cme_w_xz,
               covars,
               y,
               lam,
               scale=1.,
               method="original",
               lam_min=-4,
               lam_max=-1,
               kernel_dict=None):
    """Initiate the parameters
    Args:
      cme_w_xz: object, ConditionalMeanEmbed
      covars: covariates, dict {"C": ndarray shape=(n2_samples, n1_features),
                                 "X": ndarray shape=(n2_samples, n2_features)}
      y: labels, (n2_samples,)
      lam: reuglarization parameter, lam
      scale: kernel length scale, float
      method: approximation method, str
      'original' for linear solver, 'nystrom' for Nystrom approximation
      lam_min: minimum of lambda (log space) for hyperparameter tuning, float
      lam_max: maximum of lambda (log space) for hyperparameter tuning, float
      kernel_dict: specify kernel functions, dict
    """
    t1 = time.time()
    self.lam_min = lam_min
    self.lam_max = lam_max
    self.sc = scale
    n_sample = y.shape[0]

    # construct A matrix
    x = covars["X"]

    if kernel_dict is None:
      kernel_dict = {}
      kernel_dict["X"] = "rbf"

    ker_xx = ker_mat(jnp.array(x), jnp.array(x),
                    kernel=kernel_dict["X"],
                    scale=self.sc)
    self.x = x
    params = cme_w_xz.get_params()

    w = params["Y"]
    self.w_sc = params["scale"]
    self.w = w
    kernel_dict["W"] = params["kernel_dict"]["Y"]
    ker_ww = ker_mat(jnp.array(w), jnp.array(w),
                     kernel=kernel_dict["W"],
                     scale=params["scale"])
    self.kernel_dict = kernel_dict

    assert(set(params["Xlist"]) == set(covars.keys()))
    # construct Gamma_xc matrix
    gamma_xz = cme_w_xz.get_mean_embed(covars)["Gamma"]
    #shape = (n1_samples, n2_samples)

    # construct sigma
    sigma = hadamard_prod(mat_mul(mat_mul(gamma_xz.T, ker_ww), gamma_xz),
                                  ker_xx)

    if lam is None:
      lam = self.model_select(n_sample, ker_ww, ker_xx, gamma_xz, y)

    #print("rank of sigma", jnp.linalg.matrix_rank(sigma))
    f_mat = sigma + n_sample*lam*jnp.eye(n_sample)
    #print("F is pd", is_pos_def(F))

    t2 = time.time()
    #using linear solver

    print("use linear solver to estimate m0")
    vec_alpha = jsla.solve(f_mat, y)

    t25 = time.time()

    def fn(a):
      return stage2_weights(gamma_xz, a).reshape((-1, n_sample))
    parallel_stage2 = vmap(fn, (1))

    self.alpha = parallel_stage2(vec_alpha).transpose((1,2,0))

    t3 = time.time()
    print(f" time: matrix preparation:{t2-t1} solve inverse:{t25-t2}, {t3-t25}")

  def __call__(self, new_w, new_x, gamma_x):
    """return m0(w,c)
    Args:
        new_w: variable W, ndarray shape = (n3_samples, n1_features)
        new_x: variable X, ndarray shape = (n3_samples, n2_features)}
    Returns:
        m0(w,x): ndarray shape = (n3_samples)
    """
    # compute K_newWW

    ker_wneww = ker_mat(jnp.array(self.w), jnp.array(new_w),
                        kernel=self.kernel_dict["W"],
                        scale=self.w_sc) #(n1_sample, n3_sample)

    ker_wneww = mat_mul(ker_wneww, gamma_x)

    # compute K_newCC
    ker_xnewx = ker_mat(jnp.array(self.x), jnp.array(new_x),
                        kernel=self.kernel_dict["X"],
                        scale=self.sc) #(n2_sample, n3_sample)


    n_categories = self.alpha.shape[2]

    def h_wx(kx, kw):
      def fn(d):
        return jnp.dot(mat_mul(self.alpha[:,:, d], kx), kw)
      return vmap(fn)(jnp.arange(n_categories))

    outer_v = vmap(h_wx, (1,1))(ker_xnewx, ker_wneww)
    return outer_v #(n3_samples, n_category)




class BridgeM0CATCLF(BridgeM0CAT):
  """Construct the bridge function m0 for classification with discrete z
  """
  def __init__(self,
               cme_w_xz_dict,
               covars,
               y,
               lam,
               scale=1.,
               method="original",
               lam_min=-4,
               lam_max=-1,
               kernel_dict=None):
    """Initiate the parameters
    Args:
      cme_w_xz_dict: dict, ConditionalMeanEmbed
      covars: covariates, dict {"C": ndarray shape=(n2_samples, n1_features), 
                                "X": ndarray shape=(n2_samples, n2_features)}
      Y: labels labels,  (n2_samples,)
      lam: reuglarization parameter, lam
      scale: kernel length scale, float
      method: approximation method, str
      'original' for linear solver, 'nystrom' for Nystrom approximation
      lam_min: minimum of lambda (log space) for hyperparameter tuning, float
      lam_max: maximum of lambda (log space) for hyperparameter tuning, float
    """
    self.sc = scale

    if kernel_dict is None:
      kernel_dict = {}
      kernel_dict["X"] = "rbf"

    # concatenate Z
    z_label = cme_w_xz_dict.keys()
    #print(z_label)
    #print(type(z_label))
    m0_lookup = {}
    for z in z_label:
      idx = jnp.where(covars["Z"] == z)[0]

      y_z = y[idx,...]
      covarsz = {}
      covarsz["X"] = covars["X"][idx,...]


      cme_w_xz = cme_w_xz_dict[z]
      params = cme_w_xz.get_params()
      kernel_dict["W"] = params["kernel_dict"]["Y"]


      m0 = BridgeM0CLF(cme_w_xz,
                       covarsz,
                       y_z,
                       lam,
                       kernel_dict = kernel_dict,
                       scale = self.sc,
                       method=method,
                       lam_min=lam_min,
                       lam_max=lam_max)

      m0_lookup[z] = m0

    self.kernel_dict = kernel_dict
    self.m0_lookup = m0_lookup
    self.z_label = z_label
