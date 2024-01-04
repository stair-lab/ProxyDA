"""implements the bridge function h0"""
#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT

from KPLA.models.plain_kernel.kernel_utils import hadamard_prod, ker_mat, mat_mul, modif_kron, stage2_weights, mat_trans, cal_l_yw
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import time
from jax import vmap


class BridgeH0:
  """ Construct the bridge function h0.
  """
  def __init__(self,
              cme_w_xc,
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
      cme_w_xc: object, ConditionalMeanEmbed
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
    n_sample = y.shape[0]
    # construct A matrix
    c = covars["C"]

    self.lam_max = lam_max
    self.lam_min = lam_min

    if kernel_dict is None:
      kernel_dict = {}
      kernel_dict["C"] = "rbf"

    ker_cc = ker_mat(jnp.array(c), jnp.array(c),
                     kernel=kernel_dict["C"],
                     scale=self.sc)
    self.c = c
    params = cme_w_xc.get_params()
    w = params["Y"]
    self.w_sc = params["scale"]
    self.w = w
    kernel_dict["W"] = params["kernel_dict"]["Y"]
    ker_ww = ker_mat(jnp.array(w), jnp.array(w),
                     kernel=kernel_dict["W"],
                     scale=params["scale"])
    self.kernel_dict = kernel_dict

    assert(set(params["Xlist"]) == set(covars.keys()))
    # construct gamma_xc matrix
    #shape = (n1_samples, n2_samples)
    gamma_xc = cme_w_xc.get_mean_embed(covars)["Gamma"]

    # construct sigma
    sigma = hadamard_prod(mat_mul(mat_mul(gamma_xc.T, ker_ww),
                                  gamma_xc), ker_cc)

    if lam is None:
      #implement parameter selection
      #use random subsample
      lam = self.model_select(n_sample, ker_ww, ker_cc, gamma_xc, y)

    #print("rank of sigma", jnp.linalg.matrix_rank(Sigma))
    f_mat = sigma + n_sample*lam*jnp.eye(n_sample)
    #print("F is pd", is_pos_def(F))

    t2 = time.time()
    if method == "original":
      print("use linear solver to estimate h0")
      vec_alpha = jsla.solve(f_mat, y)
    t25 = time.time()

    vec_alpha = stage2_weights(gamma_xc, vec_alpha)

    t3 = time.time()

    print(f"time: matrix preparation:{t2-t1} solve inverse:{t25-t2}, {t3-t25}")
    self.alpha = vec_alpha.reshape((-1, n_sample)) #shape=(n1_sample, n2_sample)

  def model_select(self, n_sample, ker_ww, ker_cc, gamma_xc, y):
    """ model selection for lambda
      Args:
        n_sample: number of samples, int
        ker_ww: Gram matrix of W, ndarray
        ker_cc: Gram matrix of C, ndarray
        gamma_xc: coefficient matrix, ndarray
        y: response, ndarray
    """
    if (n_sample >= 1000) or (ker_ww.shape[0]>1000):
      select_id = np.random.choice(n_sample, min(n_sample, 1000), replace=False)
      select_id2 = np.random.choice(ker_ww.shape[0],
                                    min(ker_ww.shape[0], 1000),
                                    replace=False)

      ker_sub_ww = ker_ww[select_id2,:]
      ker_sub_ww = ker_sub_ww[:, select_id2]

      ker_sub_cc = ker_cc[select_id, :]
      ker_sub_cc = ker_sub_cc[:, select_id]

      y_sub = y[select_id,...]

      gamma_sub_xc = gamma_xc[select_id2, :]
      gamma_sub_xc = gamma_sub_xc[:, select_id]
    else:
      ker_sub_ww = ker_ww
      ker_sub_cc = ker_cc
      y_sub = y
      gamma_sub_xc = gamma_xc

    sub_sigma = hadamard_prod(mat_mul(mat_mul(gamma_sub_xc.T, ker_sub_ww),
                                            gamma_sub_xc), ker_sub_cc)

    d_t = modif_kron(mat_mul(ker_sub_ww, gamma_sub_xc), ker_sub_cc)

    mk_gamma_i = mat_trans(modif_kron(gamma_sub_xc, jnp.eye(1000)))

    lam, _ = cal_l_yw(d_t, sub_sigma, mk_gamma_i , y_sub,
                      self.lam_min, self.lam_max)

    print("selected lam of h_0:", lam)

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

    ker_wnew = ker_mat(jnp.array(self.w), jnp.array(new_w),
                      kernel=self.kernel_dict["W"],
                      scale=self.w_sc) #(n1_sample, n3_sample)


    # compute K_newCC
    ker_cnewc = ker_mat(jnp.array(self.c), jnp.array(new_c),
                      kernel=self.kernel_dict["C"],
                      scale=self.sc) #(n2_sample, n3_sample)

    print(ker_wnew.shape, ker_cnewc.shape, self.alpha.shape)

    def h_wc(kc, kw):
      return jnp.dot(mat_mul(self.alpha, kc), kw)
    v = vmap(h_wc, (1,1))
    return v(ker_cnewc, ker_wnew)

  def get_exp_y_x(self, new_x, cme_wc_x):
    """ computing E[Y|c,x]=<h0, phi(c) otimes mu_w|x,c>
    Args:
      new_x: ndarray shape=(n4_samples, n_features)
      cme_wc_x: ConditionalMeanEmbed
    """

    t1 = time.time()
    params = cme_wc_x.get_mean_embed(new_x)
    t2 = time.time()
    if len(self.w.shape) == 1:
      w_features = 1
    else:
      w_features = self.w.shape[1]

    if len(self.c.shape) == 1:
      c_features = 1
    else:
      c_features = self.c.shape[1]

    # params["Y"] shape=(n1_samples, w_features+c_features)
    new_w = params["Y"][:, 0:w_features]
    new_c = params["Y"][:, w_features:w_features+c_features]
    # Gamma shape=(n1_samples, n4_samples)
    kctalphakw = self(new_w, new_c)
    t3 = time.time()

    def fn(x):
      return jnp.dot(kctalphakw, x)
    v = vmap(fn, (1))

    result = v(params["Gamma"])
    t4 = time.time()

    print(f"inference time: {t2-t1}/{t3-t2}/{t4-t3}")
    return result


class BridgeH0CLF(BridgeH0):
  """ Construct the bridge function h0 for classification.
  """
  def __init__(self,
               cme_w_xc,
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
      cme_w_xc: object, ConditionalMeanEmbed
      covars: covariates, dict {"C": ndarray shape=(n2_samples, n1_features),
                                "X": ndarray shape=(n2_samples, n2_features)}
      Y: labels, (n2_samples,)
      lam: reuglarization parameter, lam
      scale: kernel length scale, float
      method: approximation method, str
      lam_min: minimum of lambda (log space) for hyperparameter tuning, float
      lam_max: maximum of lambda (log space) for hyperparameter tuning, float
      kernel_dict: specify kernel functions, dict
    """
    t1 = time.time()
    self.sc = scale
    n_sample = y.shape[0]
    # construct A matrix
    c = covars["C"]

    self.lam_max = lam_max
    self.lam_min = lam_min

    if kernel_dict is None:
      kernel_dict = {}
      kernel_dict["C"] = "rbf"

    ker_cc = ker_mat(jnp.array(c), jnp.array(c),
                     kernel=kernel_dict["C"],
                     scale=self.sc)
    self.c = c
    params = cme_w_xc.get_params()
    w = params["Y"]
    self.w_sc = params["scale"]
    self.w = w
    kernel_dict["W"] = params["kernel_dict"]["Y"]
    ker_ww = ker_mat(jnp.array(w), jnp.array(w),
                     kernel=kernel_dict["W"],
                     scale=params["scale"])

    self.kernel_dict = kernel_dict

    assert(set(params["Xlist"]) == set(covars.keys()))
    # construct gamma_xc matrix
    gamma_xc = cme_w_xc.get_mean_embed(covars)["Gamma"]
    #shape = (n1_samples, n2_samples)

    # construct sigma
    sigma = hadamard_prod(mat_mul(mat_mul(gamma_xc.T, ker_ww), gamma_xc),
                          ker_cc)


    if lam is None:
      #implement parameter selection
      #use random subsample
      lam = self.model_select(n_sample, ker_ww, ker_cc, gamma_xc, y)

    f_mat = sigma + n_sample*lam*jnp.eye(n_sample)
    t2 = time.time()

    #using linear solver

    print("use linear solver to estimate h0")
    vec_alpha = jsla.solve(f_mat, y)
    t25 = time.time()

    def fn(a):
      return stage2_weights(gamma_xc, a).reshape((-1, n_sample))
    parallel_stage2 = vmap(fn, (1))
    self.alpha = parallel_stage2(vec_alpha).transpose((1,2,0))


    t3 = time.time()
    print(f"time: matrix preparation:{t2-t1} solve inverse:{t25-t2}, {t3-t25}")

  def __call__(self, new_w, new_c):
    """return h0(w,c)
    Args:
        new_w: variable W, ndarray shape = (n3_samples, n1_features)
        new_c: variable C, ndarray shape = (n3_samples, n2_features)}
    Returns:
        h0(w,c): ndarray shape = (n3_samples)
    """
    # compute ker_wneww
    ker_wneww = ker_mat(jnp.array(self.w), jnp.array(new_w),
                      kernel=self.kernel_dict["W"],
                      scale=self.w_sc) #(n1_sample, n3_sample)


    # compute ker_cnewc
    ker_cnewc = ker_mat(jnp.array(self.c), jnp.array(new_c),
                      kernel=self.kernel_dict["C"],
                      scale=self.sc) #(n2_sample, n3_sample)

    n_categories = self.alpha.shape[2]

    def h_wc(kc,kw):
      def temp_fun(d):
        return jnp.dot(mat_mul(self.alpha[:,:, d], kc), kw)
      return vmap(temp_fun)(jnp.arange(n_categories))

    outer_v = vmap(h_wc, (1,1))(ker_cnewc, ker_wneww)
    return outer_v #(n3_samples, outer)

  def get_exp_y_x(self, new_x, cme_wc_x):
    """ when computing E[Y|c,x]=<h0, phi(c) otimes mu_w|x,c>
    Args:
      new_x: ndarray shape=(n4_samples, n_features)
      cme_wc_x: ConditionalMeanEmbed
    """

    t1 = time.time()
    params = cme_wc_x.get_mean_embed(new_x)
    t2 = time.time()
    if len(self.w.shape) == 1:
      w_features = 1
    else:
      w_features = self.w.shape[1]

    if len(self.c.shape) == 1:
      c_features = 1
    else:
      c_features = self.c.shape[1]

    # params["Y"] shape=(n1_samples, w_features+c_features)
    new_w = params["Y"][:, 0:w_features]
    new_c = params["Y"][:, w_features:w_features+c_features]
    # Gamma shape=(n1_samples, n4_samples)
    kctalphakw = self(new_w, new_c)
    t3 = time.time()
    #fn = lambda x: jnp.dot(kcTalphakw, x)
    #v = vmap(fn, (1))

    result = mat_mul(kctalphakw.T, params["Gamma"]).T
    #(n4_samples, n_categories)
    t4 = time.time()

    print(f"inference time: {t2-t1}/{t3-t2}/{t4-t3}")
    return result
