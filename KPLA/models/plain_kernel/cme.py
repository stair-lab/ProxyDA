"""
Implementation of conditional mean embedding
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT

from KPLA.models.plain_kernel.kernel_utils import ker_mat, hadamard_prod, cal_l_w, mat_mul
import jax.numpy as jnp
import jax.scipy.linalg as jsla



class ConditionalMeanEmbed:
  """function class of conditional mean embedding
    C(Y|X) = Phi_Y(ker_xx+lam*n1_samples*I)^{-1}Phi_X
    mu(Y|x) = C(Y|x=x) = Phi_Y(ker_xx+lam*n1_samples*I)^{-1}Phi_X(x)
    E[phi(Y,y)|X=x] = <y, mu(Y|x)>

    Example:
    X  = {}
    n1_samples = 50
    X["X1"] = jax.random.normal(key, shape=(n1_samples,))
    X["X2"] = jax.random.normal(key, shape=(n1_samples, 2))
    Y = jax.random.normal(key2, shape=(n1_samples,))
    C_YX = ConditionalMeanEmbed(Y, X, 0.1)

    new_x = {}
    n2_samples = 5
    new_x["X1"] = jax.random.normal(key, shape=(n2_samples,))
    new_x["X2"] = jax.random.normal(key, shape=(n2_samples, 2))
    n3_samples = 20
    new_y = jax.random.normal(key2, shape=(n3_samples,))
    C_YX(new_y, new_x)
  """
  def __init__(self,
               y,
               x,
               lam,
               scale=1,
               method="original",
               lam_min=-4,
               lam_max=-1,
               kernel_dict=None):
    """ initiate the parameters
      Args:
        Y: dependent variables, ndarray shape=(n1_samples, n2_features)
        X: independent varaibles, 
        dict {"Xi": ndarray shape=(n1_samples, n1_features)}
        lam: regularization parameter
        scale: kernel length scale
        method: approximation method, "orginal" for linear solver
        kernel_dict: Dictionary of kernel_function, 
        dictionary keys are the variable name
        lam_min: minimum of lambda (log space) for hyperparameter tuning, float
        lam_min: maximum of lambda (log space) for hyperparameter tuning, float
    """
    self.n_samples = y.shape[0]
    self.x_list = list(x.keys())

    self.x = x
    self.y = y
    #assert(lam >= 0.)
    self.lam = lam
    self.sc = scale
    self.method = method

    #set the kernel functions, default is rbf kernel
    if kernel_dict is None:
      kernel_dict = {}
      kernel_dict["Y"] = "rbf"
      for key in self.x_list:
        kernel_dict[key] = "rbf"
    self.kernel_dict = kernel_dict
    # construct of gram matrix
    ker_xx = jnp.ones((self.n_samples, self.n_samples))
    for key in self.x_list:
      x_idx = x[key]
      temp = ker_mat(jnp.array(x_idx), jnp.array(x_idx),
                     kernel=self.kernel_dict[key],
                     scale=self.sc)

      ker_xx= hadamard_prod(ker_xx, temp)
    self.ker_xx = ker_xx

    #select lambda
    if self.lam is None:
      ker_yy = ker_mat(jnp.array(self.y), jnp.array(self.y),
                     kernel=self.kernel_dict["Y"],
                     scale=self.sc)
      l_w, _ = cal_l_w(ker_xx, ker_yy, low=lam_min, high=lam_max, n=10)
      print("selected lam of cme:", l_w)
      self.lam = l_w


    if self.method == "original":
      gx = self.ker_xx + self.lam*self.n_samples*jnp.eye(self.n_samples)
      inv_gx = jsla.solve(gx, jnp.eye(self.n_samples), assume_a="pos")
      self.inv_gx = inv_gx


  def get_params(self):
    """Return parameters.
    """
    gx = self.ker_xx + self.lam*self.n_samples*jnp.eye(self.n_samples)

    out_dict = {"GramX": gx,
                "Y":self.y,
                "X":self.x,
                "Xlist":self.x_list,
                "scale":self.sc,
                "kernel_dict":self.kernel_dict}
    return out_dict

  def get_mean_embed(self, new_x):
    """ compute the mean embedding given new_x C(Y|new_x)
      Args:
        new_x: independent varaibles, 
        dict {"Xi": ndarray shape=(n2_samples, n1_features)}
      Returns:
    """

    #gx = self.ker_xx + self.lam*self.n_samples*jnp.eye(self.n_samples)
    n2_samples = new_x[self.x_list[0]].shape[0]

    phi_xnx = jnp.ones((self.n_samples, n2_samples))

    for key in self.x_list:
      temp = ker_mat(jnp.array(self.x[key]), jnp.array(new_x[key]),
                     kernel=self.kernel_dict[key],
                     scale=self.sc)
      phi_xnx = hadamard_prod(phi_xnx, temp)

    if self.method == "original":
      gamma = mat_mul(self.inv_gx, phi_xnx)


    #evaluate = False
    #if evaluate:
    #  gx = self.ker_xx + self.lam*self.n_samples*jnp.eye(self.n_samples)
    #  inv_Gx = jsla.solve(gx, jnp.eye(self.n_samples), assume_a="pos")
    #  gamma2 = inv_Gx.dot(phi_xnx)
    #  print("difference of gamma", jnp.linalg.norm(gamma-gamma2))

    return {"Y": self.y,
            "Gamma": gamma,
            "scale": self.sc} # jnp.dot(kernel(Y,y; sc), gamma)

  def __call__(self, new_y, new_x):
    """
      Args:
        new_y: dependent variables, ndarray shape=(n3_samples, n2_features)
        new_x: independent varaibles,
        dict {"Xi": ndarray shape=(n2_samples, n1_features)}
      Returns:
        out: ndarray shape=(n3_samples, n2_samples)
    """
    memb_nx = self.get_mean_embed(new_x)
    gamma = memb_nx["Gamma"]
    phi_yny = ker_mat(jnp.array(new_y), jnp.array(self.y),
                      kernel=self.kernel_dict["Y"],
                      scale=self.sc)


    return mat_mul(phi_yny, gamma)

  def get_coefs(self, new_x):

    memb_nx = self.get_mean_embed(new_x)
    gamma = memb_nx["Gamma"]
    return gamma
