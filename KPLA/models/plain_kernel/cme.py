from .kernel_utils import *
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import scipy


class ConditionalMeanEmbed:
  """function class of conditional mean embedding
    C(Y|X) = Phi_Y(K_XX+lam*n1_samples*I)^{-1}Phi_X
    mu(Y|x) = C(Y|x=x) = Phi_Y(K_XX+lam*n1_samples*I)^{-1}Phi_X(x)
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
  def __init__(self, Y, X, lam, scale=1, method='original', q=None, lam_min=-4, lam_max=-1, kernel_dict=None):
    """ initiate the parameters
      Args:
        Y: dependent variables, ndarray shape=(n1_samples, n2_features)
        X: independent varaibles, dict {"Xi": ndarray shape=(n1_samples, n1_features)}
        lam: regularization parameter
        scale: kernel length scale
        method: approximation method, str
        'orginal' for linear solver, 'nystrom' for  Nystrom approximation
        kernel_dict: Dictionary of kernel_function, dictionary keys are the variable name
        q: number of components to sample if NYstrom approximation is used, int
        lam_min: minimum of lambda (log space) for hyperparameter tuning, float
        lam_min: maximum of lambda (log space) for hyperparameter tuning, float
    """
    self.n_samples = Y.shape[0]
    self.X_list = list(X.keys())

    self.X = X
    self.Y = Y
    #assert(lam >= 0.)
    self.lam = lam
    self.sc = scale
    self.method = method
    #check method is correctly specified
    if self.method != 'original':
      if self.method != 'nystrom':
        raise Exception("method specified not implemented please select again")

    #set the kernel functions, default is rbf kernel
    if kernel_dict == None:
      kernel_dict = {}
      kernel_dict['Y'] = 'rbf'
      for key in self.X_list:
        kernel_dict[key] = 'rbf'
    self.kernel_dict = kernel_dict
    # construct of gram matrix
    K_XX = jnp.ones((self.n_samples, self.n_samples))
    for key in self.X_list:
      x = X[key]
      temp = ker_mat(jnp.array(x), jnp.array(x),  kernel=self.kernel_dict[key], scale=self.sc)

      K_XX= Hadamard_prod(K_XX, temp)
    self.K_XX = K_XX




    #select lambda
    if (self.lam == None):
      K_YY = ker_mat(jnp.array(self.Y), jnp.array(self.Y), kernel=self.kernel_dict['Y'], scale=self.sc)
      scale_dict = {}
      l_w, loo1 = cal_l_w(K_XX, K_YY, low=lam_min, high=lam_max, n=10)
      print('selected lam of cme:', l_w)
      self.lam = l_w


    #compute Nystrom approximation
    if self.method=='nystrom':
      if q == None:
      
        q = min(250, self.n_samples)
      if q < self.n_samples:
        select_x = np.random.choice(self.n_samples, q, replace=False)

      else:
        select_x = np.arange(self.n_samples)
        #reorder_x = np.arange(self.n_samples)
      K_q = self.K_XX[select_x, :][:, select_x]
      K_nq = self.K_XX[:, select_x]


      inv_Kq_sqrt =  jnp.array(truncate_sqrtinv(K_q))
      Q = mat_mul(K_nq, inv_Kq_sqrt)

      
      inv_temp = jsla.solve(self.lam*self.n_samples*jnp.eye(q)+Q.T.dot(Q), jnp.eye(q))
      if jnp.isnan(inv_temp).any():
        print("inv_temp is nan")         
      self.aprox_K_XX = (jnp.eye(self.n_samples)-(Q.dot(inv_temp)).dot(Q.T))/(self.lam*self.n_samples)


    elif self.method == 'original':
      Gx = self.K_XX + self.lam*self.n_samples*jnp.eye(self.n_samples)
      inv_Gx = jsla.solve(Gx, jnp.eye(self.n_samples), assume_a='pos')
      self.inv_Gx = inv_Gx      
      

  def get_params(self):
    """Return parameters.
    """
    Gx = self.K_XX + self.lam*self.n_samples*jnp.eye(self.n_samples)

    #K_YY = ker_mat(jnp.array(self.Y), jnp.array(self.Y), self.sc)
    out_dict = {"GramX": Gx, "Y":self.Y, "X":self.X, "Xlist":self.X_list, "scale":self.sc, 'kernel_dict':self.kernel_dict}
    return out_dict
  

  def get_mean_embed(self, new_x):
    """ compute the mean embedding given new_x C(Y|new_x)
      Args:
        new_x: independent varaibles, dict {"Xi": ndarray shape=(n2_samples, n1_features)}
      Returns:
    """
    
    Gx = self.K_XX + self.lam*self.n_samples*jnp.eye(self.n_samples)
    n2_samples = new_x[self.X_list[0]].shape[0]

    Phi_Xnx = jnp.ones((self.n_samples, n2_samples))

    for key in self.X_list:
      temp = ker_mat(jnp.array(self.X[key]), jnp.array(new_x[key]), kernel=self.kernel_dict[key], scale=self.sc)
      Phi_Xnx = Hadamard_prod(Phi_Xnx, temp)
    
    

    # use Nystrom approximation
    if self.method == 'nystrom':
      # print('use Nystrom method to estimate cme')
      Gamma = self.aprox_K_XX.dot(Phi_Xnx)

    elif self.method == 'original':
      Gamma = mat_mul(self.inv_Gx, Phi_Xnx)
    

    evaluate = False
    if evaluate:
      Gx = self.K_XX + self.lam*self.n_samples*jnp.eye(self.n_samples)
      inv_Gx = jsla.solve(Gx, jnp.eye(self.n_samples), assume_a='pos')
      Gamma2 = inv_Gx.dot(Phi_Xnx)
      print('difference of Gamma', jnp.linalg.norm(Gamma-Gamma2))
      
    
    return {"Y": self.Y, "Gamma": Gamma, "scale": self.sc} # jnp.dot(kernel(Y,y; sc), Gamma)

  def __call__(self, new_y, new_x):
    """
      Args:
        new_y: dependent variables, ndarray shape=(n3_samples, n2_features)
        new_x: independent varaibles, dict {"Xi": ndarray shape=(n2_samples, n1_features)}
      Returns:
        out: ndarray shape=(n3_samples, n2_samples)
    """
    memb_nx = self.get_mean_embed(new_x)
    Gamma = memb_nx["Gamma"]
    Phi_Yny = ker_mat(jnp.array(new_y), jnp.array(self.Y), kernel=self.kernel_dict['Y'], scale=self.sc)


    return mat_mul(Phi_Yny, Gamma)

  def get_coefs(self, new_x):

    memb_nx = self.get_mean_embed(new_x)
    Gamma = memb_nx["Gamma"]
    return Gamma
