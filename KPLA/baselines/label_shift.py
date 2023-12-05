"""
implementation of label shift adaptation method.
"""
#Author: Nicole Chiou, Katherine Tsai<kt14@illinois.edu>
#MIT LICENSE



import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
import cvxpy as cp
import numpy as np
import scipy

class ConLABEL:
  """
  implement continuous label shift

  Zhang, K., Schölkopf, B., Muandet, K., & Wang, Z. (2013, May). 
  Domain adaptation under target and conditional shift. 
  In International conference on machine learning (pp. 819-827). PMLR.
  """
  def __init__(self, lam, bp, alpha, kernel, kernel2):
    self.lam = lam
    self.bp = bp
    self.kernel = kernel
    self.label_model = KernelRidge(alpha=alpha, kernel=kernel2)
  
  def fit(self, source_data, target_x):

    # learn the weight
    K_Y = self.kernel(source_data['Y'], source_data['Y'])
    
    K_X = self.kernel(source_data['X'], source_data['X'])
    m1 = K_X.shape[0]

    inv_KY = scipy.linalg.solve(K_Y+self.lam*np.eye(m1), np.eye(m1))
    #lr = LinearRegression()
    #lr.fit(K_Y+self.lam*np.eye(m1), np.eye(m1))
    #np.allclose(lr.coe)

    inv_KY_KY = np.einsum('ij,jk->ik', inv_KY, K_Y)

    K_X1X2 = self.kernel(source_data['X'], target_x)
    m2 = K_X1X2.shape[1]
    A = inv_KY_KY.T @ K_X @ inv_KY_KY
    
    B = K_X1X2.T @ inv_KY_KY
    B = (m1/m2)*B
    B = np.sum(B, axis=0)
    G = np.eye(m1)
    G2 = -np.eye(m1)
    x = cp.Variable(m1)
    h = self.bp*np.ones(m1)
    h2 = np.zeros(m1)
    C = np.ones(m1)
    eps = self.bp*np.sqrt(m1)/4
    print('start fitting')
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, A)+B.T@x),
                    [G@x <= h,
                     G2@x <= h2,
                     C@x <= eps + m1,
                     - C@x <= -m1 + eps
                    ]
                    )

    prob.solve()
    print('finished')
    print(f'Problem status {prob.status}')
    beta = np.array(x.value)
    idx = np.where(beta < 0)[0]
    beta[idx] = 0
    beta = normalize(beta[:,np.newaxis]).squeeze()
    self.label_model.fit(source_data['X'],
                                source_data['Y'],
                                sample_weight=beta)
                                
  def predict(self, test_data):
    return self.label_model.predict(test_data)



class LABEL:
  """
  implementation of label shift adaptation method.
  """
  def __init__(self, alpha, kernel='rbf', kernel2='gaussian', bandwidth=1):
    self.source_kde = KernelDensity(kernel=kernel2, bandwidth=bandwidth)
    self.target_kde = KernelDensity(kernel=kernel2, bandwidth=bandwidth)

    self.source_label_model = KernelRidge(alpha=alpha, kernel=kernel)
    self.targegt_label_model = KernelRidge(alpha=alpha, kernel=kernel)



  def fit(self, source_train, source_val, target_train):
    self.source_kde.fit(source_val['Y'])
    self.target_kde.fit(target_train['Y'])

    # Compute sample weights q(Y)/p(Y)
    log_q_y = self.target_kde.score_samples(source_train['Y'])
    log_p_y = self.source_kde.score_samples(source_train['Y'])

    source_sample_weight_train = np.exp(log_q_y - log_p_y)

    #fit source model
    self.source_label_model.fit(source_train['X'],
                                source_train['Y'],
                                sample_weight=source_sample_weight_train)

    # Compute sample weights p(Y)/q(Y)
    target_sample_weight_train = np.exp(log_p_y - log_q_y)
    #fit target model
    self.target_label_model.fit(target_train['X'],
                                target_train['Y'],
                                sample_weight=target_sample_weight_train)

  def predict(self, test_data):
    return self.source_label_model.predict(test_data['X'])

  def predict_target(self, test_data):
    return self.target_label_model.predict(test_data['X'])