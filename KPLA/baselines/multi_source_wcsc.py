"""
implements multi-source MMD
Zhang, K., Gong, M., & Schölkopf, B.
Multi-source domain adaptation: A causal view.
In Proceedings of the AAAI Conference on Artificial Intelligence.
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT


import numpy as np
from sklearn.neighbors import KernelDensity
from cvxopt import matrix
from cvxopt import solvers


class MultiSourceMMD:
  """
  implements multi-source MMD
  Zhang, K., Gong, M., & Schölkopf, B.
  Multi-source domain adaptation: A causal view.
  In Proceedings of the AAAI Conference on Artificial Intelligence.
  """
  def __init__(self, source_data, kernel, kde_kernel, bandwidth=1.0):

    self.kernel = kernel # kernel for computing MMD
    self.kde_kernel = kde_kernel # kernel for KDE
    self.bandwidth = bandwidth
    self.x = [np.asarray(d['X']) for d in source_data]
    self.y = [np.asarray(d['Y']) for d in source_data]

    self.n_env = len(source_data)
    longy = []
    for y in self.y:
      longy += list(np.unique(y))
    self.labels = list(set(longy))
    self.n_label = len(self.labels)
    self.count_table = np.zeros((self.n_env, self.n_label))

    #print('label set', self.labels)
    a_mat = np.zeros((self.n_env*self.n_label, self.n_env*self.n_label))

    def element_a(i, j, k, m):
      xi_k, n_ik = self._get_x_by_label(i, k)
      xj_m, n_jm = self._get_x_by_label(j, m)

      k_ij_km = self.kernel(xi_k, xj_m)

      #update the count table
      if self.count_table[i, k] == 0.:
        self.count_table[i, k] = n_ik
      if self.count_table[j, m] == 0.:
        self.count_table[j, m] = n_jm

      return np.sum(k_ij_km) / (n_ik*n_jm)


    # construct a_mat
    for i in range(self.n_env):
      for j in range(self.n_env):
        for k in range(self.n_label):
          for m in range(self.n_label):
            a_mat[i*self.n_label+k, j*self.n_label+m] = element_a(i, j, k, m)

    #f1 = lambda i,j: np.vectorize(lambda k,m: element_a(i, j, k, m))(np.arange)

    print('construct a_mat')
    self.a_mat = a_mat

    # construct the density esimator of P_Xi_y
    kde_x_y = []
    for k in range(self.n_label):
      tmp_kde_x_y = self._get_kde_x_y(k) #y= self.labels[k]
      kde_x_y.append(tmp_kde_x_y)
    print('construct KDE')
    self.kde_x_y = kde_x_y



  def _get_kde_x_y(self, k):
    px_y = []
    for i in range(self.n_env):
      xi_k, _ = self._get_x_by_label(i, k)
      kde = KernelDensity(bandwidth=self.bandwidth,
                          kernel=self.kde_kernel).fit(xi_k)

      px_y.append(kde) #y=self.labels[k]

    return px_y #len = n_env

  def _get_pdf_kde_x_y(self, k, x_new):
    pdf_list = []
    for kde in self.kde_x_y[k]:
      log_pdf = kde.score_samples(x_new)
      pdf_list.append(np.exp(log_pdf))
    return np.array(pdf_list)



  def _get_x_by_label(self, i, k):
    label_i = self.labels[k]
    loc_i = np.where(self.y[i] == label_i)[0]
    n_ik = loc_i.size
    xi_k  = self.x[i][loc_i,...]
    return xi_k, n_ik

  def _get_beta(self, b):
    c_mat = matrix(np.ones(self.n_env*self.n_label)).T
    d =  matrix(np.ones(1))
    g_mat = matrix(-np.eye(self.n_env*self.n_label)) #beta is positive
    h = matrix(np.zeros(self.n_env*self.n_label))#beta is positive
    p_mat = matrix(self.a_mat)
    q_mat = matrix(b)
    sol = solvers.qp(p_mat, q_mat, g_mat, h, A=c_mat, b=d)
    print('solve beta status:', sol['status'])
    return np.array(sol['x'])

  def fit(self, target_x):
    """fit target domain data
    Args:
        newX: nddarry
    """
    # construct b
    b = np.zeros(self.n_env*self.n_label)
    n_new = target_x.shape[0]

    def element_b(i, k):
      xi_k, n_ik = self._get_x_by_label(i, k)
      k_inew_k = self.kernel(xi_k, target_x)

      return np.sum(k_inew_k)/(n_new*n_ik)

    for i in range(self.n_env):
      for k in range(self.n_label):
        b[i*self.n_label+k] = element_b(i,k)

    # get beta using cvx
    beta = self._get_beta(b)

    beta = beta.reshape((self.n_env, self.n_label), order='F')

    py_new = np.zeros(self.n_label)

    for j in range(self.n_label):
      py_new[j] = np.sum(beta[:,j])

    alpha = np.zeros((self.n_env, self.n_label))
    for i in range(self.n_env):
      for j in range(self.n_label):
        alpha[i, j] = beta[i, j]/py_new[j]

    self.py_target_ = py_new
    self.alpha_ = alpha

class MuiltiSourceCombCLF(MultiSourceMMD):
  """
  implements multi-source MMD
  Zhang, K., Gong, M., & Schölkopf, B.
  Multi-source domain adaptation: A causal view.
  In Proceedings of the AAAI Conference on Artificial Intelligence.
  """
  def predict(self, x):
    """
    predict Y from x
    """
    out_prob = np.zeros((x.shape[0], self.n_label))
    #print(out_prob.shape)
    for j in range(self.n_label):
      #print(self._get_pdf_kde_x_y(j, x).shape)
      tmp = self._get_pdf_kde_x_y(j, x)*self.alpha_[:,j][:,np.newaxis]
      tmp = np.sum(tmp, axis=0)
      p = self.py_target_[j] * tmp
      #print('p', p.shape)
      out_prob[:, j] = p

    # normalize probability
    out_prob /= np.sum(out_prob, axis=1)[:,np.newaxis]
    return out_prob

