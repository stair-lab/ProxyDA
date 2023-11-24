"""
implements multi-source SVM
Blanchard, G., Lee, G., & Scott, C. (2011). 
Generalizing from several related classification
tasks to a new unlabeled sample. 
Advances in neural information processing systems, 24.
"""
#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT

import numpy as np
from sklearn.linear_model import SGDClassifier


class MultiSourceMK:
  """
  Blanchard, G., Lee, G., & Scott, C. (2011). 
  Generalizing from several related classification 
  tasks to a new unlabeled sample. 
  Advances in neural information processing systems, 24.
  """

  def __init__(self, p_kernel, x_kernel, max_iter=300):
    #self.svc = make_pipeline(StandardScaler(), LinearSVC())
    self.svc = SGDClassifier(max_iter=max_iter, tol=1e-3, loss='hinge')
    self.p_kernel = p_kernel #kernel to compute the probability distance
    self.x_kernel = x_kernel

  def _compute_pdist(self, source_x_i, target_x):
    ker = self.p_kernel(source_x_i, target_x)
    d_source_i_target = np.mean(ker)
    return d_source_i_target

  def fit(self, source_data, target_data):

    self.n_env = len(source_data)
    dist_weight = np.zeros(len(source_data))

    for j, source_d in enumerate(source_data):
      dist_weight[j] = self._compute_pdist(np.array(source_d['X']),
                                           np.array(target_data['X']))

    self.dist_weight_ = dist_weight

    n_size      = [len(d['X']) for d in source_data]
    self.n_size_ = n_size

    big_feature = np.ones((sum(n_size), sum(n_size)))
    weights     = np.ones(sum(n_size))

    for i in range(self.n_env):
      if i == 0 :
        big_x = np.array(source_data[i]['X'])
        big_y = np.array(source_data[i]['Y'])

      else:
        big_x = np.concatenate((big_x, np.array(source_data[i]['X'])))
        big_y = np.concatenate((big_y, np.array(source_data[i]['Y'])))
      for j in range(i, self.n_env):
        w = self._compute_pdist(np.array(source_data[i]['X']),
                                np.array(source_data[j]['X']))

        start_x = sum(n_size[0:i])
        start_y = sum(n_size[0:j])

        len_i = n_size[i]
        len_j = n_size[j]

        ker_xx = self.x_kernel(np.array(source_data[i]['X']),
                            np.array(source_data[j]['X']))

        big_feature[start_x:start_x+len_i, start_y:start_y+len_j] = ker_xx*w
        big_feature[start_y:start_y+len_j, start_x:start_x+len_i] = (ker_xx.T)*w
        weights[start_x:start_x+len_i] = self.dist_weight_[i]

    self.x_ = big_x
    self.weights_ = weights

    self.svc.fit(big_feature*self.weights_[np.newaxis,:], big_y)
    #fit the svm model

  def transform_feature_x(self, xnew):
    ker_newxx = self.x_kernel(xnew, self.x_)

    return ker_newxx*self.weights_[np.newaxis,:]

  def predict(self, xnew):
    #create feature map
    weight_ker_newxx = self.transform_feature_x(np.array(xnew))
    predicty = self.svc.predict(weight_ker_newxx)

    return predicty

  def decision(self, xnew):
    """Min-max scale output of `decision_function` to [0, 1]."""

    weight_ker_newxx = self.transform_feature_x(np.array(xnew))
    decision_x = self.svc.decision_function(weight_ker_newxx)

    return decision_x
