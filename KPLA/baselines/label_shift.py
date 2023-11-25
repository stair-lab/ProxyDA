"""
implementation of label shift adaptation method.
"""
#Author: Nicole Chiou, Katherine Tsai<kt14@illinois.edu>
#MIT LICENSE



import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KernelDensity




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