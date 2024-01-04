"""
implements simple baselines
"""
#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor




class MultiSourceCat:
  """concatenate data across domains
  """
  def __init__(self, max_iter=300):
    self.classifier = MLPClassifier(random_state=1, max_iter=max_iter)

  def fit(self, source_data):

    cat_train_x = np.array(source_data[0]['X'])
    cat_train_y = np.array(source_data[0]['Y'])
    long_y = list(np.unique(cat_train_y))

    for _, train_data in enumerate(source_data[1::]):

      x_train = np.array(train_data['X'])
      y_train = np.array(train_data['Y'])
      cat_train_x = np.concatenate((cat_train_x, x_train))
      cat_train_y = np.concatenate((cat_train_y, y_train))

      long_y += list(np.unique(y_train))

    self.classifier.fit(cat_train_x, cat_train_y)

    self.n_labels_ =  list(set(long_y))

    return self

  def predict(self, new_x):

    return self.classifier.predict(new_x)

  def predict_proba(self, new_x):

    return self.classifier.predict_proba(new_x)


class MultiSourceCatReg:
  """concatenate data across domains, MLP regressor
  """
  def __init__(self, max_iter=300):
    self.regressor = MLPRegressor(random_state=1, max_iter=max_iter)

  def fit(self, source_data):

    cat_train_x = np.array(source_data[0]['X'])
    cat_train_y = np.array(source_data[0]['Y'])
    long_y = list(np.unique(cat_train_y))

    for _, train_data in enumerate(source_data[1::]):

      x_train = np.array(train_data['X'])
      y_train = np.array(train_data['Y'])
      cat_train_x = np.concatenate((cat_train_x, x_train))
      cat_train_y = np.concatenate((cat_train_y, y_train))

      long_y += list(np.unique(y_train))

    self.regressor.fit(cat_train_x, cat_train_y)

    self.n_labels_ =  list(set(long_y))

    return self

  def predict(self, new_x):

    return self.regressor.predict(new_x)

