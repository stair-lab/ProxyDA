"""implements the full adaptation pipeline"""
#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT

import pandas as pd
import numpy as np
import jax.numpy as jnp
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

from .method import KernelMethod, split_data_widx
from .cme import ConditionalMeanEmbed
from .bridge_h0 import BridgeH0, BridgeH0CLF
from .kernel_utils import flatten


class FullAdapt(KernelMethod):
  """
  Adaptation setting: observe (W,X,Y,C) from the source,
  (W,X,C) from the target
  """
  def split_data(self):
    #split training data
    n = self.source_train['X'].shape[0]
    index = np.random.RandomState(seed=42).permutation(n)
    split_id = np.split(index, [int(.33*n), int(.67*n)])
    train_list = []
    for idx in split_id:
      train_list.append(split_data_widx(self.source_train, idx))

    self.source_train = train_list

    n2 = self.target_train['X'].shape[0]
    index = np.random.RandomState(seed=42).permutation(n2)
    split_id = np.split(index, [int(.33*n), int(.67*n)])
    train_list = []
    for idx in split_id:
      train_list.append(split_data_widx(self.target_train, idx))

    self.target_train = train_list


    def _fit_one_domain(self, domain_data, task='r'):
      """ fit single domain.
      Args:
          domain_data: data to train, pandas.DataFrame or
                       list of pandas.DataFrame
          task: str, 'r' for regression, 'c' for classification
      """

      if self.split:
        train_data = domain_data[0]
      else:
        train_data = domain_data

      covars = {}
      covars['X'] = jnp.array(train_data['X'])
      covars['C'] = jnp.array(train_data['C'])

      cme_w_xc = ConditionalMeanEmbed(y=jnp.array(train_data['W']),
                                      x=covars,
                                      lam=self.lam_set['cme'],
                                      kernel_dict=self.kernel_dict['cme_w_xc'],
                                      scale=self.sc,
                                      method=self.method_set['cme'],
                                      lam_min=self.lam_set['lam_min'],
                                      lam_max=self.lam_set['lam_max'])

      # estimate cme(W,C|x)
      if self.split:
        train_data = domain_data[1]
      else:
        train_data = domain_data
      covars = {}
      covars['X'] = jnp.array(train_data['X'])

      if len(train_data['W'].shape)>1:
        w = train_data['W']
      else:
        w = train_data['W'][:, jnp.newaxis]

      if len(train_data['C'].shape)>1:
        c = train_data['C']
      else:
        c = train_data['C'][:, jnp.newaxis]
      wc = jnp.hstack((w, c))


      cme_wc_x = ConditionalMeanEmbed(y=wc,
                                      x=covars,
                                      lam=self.lam_set['cme'],
                                      kernel_dict=self.kernel_dict['cme_wc_x'],
                                      scale=self.sc,
                                      method=self.method_set['cme'],
                                      lam_min=self.lam_set['lam_min'],
                                      lam_max=self.lam_set['lam_max'])

      # estimate h0
      xlist = cme_w_xc.get_params()['Xlist']

      if self.split:
        train_data = domain_data[2]
      else:
        train_data = domain_data

      covars = {}
      for key in xlist:
        covars[key] = train_data[key]

      if task == 'r':
        h0 = BridgeH0(cme_w_xc=cme_w_xc,
                       covars=covars,
                       y=train_data['Y'],
                       lam=self.lam_set['h0'],
                       kernel_dict=self.kernel_dict['h0'],
                       scale=self.sc,
                       method=self.method_set['h0'],
                       lam_min=self.lam_set['lam_min'],
                       lam_max=self.lam_set['lam_max'])

      elif task == 'c':
        h0 = BridgeH0CLF(cme_w_xc=cme_w_xc,
                        covars=covars,
                        y=train_data['Y'],
                        lam=self.lam_set['h0'],
                        kernel_dict=self.kernel_dict['h0'],
                        scale=self.sc,
                        method=self.method_set['h0'],
                        lam_min=self.lam_set['lam_min'],
                        lam_max=self.lam_set['lam_max'])
      estimator = {}
      estimator['cme_w_xc'] = cme_w_xc
      estimator['cme_wc_x'] = cme_wc_x
      estimator['h0'] = h0

      return estimator

    def predict_proba(self, x):
      """predict probability.
      Args:
        x: covariates, dict or ndarray.
      """

      if isinstance(x, dict):
        covar_x = x
      else:
        covar_x={'X':x}

      predict_y = self.predict(covar_x, self.calib_domain, self.calib_domain)
      predict_y_prob = normalize(np.array(predict_y), axis=1)

      return predict_y_prob

    def calibrated_evaluation(self, source_clf, target_clf, plot=False):
      """ calibrate probability for evaluation
      Args:
        source_clf: source classifier
        target_clf: target classifier
        plot: plot figure or not, Boolean
      """
      eval_list = []
      task = 'c'

      #source evaluation
      source_testx = {}
      source_testx['X'] = self.source_test['X']
      source_testy = self.source_test['Y']

      #source on source error
      predict_y = self.predict(source_testx, 'source', 'source')
      predict_y = source_clf.predict_proba(predict_y)
      ss_error = self.score(predict_y, source_testy, task)
      eval_list.append(flatten({'task': 'source-source',
                                'predict error': ss_error}))

      if plot:
        testy_label = np.array(jnp.argmax(source_testy, axis=1))
        plt.figure()
        true_0 = np.where(testy_label==0)
        true_1 = np.where(testy_label==1)

        plt.hist(predict_y[true_0,0])
        plt.hist(predict_y[true_1,0])

        plt.hist(predict_y[true_0,1])
        plt.hist(predict_y[true_1,1])
        plt.savefig('ss.png')


      # target on source error
      predict_y = self.predict(source_testx, 'target', 'target')
      predict_y = target_clf.predict_proba(predict_y)
      ts_error = self.score(predict_y, source_testy, task)
      eval_list.append(flatten({'task': 'target-source',
                                'predict error': ts_error}))
      if plot:
        plt.figure()
        true_0 = np.where(testy_label==0)
        true_1 = np.where(testy_label==1)

        plt.hist(predict_y[true_0,0])
        plt.hist(predict_y[true_1,0])

        plt.hist(predict_y[true_0,1])
        plt.hist(predict_y[true_1,1])
        plt.savefig('ts.png')

      #target evaluation
      target_testx = {}
      target_testx['X'] = self.target_test['X']
      target_testy = self.target_test['Y']

      # target on target errror
      predict_y = self.predict(target_testx, 'target', 'target')
      predict_y = target_clf.predict_proba(predict_y)

      tt_error = self.score(predict_y, target_testy, task)
      eval_list.append(flatten({'task': 'target-target',
                                'predict error': tt_error}))

      if plot:
        testy_label = np.array(jnp.argmax(target_testy, axis=1))

        plt.figure()
        true_0 = np.where(testy_label==0)
        true_1 = np.where(testy_label==1)

        plt.hist(predict_y[true_0,0])
        plt.hist(predict_y[true_1,0])

        plt.hist(predict_y[true_0,1])
        plt.hist(predict_y[true_1,1])
        plt.savefig('tt.png')


      # source on target error
      predict_y = self.predict(target_testx, 'source', 'source')
      predict_y = source_clf.predict_proba(predict_y)

      st_error = self.score(predict_y,  target_testy, task)
      eval_list.append(flatten({'task': 'source-target',
                                'predict error': st_error}))

      if plot:
        plt.figure()
        true_0 = np.where(testy_label==0)
        true_1 = np.where(testy_label==1)

        plt.hist(predict_y[true_0,0])
        plt.hist(predict_y[true_1,0])

        plt.hist(predict_y[true_0,1])
        plt.hist(predict_y[true_1,1])
        plt.savefig('st.png')

      #adaptation error
      predict_y = self.predict(target_testx, 'source', 'target')
      predict_y = source_clf.predict_proba(predict_y)

      adapt_error = self.score(predict_y,  target_testy, task)
      eval_list.append(flatten({'task': 'adaptation',
                                'predict error': adapt_error}))

      if plot:
        plt.figure()
        true_0 = np.where(testy_label==0)
        true_1 = np.where(testy_label==1)

        plt.hist(predict_y[true_0,0])
        plt.hist(predict_y[true_1,0])

        plt.hist(predict_y[true_0,1])
        plt.hist(predict_y[true_1,1])
        plt.savefig('adapt.png')

      df = pd.DataFrame(eval_list)
      print(df)

      return df

    def evaluation(self, task='r', plot=False):
      eval_list = []
      print('start evaluation')
      #source evaluation
      source_testx = {}
      source_testx['X'] = self.source_test['X']
      source_testy = self.source_test['Y']

      #source on source error
      predict_y = self.predict(source_testx, 'source', 'source')
      ss_error = self.score(predict_y, source_testy, task)
      eval_list.append(flatten({'task': 'source-source',
                                'predict error': ss_error}))

      if plot:
        testy_label = np.array(jnp.argmax(source_testy, axis=1))
        plt.figure()
        true_0 = np.where(testy_label==0)
        true_1 = np.where(testy_label==1)

        plt.hist(predict_y[true_0,0])
        plt.hist(predict_y[true_1,0])

        plt.hist(predict_y[true_0,1])
        plt.hist(predict_y[true_1,1])
        plt.savefig('ss.png')


      # target on source error
      predict_y = self.predict(source_testx, 'target', 'target')
      ts_error = self.score(predict_y, source_testy, task)
      eval_list.append(flatten({'task': 'target-source',
                                'predict error': ts_error}))
      if plot:
        plt.figure()
        true_0 = np.where(testy_label==0)
        true_1 = np.where(testy_label==1)

        plt.hist(predict_y[true_0,0])
        plt.hist(predict_y[true_1,0])

        plt.hist(predict_y[true_0,1])
        plt.hist(predict_y[true_1,1])
        plt.savefig('ts.png')

      #target evaluation
      target_testx = {}
      target_testx['X'] = self.target_test['X']
      target_testy = self.target_test['Y']

      # target on target errror
      predict_y = self.predict(target_testx, 'target', 'target')
      tt_error = self.score(predict_y, target_testy, task)
      eval_list.append(flatten({'task': 'target-target',
                                'predict error': tt_error}))

      if plot:
        testy_label = np.array(jnp.argmax(target_testy, axis=1))

        plt.figure()
        true_0 = np.where(testy_label==0)
        true_1 = np.where(testy_label==1)

        plt.hist(predict_y[true_0,0])
        plt.hist(predict_y[true_1,0])

        plt.hist(predict_y[true_0,1])
        plt.hist(predict_y[true_1,1])
        plt.savefig('tt.png')


      # source on target error
      predict_y = self.predict(target_testx, 'source', 'source')
      st_error = self.score(predict_y,  target_testy, task)
      eval_list.append(flatten({'task': 'source-target',
                                'predict error': st_error}))

      if plot:
        plt.figure()
        true_0 = np.where(testy_label==0)
        true_1 = np.where(testy_label==1)

        plt.hist(predict_y[true_0,0])
        plt.hist(predict_y[true_1,0])

        plt.hist(predict_y[true_0,1])
        plt.hist(predict_y[true_1,1])
        plt.savefig('st.png')

      #adaptation error
      predict_y = self.predict(target_testx, 'source', 'target')
      adapt_error = self.score(predict_y,  target_testy, task)
      eval_list.append(flatten({'task': 'adaptation',
                                'predict error': adapt_error}))

      if plot:
        plt.figure()
        true_0 = np.where(testy_label==0)
        true_1 = np.where(testy_label==1)

        plt.hist(predict_y[true_0,0])
        plt.hist(predict_y[true_1,0])

        plt.hist(predict_y[true_0,1])
        plt.hist(predict_y[true_1,1])
        plt.savefig('adapt.png')

      df = pd.DataFrame(eval_list)
      print(df)

      return df

    def predict(self, test_x, h_domain, cme_domain):
      """ predict.
      Args:
        test_x: test covariates, dict
        h_domain: domain of h0, "source" or "target"
        cme_domain: domain of cme, "source" or "target"
      """
      if h_domain == 'source':
        h0 =  self.source_estimator['h0']
      else:
        h0 = self.target_estimator['h0']

      if cme_domain == 'source':
        cme_wc_x = self.source_estimator['cme_wc_x']
      else:
        cme_wc_x = self.target_estimator['cme_wc_x']

      predict_y = h0.get_EYx(test_x, cme_wc_x)
      return predict_y
