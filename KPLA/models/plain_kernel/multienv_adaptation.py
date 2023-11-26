"""
  implements the multi-source adaptation pipeline
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT
import numpy as np
import jax.numpy as jnp

from KPLA.models.plain_kernel.multienv_method import MultiKernelMethod

from KPLA.models.plain_kernel.cme import ConditionalMeanEmbed
from KPLA.models.plain_kernel.bridge_m0 import BridgeM0, BridgeM0CAT, BridgeM0CLF, BridgeM0CATCLF





class MultiEnvAdapt(MultiKernelMethod):
  """
  Adaptation setting: observe (W,X,Y,Z) from multiple environments,
  (W,X) from the target
  """

  def _fit_source_domains(self, domain_data, task='r'):
    """ fit single domain.
    Args:
        domain_data: data to train, [list of data of each environment]
        task: regression for 'r', classification for 'c'
    """

    if self.split:
      train_data = domain_data[0]
    else:
      #keys = domain_data.keys()
      #empty_dict = dict(zip(keys, [None]*len(keys)))
      train_data = domain_data[0]

    covars = {}
    covars['X'] = jnp.array(train_data['X'])
    covars['Z'] = jnp.array(train_data['Z'])

    cme_w_xz = ConditionalMeanEmbed(jnp.array(train_data['W']),
                                    covars,
                                    self.lam_set['cme'],
                                    kernel_dict=self.kernel_dict['cme_w_xz'],
                                    scale=self.sc,
                                    method=self.method_set['cme'],
                                    lam_min=self.lam_set['lam_min'],
                                    lam_max=self.lam_set['lam_max'])


    # estimate m0
    xlist = cme_w_xz.get_params()['Xlist']
    if self.split:
      train_data = domain_data[1]
    else:
      train_data = domain_data[0]

    covars = {}
    for key in xlist:
      covars[key] = train_data[key]
    if task == 'r':
      m0 = BridgeM0(cme_w_xz,
                    covars,
                    train_data['Y'],
                    self.lam_set['m0'],
                    kernel_dict = self.kernel_dict['m0'],
                    scale = self.sc,
                    method=self.method_set['m0'],
                    lam_min=self.lam_set['lam_min'],
                    lam_max=self.lam_set['lam_max'])
    elif task == 'c':
      m0 = BridgeM0CLF(cme_w_xz,
                        covars,
                        train_data['Y'],
                        self.lam_set['m0'],
                        kernel_dict = self.kernel_dict['m0'],
                        scale = self.sc,
                        method=self.method_set['m0'],
                        lam_min=self.lam_set['lam_min'],
                        lam_max=self.lam_set['lam_max'])

    #esitmate cme_w_x

    if self.split:
      train_data = domain_data[2]
    else:
      train_data = domain_data[1]


    estimator = {}
    estimator['cme_w_xz'] = cme_w_xz
    estimator['cme_w_x'] = {}
    estimator['m0'] = m0

    #estimate cme_w_x for each environment
    #print(type(train_data))
    #print(len(train_data))

    for env, d_data in enumerate(train_data):
      covars = {}
      covars['X'] = jnp.array(d_data['X'])

      cme_w_x = ConditionalMeanEmbed(jnp.array(d_data['W']),
                                     covars,
                                     self.lam_set['cme'],
                                     kernel_dict=self.kernel_dict['cme_w_x'],
                                     scale=self.sc,
                                     method=self.method_set['cme'],
                                     lam_min=self.lam_set['lam_min'],
                                     lam_max=self.lam_set['lam_max'])

      estimator['cme_w_x'][env]  = cme_w_x


    return estimator




class MultiEnvAdaptCAT(MultiKernelMethod):
  """
  Adaptation setting: observe (W,X,Y,Z) from multiple environments,
  (W,X) from the target when Z is a discrete variable
  """

  def _fit_source_domains(self, domain_data, task='r'):
    """ fit single domain.
    Args:
        domain_data: data to train, [list of data of each environment]
        task: specify the task, 'r': for regression, 'c': for classification
    """



    if self.split:
      train_data = domain_data[0]
    else:
      train_data = domain_data[0]

    unique_z, indices = jnp.unique(jnp.array(train_data['Z']),
                                   return_inverse=True)
    unique_z = np.asarray(unique_z) #convert to ndarray
    cme_w_xz_lookup = {}


    # for each Z, learn a cme_w_xz.
    for i, z in enumerate(unique_z):
      select_id = jnp.where(indices == i)[0]
      covars = {}

      covars['X'] = jnp.array(train_data['X'][select_id,...])


      cme_w_xz = ConditionalMeanEmbed(jnp.array(train_data['W'][select_id,...]),
                                      covars,
                                      self.lam_set['cme'],
                                      kernel_dict=self.kernel_dict['cme_w_xz'],
                                      scale=self.sc,
                                      method=self.method_set['cme'],
                                      lam_min=self.lam_set['lam_min'],
                                      lam_max=self.lam_set['lam_max'])

      cme_w_xz_lookup[z] = cme_w_xz


    # estimate m0
    xlist = cme_w_xz_lookup[unique_z[0]].get_params()['Xlist']
    if self.split:
      train_data = domain_data[1]
    else:
      train_data = domain_data[0]

    covars = {}
    for key in xlist:
      covars[key] = train_data[key]
    covars['Z'] = train_data['Z']
    if task == 'r':
      m0_cat = BridgeM0CAT(cme_w_xz_lookup,
                            covars,
                            train_data['Y'],
                            self.lam_set['m0'],
                            kernel_dict = self.kernel_dict['m0'],
                            scale = self.sc,
                            method=self.method_set['m0'],
                            lam_min=self.lam_set['lam_min'],
                            lam_max=self.lam_set['lam_max'])
    elif task == 'c':
      m0_cat = BridgeM0CATCLF(cme_w_xz_lookup,
                              covars,
                              train_data['Y'],
                              self.lam_set['m0'],
                              kernel_dict = self.kernel_dict['m0'],
                              scale = self.sc,
                              method=self.method_set['m0'],
                              lam_min=self.lam_set['lam_min'],
                              lam_max=self.lam_set['lam_max'])
    #esitmate cme_w_x

    if self.split:
      train_data = domain_data[2]
    else:
      train_data = domain_data[1]


    estimator = {}
    estimator['cme_w_xz_lookup'] = cme_w_xz_lookup
    estimator['cme_w_x'] = {}
    estimator['m0'] = m0_cat

    #estimate cme_w_x for each environment
    #print(type(train_data))
    #print(len(train_data))

    for env, d_data in enumerate(train_data):
      covars = {}
      covars['X'] = jnp.array(d_data['X'])

      cme_w_x = ConditionalMeanEmbed(jnp.array(d_data['W']),
                                      covars,
                                      self.lam_set['cme'],
                                      kernel_dict=self.kernel_dict['cme_w_x'],
                                      scale=self.sc,
                                      method=self.method_set['cme'],
                                      lam_min=self.lam_set['lam_min'],
                                    lam_max=self.lam_set['lam_max'])

      estimator['cme_w_x'][env]  = cme_w_x


    return estimator

