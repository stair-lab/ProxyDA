
import numpy as np
import jax.numpy as jnp

from .multienv_method import MultiKernelMethod, concatenate_data

from .cme import ConditionalMeanEmbed
from .bridge_k0 import Bridge_k0, Bridge_k0_categorical, Bridge_k0_classification, Bridge_k0_catclass





class multienv_adapt(MultiKernelMethod):
    """
    Adaptation setting: observe (W,X,Y,Z) from multiple environments, (W,X) from the target

    """
        

    def _fit_source_domains(self, domain_data, task='r'):
        """ fit single domain.
        Args:
            domain_data: data to train, [list of data of each environment]
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

        cme_W_XZ = ConditionalMeanEmbed(jnp.array(train_data['W']), covars, self.lam_set['cme'], 
                                        kernel_dict=self.kernel_dict['cme_w_xz'], scale=self.sc, 
                                        method=self.method_set['cme'], lam_min=self.lam_set['lam_min'], 
                                        lam_max=self.lam_set['lam_max'])


        # estimate k0
        Xlist = cme_W_XZ.get_params()['Xlist']
        if self.split:
            train_data = domain_data[1]
        else:
            train_data = domain_data[0]
        
        covars = {}
        for key in Xlist:
            covars[key] = train_data[key]
        if task == 'r':
            k0 = Bridge_k0(cme_W_XZ, covars, train_data['Y'], self.lam_set['k0'], 
                            kernel_dict = self.kernel_dict['k0'], scale = self.sc,  
                            method=self.method_set['k0'], lam_min=self.lam_set['lam_min'], 
                            lam_max=self.lam_set['lam_max'])
        elif task == 'c':
            k0 = Bridge_k0_classification(cme_W_XZ, covars, train_data['Y'], self.lam_set['k0'], 
                            kernel_dict = self.kernel_dict['k0'], scale = self.sc,  
                            method=self.method_set['k0'], lam_min=self.lam_set['lam_min'], 
                            lam_max=self.lam_set['lam_max'])            
        
        #esitmate cme_w_x

        if self.split:
            train_data = domain_data[2]
        else:
            train_data = domain_data[1]
        

        estimator = {}
        estimator['cme_w_xz'] = cme_W_XZ
        estimator['cme_w_x'] = {}
        estimator['k0'] = k0

        #estimate cme_w_x for each environment
        #print(type(train_data))
        #print(len(train_data))

        for env, d_data in enumerate(train_data):
            covars = {}
            covars['X'] = jnp.array(d_data['X'])

            cme_W_X = ConditionalMeanEmbed(jnp.array(d_data['W']), covars, self.lam_set['cme'], 
                                            kernel_dict=self.kernel_dict['cme_w_x'], scale=self.sc, 
                                            method=self.method_set['cme'], lam_min=self.lam_set['lam_min'], 
                                            lam_max=self.lam_set['lam_max'])
            
            estimator['cme_w_x'][env]  = cme_W_X                     
            
        
        


        return estimator




class multienv_adapt_categorical(MultiKernelMethod):
    """
    Adaptation setting: observe (W,X,Y,Z) from multiple environments, (W,X) from the target
    when Z is a discrete variable

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
        
        unique_z, indices = jnp.unique(jnp.array(train_data['Z']), return_inverse=True)
        unique_z = np.asarray(unique_z) #convert to ndarray
        cme_W_XZ_lookup = {}


        # for each Z, learn a cme_W_XZ.
        for i, z in enumerate(unique_z):
            select_id = jnp.where(indices == i)[0]
            covars = {}
            
            covars['X'] = jnp.array(train_data['X'][select_id,...])


            cme_W_XZ = ConditionalMeanEmbed(jnp.array(train_data['W'][select_id,...]), covars, self.lam_set['cme'], 
                                            kernel_dict=self.kernel_dict['cme_w_xz'], scale=self.sc, 
                                            method=self.method_set['cme'], lam_min=self.lam_set['lam_min'], 
                                        lam_max=self.lam_set['lam_max'])
      
            cme_W_XZ_lookup[z] = cme_W_XZ


        # estimate k0
        Xlist = cme_W_XZ_lookup[unique_z[0]].get_params()['Xlist']
        if self.split:
            train_data = domain_data[1]
        else:
            train_data = domain_data[0]
        
        covars = {}
        for key in Xlist:
            covars[key] = train_data[key]
        covars['Z'] = train_data['Z']
        if task == 'r':
            k0_cat = Bridge_k0_categorical(cme_W_XZ_lookup, covars, train_data['Y'], self.lam_set['k0'], 
                        kernel_dict = self.kernel_dict['k0'], scale = self.sc,  
                        method=self.method_set['k0'], lam_min=self.lam_set['lam_min'], 
                                        lam_max=self.lam_set['lam_max'])
        elif task == 'c':
            k0_cat = Bridge_k0_catclass(cme_W_XZ_lookup, covars, train_data['Y'], self.lam_set['k0'], 
                        kernel_dict = self.kernel_dict['k0'], scale = self.sc,  
                        method=self.method_set['k0'], lam_min=self.lam_set['lam_min'], 
                                        lam_max=self.lam_set['lam_max'])      
        #esitmate cme_w_x

        if self.split:
            train_data = domain_data[2]
        else:
            train_data = domain_data[1]
        

        estimator = {}
        estimator['cme_w_xz_lookup'] = cme_W_XZ_lookup
        estimator['cme_w_x'] = {}
        estimator['k0'] = k0_cat

        #estimate cme_w_x for each environment
        #print(type(train_data))
        #print(len(train_data))

        for env, d_data in enumerate(train_data):
            covars = {}
            covars['X'] = jnp.array(d_data['X'])

            cme_W_X = ConditionalMeanEmbed(jnp.array(d_data['W']), covars, self.lam_set['cme'], 
                                            kernel_dict=self.kernel_dict['cme_w_x'], scale=self.sc, 
                                            method=self.method_set['cme'], lam_min=self.lam_set['lam_min'], 
                                        lam_max=self.lam_set['lam_max'])
            
            estimator['cme_w_x'][env]  = cme_W_X                     
            
        
        


        return estimator

