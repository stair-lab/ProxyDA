#this file implement the full adaptation pipeline
import pandas as pd
import numpy as np
import jax.numpy as jnp
from .method import KernelMethod, split_data_widx
from .cme import ConditionalMeanEmbed
from .bridge_h0 import Bridge_h0, Bridge_h0_classification
from .bridge_m0 import CME_m0_cme

from .kernel_utils import flatten

from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt


class full_adapt(KernelMethod):
    """
    Adaptation setting: observe (W,X,Y,C) from the source, (W,X,C) from the target

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
            domain_data: data to train, pandas.DataFrame or list of pandas.DataFrame
        """


        if self.split:
            train_data = domain_data[0]
        else:
            train_data = domain_data
        
        covars = {}
        covars['X'] = jnp.array(train_data['X'])
        covars['C'] = jnp.array(train_data['C'])

        cme_W_XC = ConditionalMeanEmbed(jnp.array(train_data['W']), covars, self.lam_set['cme'], 
                                        kernel_dict=self.kernel_dict['cme_w_xc'], scale=self.sc, 
                                        method=self.method_set['cme'], lam_min=self.lam_set['lam_min'], 
                                        lam_max=self.lam_set['lam_max'])

        # estimate cme(W,C|x)
        if self.split:
            train_data = domain_data[1]
        else:
            train_data = domain_data
        covars = {}
        covars['X'] = jnp.array(train_data['X'])
        
        if len(train_data['W'].shape)>1:
            W = train_data['W']
        else:
            W = train_data['W'][:, jnp.newaxis]

        if len(train_data['C'].shape)>1:
            C = train_data['C']
        else:
            C = train_data['C'][:, jnp.newaxis]
        WC = jnp.hstack((W, C))


        cme_WC_X = ConditionalMeanEmbed(WC, covars, self.lam_set['cme'], 
                                        kernel_dict = self.kernel_dict['cme_wc_x'], scale=self.sc,  
                                        method=self.method_set['cme'], lam_min=self.lam_set['lam_min'], 
                                        lam_max=self.lam_set['lam_max'])



        # estimate h0
        Xlist = cme_W_XC.get_params()['Xlist']
        if self.split:
            train_data = domain_data[2]
        else:
            train_data = domain_data 
        
        covars = {}
        for key in Xlist:
            covars[key] = train_data[key]
        if task == 'r':
            h0 = Bridge_h0(cme_W_XC, covars, train_data['Y'], self.lam_set['h0'], 
                            kernel_dict = self.kernel_dict['h0'], scale = self.sc,  
                            method=self.method_set['h0'], lam_min=self.lam_set['lam_min'], 
                            lam_max=self.lam_set['lam_max'])
        elif task == 'c':
            h0 = Bridge_h0_classification(cme_W_XC, covars, train_data['Y'], self.lam_set['h0'], 
                            kernel_dict = self.kernel_dict['h0'], scale = self.sc,  
                            method=self.method_set['h0'], lam_min=self.lam_set['lam_min'], 
                            lam_max=self.lam_set['lam_max'])
        estimator = {}
        estimator['cme_w_xc'] = cme_W_XC
        estimator['cme_wc_x'] = cme_WC_X
        estimator['h0'] = h0

        return estimator


    
    def predict_proba(self, X):
        if isinstance(X, dict):
            covar_x = X
        else:
            covar_x={'X':X}
        predictY = self.predict(covar_x, self.calib_domain, self.calib_domain)
        predictY_label = np.array(jnp.argmax(predictY, axis=1))
        predictY_prob = normalize(np.array(predictY), axis=1)

        #return predictY_prob
        return predictY_prob
    
    def calibrated_evaluation(self, source_clf, target_clf):
        eval_list = []
        task = 'c'

        #source evaluation
        source_testX = {}
        source_testX['X'] = self.source_test['X']
        source_testY = self.source_test['Y']

        
        #source on source error
        predictY = self.predict(source_testX, 'source', 'source')
        predictY = source_clf.predict_proba(predictY)
        ss_error = self.score(predictY, source_testY, task)
        eval_list.append(flatten({'task': 'source-source', 'predict error': ss_error}))
        
        
        PLOT = True
        
        if PLOT:
            testY_label = np.array(jnp.argmax(source_testY, axis=1))
            plt.figure()
            true_0 = np.where(testY_label==0)
            true_1 = np.where(testY_label==1)

            plt.hist(predictY[true_0,0])
            plt.hist(predictY[true_1,0])

            plt.hist(predictY[true_0,1])
            plt.hist(predictY[true_1,1])
            plt.savefig('ss.png')


        # target on source error
        predictY = self.predict(source_testX, 'target', 'target')
        predictY = target_clf.predict_proba(predictY)
        ts_error = self.score(predictY, source_testY, task)
        eval_list.append(flatten({'task': 'target-source', 'predict error': ts_error}))
        if PLOT: 
            plt.figure()
            true_0 = np.where(testY_label==0)
            true_1 = np.where(testY_label==1)

            plt.hist(predictY[true_0,0])
            plt.hist(predictY[true_1,0])

            plt.hist(predictY[true_0,1])
            plt.hist(predictY[true_1,1])
            plt.savefig('ts.png')

        #target evaluation
        target_testX = {}
        target_testX['X'] = self.target_test['X']
        target_testY = self.target_test['Y']
      
        # target on target errror
        predictY = self.predict(target_testX, 'target', 'target')
        predictY = target_clf.predict_proba(predictY)
    
        tt_error = self.score(predictY, target_testY, task)
        eval_list.append(flatten({'task': 'target-target', 'predict error': tt_error}))
        
        if PLOT:
            testY_label = np.array(jnp.argmax(target_testY, axis=1))

            plt.figure()
            true_0 = np.where(testY_label==0)
            true_1 = np.where(testY_label==1)

            plt.hist(predictY[true_0,0])
            plt.hist(predictY[true_1,0])

            plt.hist(predictY[true_0,1])
            plt.hist(predictY[true_1,1])
            plt.savefig('tt.png')


        # source on target error
        predictY = self.predict(target_testX, 'source', 'source')
        predictY = source_clf.predict_proba(predictY)

        st_error = self.score(predictY,  target_testY, task)
        eval_list.append(flatten({'task': 'source-target', 'predict error': st_error}))

        if PLOT:
            plt.figure()
            true_0 = np.where(testY_label==0)
            true_1 = np.where(testY_label==1)

            plt.hist(predictY[true_0,0])
            plt.hist(predictY[true_1,0])

            plt.hist(predictY[true_0,1])
            plt.hist(predictY[true_1,1])
            plt.savefig('st.png')

        #adaptation error
        predictY = self.predict(target_testX, 'source', 'target')
        predictY = source_clf.predict_proba(predictY)

        adapt_error = self.score(predictY,  target_testY, task)
        eval_list.append(flatten({'task': 'adaptation', 'predict error': adapt_error}))

        if PLOT:
            plt.figure()
            true_0 = np.where(testY_label==0)
            true_1 = np.where(testY_label==1)

            plt.hist(predictY[true_0,0])
            plt.hist(predictY[true_1,0])

            plt.hist(predictY[true_0,1])
            plt.hist(predictY[true_1,1])
            plt.savefig('adapt.png')

        df = pd.DataFrame(eval_list)
        print(df)

        return df
       


    def evaluation(self, task='r'):
        eval_list = []
        print('start evaluation')
        #source evaluation
        source_testX = {}
        source_testX['X'] = self.source_test['X']
        source_testY = self.source_test['Y']

        
        #source on source error
        predictY = self.predict(source_testX, 'source', 'source')
        ss_error = self.score(predictY, source_testY, task)
        eval_list.append(flatten({'task': 'source-source', 'predict error': ss_error}))
        
        
        PLOT = False
        
        if PLOT:
            testY_label = np.array(jnp.argmax(source_testY, axis=1))
            plt.figure()
            true_0 = np.where(testY_label==0)
            true_1 = np.where(testY_label==1)

            plt.hist(predictY[true_0,0])
            plt.hist(predictY[true_1,0])

            plt.hist(predictY[true_0,1])
            plt.hist(predictY[true_1,1])
            plt.savefig('ss.png')


        # target on source error
        predictY = self.predict(source_testX, 'target', 'target')
        ts_error = self.score(predictY, source_testY, task)
        eval_list.append(flatten({'task': 'target-source', 'predict error': ts_error}))
        if PLOT: 
            plt.figure()
            true_0 = np.where(testY_label==0)
            true_1 = np.where(testY_label==1)

            plt.hist(predictY[true_0,0])
            plt.hist(predictY[true_1,0])

            plt.hist(predictY[true_0,1])
            plt.hist(predictY[true_1,1])
            plt.savefig('ts.png')

        #target evaluation
        target_testX = {}
        target_testX['X'] = self.target_test['X']
        target_testY = self.target_test['Y']
      
        # target on target errror
        predictY = self.predict(target_testX, 'target', 'target')
        tt_error = self.score(predictY, target_testY, task)
        eval_list.append(flatten({'task': 'target-target', 'predict error': tt_error}))
        
        if PLOT:
            testY_label = np.array(jnp.argmax(target_testY, axis=1))

            plt.figure()
            true_0 = np.where(testY_label==0)
            true_1 = np.where(testY_label==1)

            plt.hist(predictY[true_0,0])
            plt.hist(predictY[true_1,0])

            plt.hist(predictY[true_0,1])
            plt.hist(predictY[true_1,1])
            plt.savefig('tt.png')


        # source on target error
        predictY = self.predict(target_testX, 'source', 'source')
        st_error = self.score(predictY,  target_testY, task)
        eval_list.append(flatten({'task': 'source-target', 'predict error': st_error}))

        if PLOT:
            plt.figure()
            true_0 = np.where(testY_label==0)
            true_1 = np.where(testY_label==1)

            plt.hist(predictY[true_0,0])
            plt.hist(predictY[true_1,0])

            plt.hist(predictY[true_0,1])
            plt.hist(predictY[true_1,1])
            plt.savefig('st.png')

        #adaptation error
        predictY = self.predict(target_testX, 'source', 'target')
        adapt_error = self.score(predictY,  target_testY, task)
        eval_list.append(flatten({'task': 'adaptation', 'predict error': adapt_error}))

        if PLOT:
            plt.figure()
            true_0 = np.where(testY_label==0)
            true_1 = np.where(testY_label==1)

            plt.hist(predictY[true_0,0])
            plt.hist(predictY[true_1,0])

            plt.hist(predictY[true_0,1])
            plt.hist(predictY[true_1,1])
            plt.savefig('adapt.png')

        df = pd.DataFrame(eval_list)
        print(df)

        return df



        #adaptation
    def predict(self, testX, h_domain, cme_domain):
        if h_domain == 'source':
            h0 =  self.source_estimator['h0']
        else:
            h0 = self.target_estimator['h0']

        if cme_domain == 'source':
            cme_wc_x = self.source_estimator['cme_wc_x']
        else:
            cme_wc_x = self.target_estimator['cme_wc_x']
        
        predictY = h0.get_EYx(testX, cme_wc_x)
        return predictY




class partial_adapt(KernelMethod):
    """
    Adaptation setting: observe (W,X,Y,C) from the source, (W,C) from the target
    """
    def split_data(self):
        #split to four batches evenly
        n = self.source_train['X'].shape[0]

        index = np.random.RandomState(seed=42).permutation(n)
        split_id = np.split(index, [int(.25*n), int(.5*n), int(.75*n)])
        train_list = []
        for idx in split_id:
            train_list.append(split_data_widx(self.source_train, idx))
        self.source_train = train_list

        n2 = self.target_train['X'].shape[0]
        index = np.random.RandomState(seed=42).permutation(n2)
        split_id = np.split(index, [int(.25*n), int(.5*n), int(.75*n)])
        train_list = []
        for idx in split_id:
            train_list.append(split_data_widx(self.target_train, idx))

        self.target_train = train_list

    def _fit_one_domain(self, domain_data, task='r'):
        """ fit single domain.
        Args:
            domain_data: data to train, pandas.DataFrame or list of pandas.DataFrame
        """

        if self.split:
            train_data = domain_data[0]
        else:
            train_data = domain_data
        
        covars = {}
        covars['X'] = train_data['X']
        covars['C'] = train_data['C']

        cme_W_XC = ConditionalMeanEmbed(train_data['W'], covars, self.lam_set['cme'],
                                        kernel_dict=self.kernel_dict['cme_w_xc'], scale=self.sc, 
                                        method=self.method_set['cme'], lam_min=self.lam_set['lam_min'], 
                                        lam_max=self.lam_set['lam_max'])

        # estimate cme(W|x) and cme(C|x)
        if self.split:
            train_data = domain_data[1]
        else:
            train_data = domain_data
        covars = {}
        covars['X'] = train_data['X']

        cme_W_X = ConditionalMeanEmbed(train_data['W'], covars, self.lam_set['cme'], 
                                        kernel_dict=self.kernel_dict['cme_w_x'], scale=self.sc,  
                                        method=self.method_set['cme'], lam_min=self.lam_set['lam_min'], 
                                        lam_max=self.lam_set['lam_max'])

        cme_C_X = ConditionalMeanEmbed(train_data['C'], covars, self.lam_set['cme'], 
                                        kernel_dict = self.kernel_dict['cme_c_x'] , scale=self.sc,  
                                        method=self.method_set['cme'], lam_min=self.lam_set['lam_min'], 
                                        lam_max=self.lam_set['lam_max'])


        # estimate m0
        if self.split:
            train_data = domain_data[3]
        else:
            train_data = domain_data

        covars = {}
        covars['X'] = train_data['X']
        covars['C'] = train_data['C']


        m0 =  CME_m0_cme(cme_W_X, covars, self.lam_set['m0'], 
                        kernel_dict=self.kernel_dict['m0'], scale=self.sc, 
                        method=self.method_set['m0'], lam_min=self.lam_set['lam_min'], 
                        lam_max=self.lam_set['lam_max'])


        # estimate h0
        Xlist = cme_W_XC.get_params()['Xlist']
        if self.split:
            train_data = domain_data[3]
        else:
            train_data = domain_data 
        
        covars = {}
        for key in Xlist:
            covars[key] = train_data[key]
    
        if task == 'r':
            h0 = Bridge_h0(cme_W_XC, covars, train_data['Y'], self.lam_set['h0'], 
                            kernel_dict = self.kernel_dict['h0'], scale = self.sc,  
                            method=self.method_set['h0'], lam_min=self.lam_set['lam_min'], 
                            lam_max=self.lam_set['lam_max'])
        elif task == 'c':
            h0 = Bridge_h0_classification(cme_W_XC, covars, train_data['Y'], self.lam_set['h0'], 
                            kernel_dict = self.kernel_dict['h0'], scale = self.sc,  
                            method=self.method_set['h0'], lam_min=self.lam_set['lam_min'], 
                            lam_max=self.lam_set['lam_max'])
    
        estimator = {}
        estimator['cme_w_xc'] = cme_W_XC
        estimator['cme_w_x'] = cme_W_X
        estimator['cme_c_x'] = cme_C_X
        estimator['h0'] = h0
        estimator['m0'] = m0

        return estimator

    def predict(self, testX, h_domain, cme_domain):
        if h_domain == 'source':
            h0 =  self.source_estimator['h0']
        else:
            h0 = self.target_estimator['h0']

        if cme_domain == 'source':
            cme_w_x = self.source_estimator['cme_w_x']
            cme_c_x = self.source_estimator['cme_c_x']
        else:
            cme_w_x = self.target_estimator['cme_w_x']
            cme_c_x = self.target_estimator['cme_c_x']
        
        predictY = h0.get_EYx_independent_cme(testX, cme_w_x, cme_c_x)
        return predictY

    def predict_adapt(self, testX, h_domain, cme_domain, m_domain):
        if h_domain == 'source':
            h0 =  self.source_estimator['h0']
        else:
            h0 = self.target_estimator['h0']

        if cme_domain == 'source':
            cme_w_x = self.source_estimator['cme_w_x']
        else:
            cme_w_x = self.target_estimator['cme_w_x']
        
        if m_domain == 'source':
            m0 = self.source_estimator['m0']
        else:
            m0 = self.target_estimator['m0']
        predictY = h0.get_EYx_independent(testX, cme_w_x, m0)

        return predictY
        
    def evaluation(self):

        eval_list = []

        #source evaluation
        source_testX = {}
        source_testX['X'] = self.source_test['X']
        source_testY = self.source_test['Y']


        #source on source error
        predictY = self.predict(source_testX, 'source', 'source')
        ss_error = self.score(predictY, source_testY)
        eval_list.append({'task': 'source-source', 'predict error': ss_error})

        predictY = self.predict_adapt(source_testX, 'source', 'source', 'source')
        ssm_error = self.score(predictY, source_testY)
        eval_list.append({'task': 'source-source (m0)', 'predict error': ssm_error})

        # target on source error
        predictY = self.predict(source_testX, 'target', 'target')
        ts_error = self.score(predictY, source_testY)
        eval_list.append({'task': 'target-source', 'predict error': ts_error})

        predictY = self.predict_adapt(source_testX, 'target', 'target', 'target')
        tsm_error = self.score(predictY, source_testY)
        eval_list.append({'task': 'target-source (m0)', 'predict error': tsm_error})


        #target evaluation
        target_testX = {}
        target_testX['X'] = self.target_test['X']
        target_testY = self.target_test['Y']
      
        # target on target errror
        predictY = self.predict(target_testX, 'target', 'target')
        tt_error = self.score(predictY, target_testY)
        eval_list.append({'task': 'target-target', 'predict error': tt_error})

        predictY = self.predict_adapt(target_testX, 'target', 'target', 'target')
        ttm_error = self.score(predictY, target_testY)
        eval_list.append({'task': 'target-target (m0)', 'predict error': ttm_error})

        # source on target error
        predictY = self.predict(target_testX, 'source', 'source')
        st_error = self.score(predictY,  target_testY)
        eval_list.append({'task': 'source-target', 'predict error': st_error})

        predictY = self.predict_adapt(target_testX, 'source', 'source', 'source')
        stm_error = self.score(predictY,  target_testY)
        eval_list.append({'task': 'source-target (m0)', 'predict error': stm_error})


        #adaptation error
        predictY = self.predict(target_testX, 'source', 'target')
        adaptm_error = self.score(predictY,  target_testY)
        eval_list.append({'task': 'adaptation (observe C)', 'predict error': adaptm_error})

        predictY = self.predict_adapt(target_testX, 'source', 'target', 'source')
        adaptm_error = self.score(predictY,  target_testY)
        eval_list.append({'task': 'adaptation (m0)', 'predict error': adaptm_error})

        df = pd.DataFrame(eval_list)
        print(df)

        return df