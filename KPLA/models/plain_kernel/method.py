"""
Implementation of the base kernel estimator
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT


import jax.numpy as jnp
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, ClassifierMixin

# Define Sklearn evaluation functions
def soft_accuracy(y_true, y_pred, threshold=0.5, **kwargs):
  return accuracy_score(y_true, y_pred >= threshold, **kwargs)

def log_loss64(y_true, y_pred, **kwargs):
  return log_loss(y_true, y_pred.astype(np.float64), **kwargs)

evals_sklearn = {
    'll': log_loss64,
    #'soft_acc': soft_accuracy, 
    'hard_acc': accuracy_score,
    'auc': roc_auc_score
}



def split_data_widx(data, split_index):
    """split data with indices, return dictionary
    Args:
        data: dict
        split_idx: ndarray
    Returns:
        sub_data: dict
    """
    sub_data = {}
    keys = data.keys()
    print('split',split_index.shape)
    for key in keys:
        if len(data[key].shape)>1:
            sub_data[key] = jnp.array(data[key][split_index,:])
        else:
            sub_data[key] = jnp.array(data[key][split_index])
    return sub_data



class KernelMethod(BaseEstimator):
    """
    Base estimator for the adaptation
    split_data(), predict(), evaluation(), are implemented by the child class
    """
    def __init__(self, source_train, target_train, source_test, target_test, split, scale=1, lam_set = None, method_set = None, kernel_dict=None):
        """ Initiate parameters
        Args:
            source_train: dictionary, keys: C,W,X,Y
            target_train: dictionary, keys: C, W, X, Y
            source_test:  dictionary, keys: X, Y
            target_test:  dictionary, keys: X, Y
            split: Boolean, split the training dataset or not. If True, the samples are evenly split into groups. 
            Hence, each estimator receive smaller number of training samples.  
            scale: length-scale of the kernel function, default: 1.  
            lam_set: a dictionary of tuning parameter, set None for leave-one-out estimation
            For example, lam_set={'cme': lam1, 'h0': lam2, 'm0': lam3}
            method_set: a dictionary of optimization methods for different estimators, default is 'original'
            kernel_dict: a dictionary of specified kernel functions
        """
        self.source_train = source_train
        self.target_train = target_train
        self.source_test  = source_test
        self.target_test  = target_test
        self.sc = scale
        self.split = split
        self._is_fitted = False
        self.calib_domain = 'source'
        if lam_set == None:
            lam_set={'cme': None, 'h0': None, 'm0': None}
        

        self.lam_set = lam_set

        if method_set == None:
            method_set = {'cme': 'original', 'h0': 'original', 'm0': 'original'}
        self.method_set = method_set

        if kernel_dict == None:
            kernel_dict['cme_w_xc'] = {'X': 'rbf', 'C': 'rbf', 'Y':'rbf'} #Y is W
            kernel_dict['cme_wc_x'] = {'X': 'rbf', 'Y': [{'kernel':'rbf', 'dim':2}, {'kernel':'rbf', 'dim':1}]} # Y is (W,C)
            kernel_dict['cme_c_x']  = {'X': 'rbf', 'Y': 'rbf'} # Y is C
            kernel_dict['cme_w_x']  = {'X': 'rbf', 'Y': 'rbf'} # Y is W
            kernel_dict['h0']       = {'C': 'rbf'}
            kernel_dict['m0']       = {'C': 'rbf', 'X':'rbf'}
        
        self.kernel_dict = kernel_dict 
    def get_params(self, deep=False):
        params = {}
        params['lam_set'] = self.lam_set
        params['method_set'] = self.method_set
        params['kernel_dict'] = self.kernel_dict
        params['split'] = self.split 
        params['scale'] = self.sc

        return params

    def set_params(self, params):
        self.lam_set     = params['lam_set']
        self.method_set  = params['method_set'] 
        self.kernel_dict = params['kernel_dict'] 
        self.split       = params['split'] 
        self.sc          = params['scale']  

    def fit(self, task='r'):
        #split dataset
        if self.split:
            self.split_data()
            print('complete data split')
        # learn estimators from the source domain
        self.source_estimator =  self._fit_one_domain(self.source_train, task)

        # learn estimators from the target domain
        self.target_estimator =  self._fit_one_domain(self.target_train, task)
        self._is_fitted = True
        if task == 'c':
            #print(np.array(self.source_train['Y']))
            self.classes_ = jnp.arange(self.source_train['Y'].shape[1])
    


    def predict(self):
        """Fits the model to the training data."""
        raise NotImplementedError("Implemented in child class.")

    def evaluation(self):
        """Fits the model to the training data."""
        raise NotImplementedError("Implemented in child class.")


    def score(self, predictY, testY, task='r'):
        ## Fix shape
        if task == 'r':
            if testY.shape > predictY.shape:
                assert testY.ndim == predictY.ndim + 1 and testY.shape[:-1] == predictY.shape, "unresolveable shape mismatch betweenn testY and predictY"
                predictY = predictY.reshape(testY.shape)
            elif testY.shape < predictY.shape:
                assert testY.ndim + 1 == predictY.ndim and testY.shape == predictY.shape[:-1], "unresolveable shape mismatch betweenn testY and predictY"
                testY = testY.reshape(predictY.shape)
            error =  {'l2':np.sum((testY-predictY)**2)/predictY.shape[0]}
        
        elif task == 'c':
            error = {}

            
            testY_label = np.array(jnp.argmax(testY, axis=1))
            predictY_label = np.array(jnp.argmax(predictY, axis=1))
            #predictY_prob = softmax(np.array(predictY), axis=1)
            predictY_prob = normalize(np.array(predictY), axis=1)



            for eva in evals_sklearn.items():
                #print(eva[1](testY_label, predictY_label))
                if eva[0] == 'hard_acc':
                    error[eva[0]] = eva[1](testY_label, predictY_label)
                else:
                    error[eva[0]] = eva[1](testY_label, predictY_prob[:,1])
        return error

    
    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted
    
