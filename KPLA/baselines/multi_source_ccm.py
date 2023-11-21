
from scipy.optimize import minimize
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from sklearn.utils.validation import  check_is_fitted
from scipy.linalg import solve
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, train_test_split
from cvxopt import matrix
from cvxopt import solvers
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class MuiltiSourceCCM:
    """
    Implement multi source convex combinations.

    Mansour, Y., Mohri, M., & Rostamizadeh, A. (2008). 
    Domain adaptation with multiple sources. 
    Advances in neural information processing systems, 21.


    """
    def __init__(self, n_env, kde_kernel='gaussian', bandwidth=1.0, max_iter=300):
        self.n_env = n_env
        self.kde_kernel = kde_kernel
        self.bandwidth = bandwidth
        self.KDE_x = []
        self.classifiers = []
        for i in range(n_env):
            self.KDE_x.append(KernelDensity(kernel=kde_kernel, bandwidth=bandwidth))
            self.classifiers.append(MLPClassifier(random_state=1, max_iter=max_iter))


    def fit(self, source_data, x_target=None, weight=None):
        # fit KDE
        longY = []

            

        for id, train_data in enumerate(source_data):
            x_train = np.array(train_data['X'])
            y_train = np.array(train_data['Y'])
            longY += list(np.unique(y_train))
        
            self.classifiers[id].fit(x_train, y_train)
            self.KDE_x[id].fit(x_train)
        
        self.n_labels_ =  list(set(longY))
        """
        if (weight == None) and (x_target is not None):
            #learn the weight by solving least-squares
            #split target for training and testing
            n_target = x_target.shape[0]
            train_test_split(np.arange(n_target), test_size = 0.3, random_state)
            
            target_KDE_x = KernelDensity(self.kde_kernel, self.bandwidth).fit()
            prob_x_target = [np.exp(probx.score_samples(x_target)) for probx in self.KDE_x]
            prob_x_target = np.array(prob_x_target)
        """

        #else:
        self.weight_ = weight
    
    def predict(self, x_new):
        weight_x = np.array([np.exp(probx.score_samples(x_new)) for probx in self.KDE_x])
        normalized_weight_x = normalize(np.array(weight_x), axis=0)
        predictY = np.array([clf.predict(x_new) for clf in self.classifiers])
        return np.sum((normalized_weight_x*predictY).T, axis=1)

    def predict_proba(self, x_new):
        weight_x = np.array([np.exp(probx.score_samples(x_new)) for probx in self.KDE_x])
        normalized_weight_x = normalize(np.array(weight_x), axis=0)
        predictY_proba = np.zeros((x_new.shape[0], self.n_env, len(self.n_labels_)))
        
        for i, clf in enumerate(self.classifiers):
            predictY_proba[:,i,:] = clf.predict_proba(x_new)
        
        return np.sum(predictY_proba*normalized_weight_x[:,:,np.newaxis].transpose((1,0,2)), axis=1)
     

class MultiSouce_simple_adapt(MuiltiSourceCCM):
    """
    Implement multi source convex combinations.

    Mansour, Y., Mohri, M., & Rostamizadeh, A. (2008). 
    Domain adaptation with multiple sources. 
    Advances in neural information processing systems, 21.
    """

    def fit(self, source_data):
        weight = np.ones(self.n_env)/self.n_env
        super().fit(source_data, x_target=None, weight=weight)

    

class MultiSourceUniform:
    def __init__(self, n_env, max_iter=300):
        self.n_env = n_env
        self.classifiers = [MLPClassifier(random_state=1, max_iter=max_iter) for _ in range(n_env)]

    def fit(self, source_data):

        longY = []
        for i, train_data in enumerate(source_data):

            x_train = train_data['X']
            y_train = train_data['Y']
            longY += list(np.unique(y_train))
            
            self.classifiers[i].fit(x_train, y_train)
        self.n_labels_ =  list(set(longY))
        
        return self 

    def predict(self, new_x):
        predictY = np.zeros((new_x.shape[0], self.n_env))
        for i, clf in enumerate(self.classifiers):
            predictY[:, i] = clf.predict(new_x)

        return np.sum(predictY, axis=1)/self.n_env

    def predict_proba(self, new_x):
        
        predict_probaY = np.zeros((new_x.shape[0], self.n_env, len(self.n_labels_)))
        for i, clf in enumerate(self.classifiers):
            predict_probaY[:, i, :] = clf.predict_proba(new_x)

        return np.sum(predict_probaY, axis=1)/self.n_env



class MultiSourceCat:
    def __init__(self, max_iter=300):
        self.classifier = MLPClassifier(random_state=1, max_iter=max_iter)

    def fit(self, source_data):

        
        cat_train_x = np.array(source_data[0]['X'])
        cat_train_y = np.array(source_data[0]['Y'])
        longY = list(np.unique(cat_train_y))

        for _, train_data in enumerate(source_data[1::]):

            x_train = np.array(train_data['X'])
            y_train = np.array(train_data['Y'])
            cat_train_x = np.concatenate((cat_train_x, x_train))
            cat_train_y = np.concatenate((cat_train_y, y_train))
            
            longY += list(np.unique(y_train))
        
        self.classifier.fit(cat_train_x, cat_train_y)
        
        self.n_labels_ =  list(set(longY))
        
        return self 
    
    def predict(self, new_x):

        return self.classifier.predict(new_x)

    def predict_proba(self, new_x):

        return self.classifier.predict_proba(new_x)