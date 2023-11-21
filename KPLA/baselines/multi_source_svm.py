from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.kernel_approximation import Nystroem

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier


class MultiSource_SVM:
    """
    Blanchard, G., Lee, G., & Scott, C. (2011). 
    Generalizing from several related classification tasks to a new unlabeled sample. 
    Advances in neural information processing systems, 24.
    """

    def __init__(self, p_kernel, x_kernel, max_iter=300):
        #self.svc = make_pipeline(StandardScaler(), LinearSVC())
        self.svc = SGDClassifier(max_iter=max_iter, tol=1e-3, loss='hinge')
        self.p_kernel = p_kernel #kernel to compute the probability distance
        self.x_kernel = x_kernel

    def _compute_pdist(self, source_x_i, target_x):
        K = self.p_kernel(source_x_i, target_x)
        d_source_i_target = np.mean(K)
        return d_source_i_target

    def fit(self, source_data, target_data):

        self.n_env = len(source_data)
        dist_weight = np.zeros(len(source_data))
        
        for j, source_d in enumerate(source_data):
            
            dist_weight[j] = self._compute_pdist(np.array(source_d['X']), np.array(target_data['X']))
        
        self.dist_weight_ = dist_weight
        
        #dist_matrix = np.zeros((self.n_env, self.n_env))
        
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

                Kxx = self.x_kernel(np.array(source_data[i]['X']), 
                                    np.array(source_data[j]['X']))
                
                big_feature[start_x:start_x+len_i, start_y:start_y+len_j] = Kxx*w
                big_feature[start_y:start_y+len_j, start_x:start_x+len_i] = (Kxx.T)*w 
                weights[start_x:start_x+len_i] = self.dist_weight_[i]

        self.X_ = big_x
        self.weights_ = weights
        #subsample
        """
        self.feature_map_ = Nystroem(gamma=self.gamma, random_state=0, n_components=self.n_components)
        sampled_fectures = self.feature_map_.fit_transform(big_feature*self.weights_)
        #self.weights_ = weights[self.feature_map_.component_indices_]
        self.svc.fit(sampled_fectures, big_y)
        df = self.svc.decision_function(sampled_fectures)
        self.df_min_ = df.min()
        self.df_max_ = df.max()
        """
        #self.feature_map_ = Nystroem(gamma=self.gamma, random_state=0, n_components=self.n_components)
        #self.feature_map_ = RBFSampler(gamma=1, random_state=1)
        #feature_x = self.feature_map_.fit_transform(big_feature*self.weights_[np.newaxis,:])
        
        #self.svc.fit(feature_x, big_y)

        self.svc.fit(big_feature*self.weights_[np.newaxis,:], big_y)
        #fit the svm model

    def transform_feature_x(self, xnew):
        Knewxx = self.x_kernel(xnew, self.X_)

        return Knewxx*self.weights_[np.newaxis,:]
    
    def predict(self, xnew):
        #create feature map
        weight_Knewxx = self.transform_feature_x(np.array(xnew))
        predictY = self.svc.predict(weight_Knewxx)
        
        return predictY
    """
    def predict_proba(self, xnew):
        weight_Knewxx = self.transform_feature_x(np.array(xnew))
        predictY_proba = self.svc.predict_proba(weight_Knewxx)

        return predictY_proba

    """
    def decision(self, xnew):
        """Min-max scale output of `decision_function` to [0, 1]."""
        
        weight_Knewxx = self.transform_feature_x(np.array(xnew))
        decision_x = self.svc.decision_function(weight_Knewxx)
                
        return decision_x



