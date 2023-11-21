
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






class MultiSourceMMD:
    def __init__(self, source_data, kernel, kde_kernel, bandwidth=1.0):

        self.kernel = kernel # kernel for computing MMD
        self.kde_kernel = kde_kernel # kernel for KDE
        self.bandwidth = bandwidth
        self.X = [np.asarray(d['X']) for d in source_data]
        self.Y = [np.asarray(d['Y']) for d in source_data]

        self.n_env = len(source_data)
        longY = []
        for y in self.Y:
            longY += list(np.unique(y))
        self.labels = list(set(longY))
        self.n_label = len(self.labels)
        self.count_table = np.zeros((self.n_env, self.n_label))

        #print('label set', self.labels)
        A = np.zeros((self.n_env*self.n_label, self.n_env*self.n_label))

        def element_A(i, j, k, m):
            Xi_k, n_ik = self._get_X_by_label(i, k)
            Xj_m, n_jm = self._get_X_by_label(j, m)

            K_ij_km = self.kernel(Xi_k, Xj_m)
            
            #update the count table
            if self.count_table[i, k] == 0.:
                self.count_table[i, k] = n_ik
            if self.count_table[j, m] == 0.:
                self.count_table[j, m] = n_jm




            return np.sum(K_ij_km) / (n_ik*n_jm)

        
        # construct A
        for i in range(self.n_env):
            for j in range(self.n_env):
                for k in range(self.n_label):
                    for m in range(self.n_label):
                        A[i*self.n_label+k, j*self.n_label+m] = element_A(i, j, k, m)
        
        #f1 = lambda i,j: np.vectorize(lambda k,m: element_A(i, j, k, m))(np.arange)
        
        
        print('construct A')
        self.A = A
        
        # construct the density esimator of P_Xi_y
        kde_X_Y = []
        for k in range(self.n_label):
            kde_X_y = self._get_KDE_X_y(k) #y= self.labels[k]
            kde_X_Y.append(kde_X_y)
        print('construct KDE')
        self.kde_X_Y = kde_X_Y


    
    def _get_KDE_X_y(self, k):
        PX_y = []
        for i in range(self.n_env):
            Xi_k, _ = self._get_X_by_label(i, k)
            kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kde_kernel).fit(Xi_k)

            PX_y.append(kde) #y=self.labels[k]
        
        return PX_y #len = n_env

    def _get_pdf_KDE_X_y(self, k, x_new):
        pdf_list = []
        for kde in self.kde_X_Y[k]:
            log_pdf = kde.score_samples(x_new)
            pdf_list.append(np.exp(log_pdf))
        return np.array(pdf_list)



    def _get_X_by_label(self, i, k):
        label_i = self.labels[k]
        loc_i = np.where(self.Y[i] == label_i)[0]
        n_ik = loc_i.size
        Xi_k  = self.X[i][loc_i,...]
        return Xi_k, n_ik
    
    def _get_beta(self, b):
        C = matrix(np.ones(self.n_env*self.n_label)).T
        d =  matrix(np.ones(1))
        G = matrix(-np.eye(self.n_env*self.n_label)) #beta is positive
        h = matrix(np.zeros(self.n_env*self.n_label))#beta is positive
        P = matrix(self.A)
        Q = matrix(b)
        sol = solvers.qp(P, Q, G, h, A=C, b=d)
        print('solve beta status:', sol['status'])
        return np.array(sol['x'])

    def fit(self, target_x):
        """fit target domain data
        Args:
            newX: nddarry
        """
        # construct b
        b = np.zeros(self.n_env*self.n_label)
        n_new = target_x.shape[0]
    
        def element_b(i, k):
            Xi_k, n_ik = self._get_X_by_label(i, k)
            K_inew_k = self.kernel(Xi_k, target_x)

            return np.sum(K_inew_k)/(n_new*n_ik)
        for i in range(self.n_env):
            for k in range(self.n_label):
                b[i*self.n_label+k] = element_b(i,k)

        # get beta using cvx
        beta = self._get_beta(b)

        beta = beta.reshape((self.n_env, self.n_label), order='F')
        
        Py_new = np.zeros(self.n_label)

        for j in range(self.n_label):
            Py_new[j] = np.sum(beta[:,j])
        
        alpha = np.zeros((self.n_env, self.n_label))
        for i in range(self.n_env):
            for j in range(self.n_label):
                alpha[i, j] = beta[i, j]/Py_new[j]

        self.Py_target_ = Py_new
        self.alpha_ = alpha
"""
class MuiltiSource_weigh_samples(MultiSourceMMD):
    def fit(self, target_x):
        super().fit(target_x)
        #fit a new classifier with weighted samples
        big_x = np.array([])
        big_y = np.array([])
        weights = np.array([])
        for e in range(self.n_env):
            for j in range(self.n_label):
                w1 = self.Py_target_[j]*self.alpha_[e, j]
                if self.count_table[e, j] > 0.:
                    w1 /= self.count_table[e, j] 
                else:
                    w1 = 0. #the labels is never seen in the enviroment
                
                idx = np.where(self.Y[e]==self.labels[j])[0]
                big_x = np.concatenate((big_x, self.X[e][idx,...]))
                big_y = np.concatenate((big_y, self.Y[e][idx,...]))
                weights = np.concarenate((weights, np.ones(idx.size)*w1))
        clf = MLPClassifier(random_state=1, max_iter=300).fit()
    def predict(self, new_x):
        clf.predict_proba
"""        
                






#class MuiltiSource_genar_model(MultiSourceMMD):
#    def 


class MuiltiSource_combn_classf(MultiSourceMMD):
    def predict(self, x):
        """
        predict Y from x
        """
        out_prob = np.zeros((x.shape[0], self.n_label))
        #print(out_prob.shape)
        for j in range(self.n_label):
            #print(self._get_pdf_KDE_X_y(j, x).shape)

            p = self.Py_target_[j] * np.sum(self._get_pdf_KDE_X_y(j, x)*self.alpha_[:,j][:,np.newaxis], axis=0)
            #print('p', p.shape)
            out_prob[:, j] = p

        # normalize probability
        out_prob /= np.sum(out_prob, axis=1)[:,np.newaxis]
        return out_prob





        

        
        



    






