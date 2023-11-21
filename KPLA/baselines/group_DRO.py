from scipy.optimize import minimize
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from sklearn.utils.validation import  check_is_fitted
from scipy.linalg import solve



class KernelRidgeGroupDRO(KernelRidge):
    def __init__(self, n_iter=1000, alpha=1, gamma=None, degree=3, coef0=1, kernel='rbf'):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.n_iter = n_iter
        self.degree = degree
        self.coef0 = coef0

    def fit(self, X, y, eta_q=None, random_seed=42, tol=1e-3):
        """
        Args:
            X: list of covariates, list of ndarray [(n_samples, n_features)]
            y: list of responses,  list of ndarray [(n_samples, )]
            eta_q: coefficients, list of length = number of groups
            tol: convergence tolerance
        """
        
        K = []
        n_groups = len(X)
        catX = X[0]
        catY = y[0]
        
        group_size = []
        group_size.append(X[0].shape[0])

        for i in range(1, n_groups):
            group_size.append(X[i].shape[0])
            catX = np.concatenate((catX, X[i]))
        
        catK = self._get_kernel(catX) 

        group_size = np.array(group_size)
        end_id     = np.cumsum(group_size)
        start_ids = np.concatenate((np.array([0]), end_id))
        
        start_ids = [int(i) for i in start_ids]

        for i in range(n_groups):
            K.append(self._get_kernel(X[i]))

        if eta_q == None:
            np.random.seed(random_seed)
            generator = lambda r: 10**np.random.uniform(-3, -1, r)
            eta_q = generator(n_groups)
        
        
        if isinstance(eta_q, float):
            eta_q = [eta_q for _ in range(n_groups)]

        def group_loss(coef, i):
            return np.mean((np.dot(catK[start_ids[i]:start_ids[i+1], :], coef) - y[i])**2)


        def loss_fun_weights(coef, weights):
            err = 0
            for i in range(n_groups):
                g_err = group_loss(coef, i)
                err += g_err*weights[i]
            
            return err

        prev_coef = np.zeros(catK.shape[1])
        group_weights = np.ones(n_groups)
        prev_err = np.inf

        for t in range(self.n_iter):
            
            for i in range(n_groups):
                group_weights[i] *= np.exp(eta_q[i]*group_loss(prev_coef, i))

                
            
            group_weights /= np.sum(group_weights+1e-8)
            
            input_y = np.sqrt(group_weights[0]) * y[0]
            input_K = np.sqrt(group_weights[0]) * catK[start_ids[0]:start_ids[1], :]

            for i in range(1, n_groups):
                new_y = np.sqrt(group_weights[i]) * y[i]
                new_K = np.sqrt(group_weights[i]) * catK[start_ids[i]:start_ids[i+1], :]
                input_y = np.concatenate((input_y, new_y))
                input_K = np.concatenate((input_K, new_K))
            
            prev_coef = solve(input_K+self.alpha*np.eye(input_K.shape[0]), input_y)
            


            loss_fun = lambda c: loss_fun_weights(c, group_weights)
            new_err = loss_fun(prev_coef)

            print('group error:', new_err)
            if(np.abs(new_err-prev_err)<tol):
                break
            prev_err = new_err

            #scipy optmize
            """
            loss_fun = lambda c: loss_fun_weights(c, group_weights)
            res = minimize(loss_fun, 
                        prev_coef, 
                        method='BFGS',
                        tol=1e-6)

            
            print('iter {}'.format(t), res.message)
            if res.status == 0: 
                prev_coef = res.x
                out_coef = res.x
                new_err = loss_fun(res.x)
                print('group error:', new_err)
                if(np.abs(new_err-prev_err)<tol):
                    break
                prev_err = new_err
            else:
                prev_coef = coef_init
            """

        self.dual_coef_ = prev_coef
        
        self.X_fit_ = catX
        self.n_groups = n_groups

    def predict(self, X):
        
        check_is_fitted(self)
        K = self._get_kernel(X, self.X_fit_)
        return np.dot(K, self.dual_coef_)