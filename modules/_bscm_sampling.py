import numpy as np
import pandas as pd


class BayesianSyntheticControl:
    def __init__(
        self,
        control_outcome: pd.DataFrame, 
        treatment_outcome: pd.DataFrame,
        X: pd.DataFrame,
        adj_mat: np.array,
        adj_vec: np.array,
        T0: int,
        time_col: str,
        unit_col: str
        ) -> None:
        self.control_outcome = control_outcome
        self.treatment_outcome = treatment_outcome
        self.adj_mat = adj_mat
        self.adj_vec = adj_vec
        self.X = X
        self.T0 = T0
        self.N = control_outcome[unit_col].nunique()
        self.T = control_outcome[time_col].nunique()
        self.k = X.shape[1] - 2
        self.time_col = time_col
        self.unit_col = unit_col
        
        
    def _validate_inputs(self) -> None:
        """
        Validate input data
        """
        if not isinstance(self.control_outcome, pd.DataFrame):
            raise ValueError("control_outcome must be a pandas DataFrame.")
        if not isinstance(self.treatment_outcome, pd.DataFrame):
            raise ValueError("treatment_outcome must be a pandas DataFrame.")
        if not isinstance(self.X, pd.DataFrame):
            raise ValueError("covariates must be a pandas DataFrame.")
        if not isinstance(self.adj_mat, np.ndarray):
            raise ValueError("adjacency_matrix must be a numpy array.")
        if not isinstance(self.adj_vec, np.ndarray):
            raise ValueError("adjacency_vector must be a numpy array.")
        if self.T0 >= self.T:
            raise ValueError("pre_treatment_period must be less than the total number of time periods.")
    
    def preprocess_data(self):
        # Implement any necessary preprocessing steps
        X = self.X.drop([self.time_col, self.unit_col], axis=1).to_numpy().reshape((self.N, self.T, self.k))
        control_outcome = self.control_outcome.drop([self.time_col, self.unit_col], axis=1).to_numpy().reshape((self.N, self.T))
        treatment_outcome = self.treatment_outcome.drop([self.time_col], axis=1).to_numpy().flatten()
        self.X = X
        self.control_outcome = control_outcome
        self.treatment_outcome = treatment_outcome
        pass
    
    def mcmc_weight(self, max_iter=10000, burn_in=5000):
        self.max_iter = max_iter
        self.burn_in = burn_in
        N = self.N
        
        
        alpha, sigma2_i, nu_sigma2_i = np.zeros((max_iter, self.N)), np.zeros((max_iter,self. N)), np.zeros((max_iter,self. N))
        tau2, nu_tau2 = np.zeros(max_iter), np.zeros(max_iter)
        sigma2, nu_sigma2 = np.zeros(max_iter), np.zeros(max_iter)
        
        sigma2_i[0, :], nu_sigma2_i[0, :] = 1, 1
        tau2[0], nu_tau2[0] = 1, 1
        sigma2[0], nu_sigma2[0] = 1, 1
        
        control_outcome_pre = self.control_outcome[:, :self.T0]
        treatment_outcome_pre = self.treatment_outcome[:self.T0]
        D_temp = control_outcome_pre.T @ control_outcome_pre
        
        for iter in range(1, max_iter):
            # Update alpha
            D = D_temp + sigma2[iter-1] * np.diag(1 / sigma2_i[iter-1, :])
            D_inv = np.linalg.inv(D)
            alpha_mean =D_inv @ control_outcome_pre.T @ treatment_outcome_pre
            alpha_cov = sigma2[iter-1] * D_inv
            
            alpha[iter, :] = np.random.multivariate_normal(alpha_mean, alpha_cov)
            
            # Update sigma2_i
            for i in range(N):
                sc = 0.5 * alpha[iter, i]**2 + 1 / nu_sigma2_i[iter-1, i]
                sigma2_i[iter, i] = 1 / np.random.gamma(1, 1 / sc)
                
            # Update nu_sigma_2_i
            for i in range(N):
                sc = 1 / sigma2_i[iter, i] + 1 / tau2[iter-1]
                nu_sigma2_i[iter, i] = 1 / np.random.gamma(1, 1 / sc)
                
            # Update tau2
            sc = np.sum(1 / nu_sigma2_i[iter, :]) + 1 / nu_tau2[iter-1]
            tau2[iter] = 1 / np.random.gamma(0.5 * (N + 1), 1 / sc)
            
            # Update nu_tau2
            sc = 1 / tau2[iter] + 1 / sigma2[iter-1]
            nu_tau2[iter] = 1 / np.random.gamma(1, 1 / sc)
            
            # Update sigma2
            resid = treatment_outcome_pre - control_outcome_pre @ alpha[iter, :]
            sc = 1 / nu_tau2[iter] + 0.5 * resid @ resid
            sigma2[iter] = 1 / np.random.gamma(0.5 * self.T0 + 1, 1 / sc)
            
            # Update nu_sigma2
            sc = 1 / sigma2[iter] + 1 / 10**2
            nu_sigma2[iter] = 1 / np.random.gamma(1, 1 / sc)
            
            # end of iteration
        self.alpha, self.sigma2 = alpha, sigma2
        pass

    def mcmc_spatial(self, max_iter=10000, burn_in=5000):
        N = self.N
        k = self.k
        W = self.adj_mat
        w = self.adj_vec
        T0 = self.T0
        X = self.X[:, :T0, :]
        control_outcome_pre = self.control_outcome[:, :T0]
        treatment_outcome_pre = self.treatment_outcome[:T0]
        
        beta, sigma2_i, nu_sigma2_i = np.zeros((max_iter, self.k)), np.zeros((max_iter, self.k)), np.zeros((max_iter, self.k))
        tau2, nu_tau2 = np.zeros(max_iter), np.zeros(max_iter)
        sigma2, nu_sigma2 = np.zeros(max_iter), np.zeros(max_iter)
        rho = np.zeros(max_iter)
        
        # TODO: Implement factor model
        
        sigma2_i[0, :], nu_sigma2_i[0, :] = 1, 1
        tau2[0], nu_tau2[0] = 1, 1
        sigma2[0], nu_sigma2[0] = 1, 1
        rho[0] = 0.5
        
        accept = 0
        
        for iter in range(1, max_iter):

            A_beta = np.zeros((k, k))
            for t in range(T0):
                A_beta += X[t, :, :].T @ X[t, :, :] + sigma2[iter-1] * np.diag(1 / sigma2_i[iter-1, :])

            B_beta = np.zeros(k)
            for t in range(T0):
                B_beta_tmp = control_outcome_pre[t, :] - rho[iter-1] * w * treatment_outcome_pre[t] - rho[iter-1] * W @ control_outcome_pre[t, :]
                B_beta += X[t, :, :].T @ B_beta_tmp

            beta_mean = np.linalg.inv(A_beta) @ B_beta
            beta_cov = sigma2[iter-1] * np.linalg.inv(A_beta)
            beta[iter, :] = np.random.multivariate_normal(beta_mean, beta_cov)

            for i in range(k):
                sc = 0.5 * beta[iter, i]**2 + \
                    1 / nu_sigma2_i[iter-1, i]
                sigma2_i[iter, i] = 1 / np.random.gamma(1, 1 / sc)

            for i in range(k):
                sc = 1 / sigma2_i[iter, i] + 1 / tau2[iter-1]
                nu_sigma2_i[iter, i] = 1 / np.random.gamma(1, 1 / sc)

            sc = 1 / nu_tau2[iter-1] + (1 / nu_sigma2_i[iter, :]).sum()
            tau2[iter] = 1 / np.random.gamma(1, 1 / sc)

            sc = 1 / tau2[iter] + 1 / sigma2[iter-1]
            nu_tau2[iter] = 1 / np.random.gamma(1, 1 / sc)

            c = 1 / 2

            rho_propose = rho[iter-1] + c * np.random.normal()

            num = np.zeros(T0)
            denom = np.zeros(T0)

            # denominator
            tmp = np.eye(N) - rho[iter-1] * W
            det_tmp = np.linalg.det(tmp)
            for t in range(T0):
                tmp2 = rho[iter-1] * w * treatment_outcome_pre[t] + X[t, :, :] @ beta[iter, :]
                denom[t] = det_tmp * np.exp(0.5 * ((np.eye(N) - rho[iter-1] * W) @ control_outcome_pre[t, :] - tmp2).T @ (
                    (np.eye(N) - rho[iter-1] * W) @ control_outcome_pre[t, :] - tmp2) / sigma2[iter-1])

            # numerator
            tmp = np.eye(N) - rho_propose * W
            det_tmp = np.linalg.det(tmp)
            for t in range(T0):
                tmp2 = rho_propose * w * treatment_outcome_pre[t] + X[t, :, :] @ beta[iter, :]
                num[t] = det_tmp * np.exp( 0.5 * ((np.eye(N) - rho_propose * W) @ control_outcome_pre[t, :] - tmp2).T @ (
                    (np.eye(N) - rho_propose * W) @ control_outcome_pre[t, :] - tmp2) / sigma2[iter-1])

            r = num.prod() / denom.prod()
            u = np.random.uniform()
            ap = np.min([r, 1])

            if u < ap:
                rho[iter] = rho_propose
                accept += 1
            else:
                rho[iter] = rho[iter-1]

            resid = np.zeros(T0)
            for t in range(T0):
                tmp = control_outcome_pre[t, :] - (rho[iter] * w * treatment_outcome_pre[t] + rho[iter]
                                        * W @ control_outcome_pre[t, :] + X[t, :, :] @ beta[iter, :])
                resid[t] = tmp.T @ tmp

            sc = 1 / nu_tau2[iter] + 1 / \
                nu_sigma2[iter-1] + 0.5 * resid.sum()
            sigma2[iter] = 1 / np.random.gamma(0.5 * T0 * N + 1, 1 / sc)

            sc = 1 / sigma2[iter] + 1 / 10**2
            nu_sigma2[iter] = 1 / np.random.gamma(1, 1 / sc)
            
        self.beta, self.sigma2 = beta, sigma2
        self.rho, self.accept = rho, accept
        pass
    
    def mcmc(self, max_iter=10000, burn_in=5000):
        print("Validating inputs...")
        self._validate_inputs()
        print("Preprocessing data...")
        self.preprocess_data()
        print("MCMC for weights started")
        self.mcmc_weight(max_iter, burn_in)
        print("MCMC for spatial model started")
        self.mcmc_spatial(max_iter, burn_in)
        print("MCMC completed!")
        pass
    
    def calculate_effects(self):
        # Calculate treatment effects
        T = self.T
        N = self.N
        k = self.k
        treatment_outcome = self.treatment_outcome
        control_outcome = self.control_outcome
        w = self.adj_vec
        W = self.adj_mat
        
        rho = self.rho[self.burn_in:]
        alpha = self.alpha[self.burn_in:]
        
        te = np.zeros((len(rho), self.T))
        spillover = np.zeros((len(rho), self.N, self.T))
        
        for iter in range(len(rho)):
            for t in range(T):
                np.linalg.inv(np.eye(N) - rho[iter] * w * alpha[iter, :].T - rho[iter] * W)  @ ((np.eye(N) - rho[iter] * W) @ control_outcome[:, t] - rho[iter] * w * treatment_outcome[t])
                te[iter, t] = treatment_outcome[t] - alpha[iter, :] @ np.linalg.inv(np.eye(N) - rho[iter] * w * alpha[iter, :].T - rho[iter] * W) @ ((np.eye(N) - rho[iter] * W) @ control_outcome[:, t] - rho[iter] * w * treatment_outcome[t])
                spillover[iter, :, t] = control_outcome[:, t] -  np.linalg.inv(np.eye(N) - rho[iter] * w * alpha[iter, :] - rho[iter] * W) @ ((np.eye(N) - rho[iter] * W) @ control_outcome[:, t] - rho[iter] * w * treatment_outcome[t])
                
        return te.mean(axis=0), spillover.mean(axis=0)
