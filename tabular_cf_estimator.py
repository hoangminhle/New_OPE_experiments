import numpy as np
import torch
import pdb
import math
import torch.nn.functional as F
from torch.optim import  RMSprop, Adam
from lbfgs import LBFGS
from scipy import optimize
import scipy
dtype = torch.double
class Tabular_State_MVM_Estimator(object):
    def __init__(self,behavior_data, config, ground_truth = None):
        self.behavior_data = behavior_data
        self.config  =config
        self.true = ground_truth
        self.pi = self.true.pi; self.mu = self.true.mu
        (self.s, self.a, self.sn, self.r) = behavior_data        
        self.n_states = self.pi.shape[0]; self.n_actions = self.pi.shape[1]; self.n_trajectories = self.s.shape[0]
        self.horizon = self.s.shape[1]; self.gamma = self.config.gamma
        self.horizon_normalization = (1-self.gamma**self.horizon)/(1-self.gamma) if self.gamma <1 else self.horizon
        self.true_P = self.true.P.reshape(self.n_states, self.n_actions, self.n_states)
        self.discount = torch.zeros(self.horizon, dtype=dtype)
        for t in range(self.horizon): self.discount[t] = self.gamma**t
        if not self.config.limited_w_rep: 
            self.dim_w = self.n_states 
        else:
            self.dim_w = int(np.ceil(np.log(self.n_states)/np.log(2)))
        if not self.config.limited_v_rep:
            self.dim_v = self.n_states
        else:
            self.dim_v = int(np.ceil(np.log(self.n_states)/np.log(2)))
        binary_dim = int(np.ceil(np.log(self.n_states)/np.log(2)))
        self.Phi_binary = torch.zeros(self.n_states, binary_dim, dtype=dtype)
        for s in range(self.n_states):
            self.Phi_binary[s,:] = torch.tensor(list(map(int,list(bin(s)[2:].zfill(binary_dim)))), dtype=dtype)

        self.Phi = torch.eye(self.n_states, dtype = dtype)
        self.I_v = torch.eye(self.dim_v, dtype=dtype)
        self.I_w = torch.eye(self.dim_w, dtype=dtype)
        self.I_s = torch.eye(self.n_states, dtype=dtype)
        self.estimate_empirical_distribution()
        self.model_based()
        self.feature_aggregation()
    # function to generate features
    def phi_w(self,s):
        if self.config.limited_w_rep:
            return self.Phi_binary[s,:]
        else:
            return self.Phi[s,:]
    def phi_v(self,s):
        if self.config.limited_v_rep:
            return self.Phi_binary[s,:]
        else:
            return self.Phi[s,:]
    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)        
    def feature_aggregation(self):
        #* bias objective components
        self.M_bias_1 = torch.zeros(self.dim_w, self.dim_v, dtype=dtype)
        self.M_bias_2 = torch.zeros(self.dim_v,1,dtype=dtype)
        self.M_bias_3 = torch.zeros(self.dim_w, self.dim_v, dtype=dtype)
        #* w-estimator component
        self.M_r = torch.zeros(self.dim_w,1, dtype=dtype)
        #* v_estimator component
        self.M_d0 = torch.zeros(self.dim_v, 1, dtype=dtype)
        #* td error component
        self.X_td = torch.zeros(self.dim_v, self.dim_v, dtype = dtype)
        self.y_td = torch.zeros(self.dim_v, 1, dtype=dtype)
        self.estimated_terminal_r = torch.matmul(torch.matrix_power(self.P_pi,self.horizon), (self.R*self.pi).sum(dim=1))
        self.D = torch.zeros(self.dim_v, self.dim_v, dtype = dtype)
        for episode in range(self.n_trajectories):
            t = 0
            s0 = self.s[episode][0]
            self.M_d0 += self.phi_v(s0).view(-1,1)
            for (s,a,sn,r) in zip(self.s[episode], self.a[episode], self.sn[episode], self.r[episode]):
                self.M_bias_1 += self.gamma**t*torch.ger(self.phi_w(s), self.gamma*self.rho[s,a]*self.phi_v(sn)-self.phi_v(s))
                self.M_r += self.gamma**t*self.rho[s,a]*r*self.phi_w(s).view(-1,1)
                if self.config.reg_true_rho:
                    self.X_td += self.gamma**t*torch.ger(self.phi_v(s), self.phi_v(s) - self.gamma*self.rho[s,a]*self.phi_v(sn))
                    self.D += self.gamma**t*torch.ger(self.phi_v(s), self.phi_v(s))
                    self.y_td[:,0] += self.gamma**t *self.pi[s,a]/self.mu[s,a]*r*self.phi_v(s)
                else:
                    self.X_td += self.gamma**t*torch.ger(self.phi_v(s), self.phi_v(s) - self.gamma*self.rho_hat[s,a]*self.phi_v(sn))
                    self.y_td[:,0] += self.gamma**t *self.rho_hat[s,a]*r*self.phi_v(s)
                    self.D += self.gamma**t*torch.ger(self.phi_v(s), self.phi_v(s))
                t+=1
                self.y_td[:,0] -= self.gamma**self.horizon*self.estimated_terminal_r[s]*self.phi_v(s)
            
            self.M_bias_3 += self.gamma**self.horizon*torch.ger(self.phi_w(sn), self.phi_v(sn))
            self.M_bias_2 += self.phi_v(s0).view(-1,1)
        self.M_bias_1 /= (self.horizon_normalization*self.n_trajectories)
        self.M_bias_2 /= (self.horizon_normalization*self.n_trajectories)
        self.M_bias_3 /= (self.horizon_normalization*self.n_trajectories)
        self.X_td /= (self.horizon_normalization*self.n_trajectories)
        self.y_td /= (self.horizon_normalization*self.n_trajectories)
        self.D /= (self.horizon_normalization*self.n_trajectories)

        self.M_r /= self.n_trajectories
        self.M_d0 /= self.n_trajectories
        if not self.config.estimate_d0:
            self.M_bias_2[:,0] = self.true.d0/self.horizon_normalization
            self.M_d0[:,0] = self.true.d0.clone().detach()
        self.X_bias = self.M_bias_1-self.M_bias_3
        self.y_bias = self.M_bias_2

        self.D = 1/self.D; self.D[torch.isinf(self.D)]=0 #* D_mu inverse
        self.v_lstd = torch.matmul(torch.pinverse(self.X_td),self.y_td)
        #* biasing the td ball?
        # self.X_td += 1e-5*self.D.sqrt()
        # self.X_td += 1e-4*self.I_v
        # self.X_td += min(self.d_mu[self.d_mu>0])*self.I_v
        # pdb.set_trace()
        try:
            self.v0 = torch.matmul(torch.inverse(self.X_td),self.y_td) #mid-point solution
        except:
            self.v0 = torch.matmul(torch.pinverse(self.X_td),self.y_td) #mid-point solution
        
        M = torch.mm(torch.mm(self.X_td.t(), self.D),self.X_td)
        self.reg = max(self.td_error(self.true.v_pi.view(-1,1)), self.td_error(self.v_mb))
        try:
            self.M_inv = torch.inverse(M)
        except:
            self.M_inv = torch.pinverse(M)
    def estimate_empirical_distribution(self):
        self.d_mu_count = torch.zeros(self.n_states, dtype=dtype)
        P_count = torch.zeros(self.n_states,self.n_actions, self.n_states, dtype=dtype)
        d0_count = torch.zeros(self.n_states)
        R_sum = torch.zeros(self.n_states, self.n_actions, dtype=dtype)
        R_count = torch.zeros(self.n_states, self.n_actions)
        mu_state_action_count = torch.zeros(self.n_states, self.n_actions)
        dH_count = torch.zeros(self.n_states)
        for episode in range(self.n_trajectories):
            t=0
            s0 = self.s[episode][0]
            d0_count[s0] += 1
            for (s,a,sn,r) in zip(self.s[episode], self.a[episode], self.sn[episode], self.r[episode]):
                self.d_mu_count[s] += self.gamma**t
                P_count[s,a,sn] += 1
                R_sum[s, a] += r
                R_count[s, a] += 1
                mu_state_action_count[s,a] += 1
                t += 1
                if t == self.horizon:
                    dH_count[sn] += 1
        assert torch.sum(d0_count) == self.n_trajectories, 'Wrong count on initial state distribution'        
        assert torch.sum(P_count) == self.n_trajectories*self.horizon
        self.d_mu = F.normalize(self.d_mu_count, p=1, dim = 0).type(dtype)
        self.P = F.normalize(P_count,p=1,dim=2).type(dtype)
        self.d_mu_H = F.normalize(dH_count, p=1, dim=0).type(dtype)
        #! speifying that terminal states will self-loop
        self.P[-1,:,-1] = 1
        if not self.config.estimate_d0:
            self.d0 = self.true.d0
        else:
            self.d0 = F.normalize(d0_count,p=1,dim=0).type(dtype)
        self.R = R_sum / R_count
        self.R[self.R != self.R] = 0
        self.mu_hat = F.normalize(mu_state_action_count, p=1, dim=1).type(dtype)

        #* setting rho and setting inf to zero?
        self.rho_hat = self.pi/self.mu_hat; self.rho_hat[torch.isinf(self.rho_hat)] = 0; self.rho_hat[torch.isnan(self.rho_hat)] = 0
        if self.config.estimate_rho:
            self.rho = self.pi/self.mu_hat; self.rho[torch.isinf(self.rho)] = 0; self.rho[torch.isnan(self.rho)] = 0
        else:
            self.rho = self.pi/self.mu; self.rho[torch.isinf(self.rho)] = 0; self.rho[torch.isnan(self.rho)] = 0
        self.P_pi = torch.einsum('san,sa->sn', (self.P,self.pi))
        self.w_pi = self.true.d_pi/self.d_mu; self.w_pi[torch.isinf(self.w_pi)]=0; self.w_pi[self.w_pi!=self.w_pi]=0;
        self.w_star = self.true.d_pi/self.true.d_mu; self.w_star[torch.isinf(self.w_star)]=0

        # Calculate d_pi_H
        if not self.config.estimate_d0:
            d_pi_t = self.true.d0.clone().detach()
        else:
            d_pi_t = self.d0.clone().detach()
        for h in range(self.horizon):
            d_pi_t = torch.matmul(self.P_pi.t(), d_pi_t)
        self.d_pi_H = d_pi_t

    def w_estimator(self,w):
        return torch.mm(w.t(), self.M_r)[0,0]

    def v_estimator(self,v):
        return torch.mm(self.M_d0.t(), v)[0,0]

    def td_error(self,v):
        td = torch.mm(self.X_td, v)-self.y_td
        td_error = torch.mm(torch.mm(td.t(), self.D), td)
        return td_error
    def minimize_bias_cf_scipy(self):
        def objective_scipy(w):
            w = torch.tensor(w, dtype=dtype).view(-1,1)
            f_w = torch.mm(w.t(), self.X_bias)+ self.y_bias.t()
            L = torch.norm(f_w)
            return L.data.numpy().astype(np.float64)
        def jac_scipy(w):
            w = torch.tensor(w,dtype=dtype).view(-1,1)
            w.requires_grad = True
            f_w = torch.mm(w.t(), self.X_bias)+ self.y_bias.t()
            L = torch.norm(f_w)
            L.backward()
            return w.grad.data.numpy().astype(np.float64)[:,0]
        min_w = optimize.minimize(objective_scipy, torch.rand(self.dim_w,dtype=dtype).data.numpy().astype(np.float64), method="BFGS", jac = jac_scipy, options={'gtol':1e-12, 'maxiter':100000,'disp':True, 'return_all': True})
        returned_w = torch.tensor(min_w.x, dtype=dtype).view(-1,1)
        if self.config.print_progress:
            print('Objective value:', objective_scipy(returned_w[:,0]))
            print('Estimator value:', self.w_estimator(returned_w))
        self.w_bias_scipy = returned_w.clone().detach()
        return self.w_estimator(returned_w)

    def minimize_bias_td_cf_scipy(self):
        opt_w = torch.rand(self.dim_w, 1,dtype=dtype)
        def objective_scipy(w):
            w = torch.tensor(w, dtype=dtype).view(-1,1)
            f_w = torch.mm(w.t(), self.X_bias)+ self.y_bias.t()
            L = (torch.abs(torch.mm(f_w, self.v0))+math.sqrt(self.reg*torch.mm(torch.mm(f_w, self.M_inv), f_w.t())))**2
            return L.data.numpy().astype(np.float64)[0,0]
        def jac_scipy(w):
            w = torch.tensor(w,dtype=dtype).view(-1,1)
            w.requires_grad = True
            f_w = torch.mm(w.t(), self.X_bias)+ self.y_bias.t()
            L = (torch.abs(torch.mm(f_w, self.v0))+math.sqrt(self.reg*torch.mm(torch.mm(f_w, self.M_inv), f_w.t())))**2
            L.backward()
            return w.grad.data.numpy().astype(np.float64)[:,0]
        min_w = optimize.minimize(objective_scipy, opt_w[:,0].data.numpy().astype(np.float64), method="BFGS", jac = jac_scipy, options={'gtol':1e-12, 'maxiter':10000,'disp':True, 'return_all': True})
        returned_w = torch.tensor(min_w.x, dtype=dtype).view(-1,1)
        if self.config.print_progress:
            print('Epsilon param:', self.reg)
            print('Objective value:', objective_scipy(returned_w[:,0]))
            print('Estimator value:', self.w_estimator(returned_w))
        self.w_bias_td_scipy = returned_w.clone().detach()
        return self.w_estimator(returned_w)

    def minimize_bias_td_var_cf_scipy(self):
        trajectory_reward_feature = ((self.discount*self.rho[self.s,self.a]*self.r)[:,:,None]*self.phi_w(self.s)).sum(dim=1).t()
        opt_w = torch.rand(self.dim_w, 1,dtype=dtype, requires_grad=True)
        assert abs(self.w_estimator(opt_w) - torch.mean(torch.mm(opt_w.t(), trajectory_reward_feature)))<1e-8
        assert torch.norm(self.M_r[:,0]/self.horizon_normalization - (trajectory_reward_feature/self.horizon_normalization).sum(dim=1)/self.n_trajectories)<1e-8

        #* creating bootstrap samples
        k=5000#self.n_trajectories
        average_reward_feature = torch.zeros(self.dim_w,k, dtype=dtype)
        for i in range(k):
            idx = torch.multinomial(torch.ones(self.n_trajectories)/self.n_trajectories,self.n_trajectories, replacement = True)
            # average_reward_feature[:,i] = torch.mean(trajectory_reward_feature[:,idx]/self.horizon_normalization, dim=1)
            average_reward_feature[:,i] = torch.mean(trajectory_reward_feature[:,idx], dim=1)
        

        def objective_scipy(w):
            w = torch.tensor(w, dtype=dtype).view(-1,1)
            f_w = torch.mm(w.t(), self.X_bias)+ self.y_bias.t()
            bias = (torch.abs(torch.mm(f_w, self.v0))+math.sqrt(self.reg*torch.mm(torch.mm(f_w, self.M_inv), f_w.t())))**2
            variance = 1/k*((torch.mm(w.t(), average_reward_feature) - torch.mean(torch.mm(w.t(), trajectory_reward_feature)))**2).sum()
            L = bias+variance
            return L.data.numpy().astype(np.float64)[0,0]
        def jac_scipy(w):
            w = torch.tensor(w,dtype=dtype).view(-1,1)
            w.requires_grad = True
            f_w = torch.mm(w.t(), self.X_bias)+ self.y_bias.t()
            bias = (torch.abs(torch.mm(f_w, self.v0))+math.sqrt(self.reg*torch.mm(torch.mm(f_w, self.M_inv), f_w.t())))**2
            variance = 1/k*((torch.mm(w.t(), average_reward_feature) - torch.mean(torch.mm(w.t(), trajectory_reward_feature)))**2).sum()
            L = bias+variance
            L.backward()
            return w.grad.data.numpy().astype(np.float64)[:,0]
        min_w = optimize.minimize(objective_scipy, opt_w[:,0].data.numpy().astype(np.float64), method="BFGS", jac = jac_scipy, options={'gtol':1e-12, 'maxiter':100000,'disp':True, 'return_all': True})
        self.w_td_var = torch.tensor(min_w.x, dtype=dtype).view(-1,1)
        if self.config.print_progress:
            print('Epsilon param:', self.reg)
            print('Objective value:', objective_scipy(self.w_td_var[:,0]))
            print('Estimator value:', self.w_estimator(self.w_td_var))
        return self.w_estimator(self.w_td_var)

    def minimize_lbfgs(self, objective):
        w = torch.rand(self.dim_w, 1,dtype=dtype, requires_grad=True)
        optimizer_w = LBFGS([w], lr=1, max_iter=10000, max_eval=15000, tolerance_grad=1e-09, tolerance_change=1e-11, history_size=100, line_search_fn='strong_wolfe')
        n_iterations = 10
        L_min = 1e10; i_min=0;
        trailing_grad_norm = 0
        trailing_objective = 0
        if objective == 'bias_td_var_opt_cf' or objective == 'bias_td_var':
            trajectory_reward_feature = ((self.discount*self.rho[self.s,self.a]*self.r)[:,:,None]*self.phi_w(self.s)).sum(dim=1).t()
            trajectory_reward_feature_hat = ((self.discount*self.rho_hat[self.s,self.a]*self.r)[:,:,None]*self.phi_w(self.s)).sum(dim=1).t()
            #* creating bootstrap samples
            k=5000#self.n_trajectories
            average_reward_feature = torch.zeros(self.dim_w,k, dtype=dtype)
            for i in range(k):
                idx = torch.multinomial(torch.ones(self.n_trajectories)/self.n_trajectories,self.n_trajectories, replacement = True)
                # average_reward_feature[:,i] = torch.mean(trajectory_reward_feature[:,idx]/self.horizon_normalization, dim=1)
                average_reward_feature[:,i] = torch.mean(trajectory_reward_feature[:,idx], dim=1)

        
        def closure():
            optimizer_w.zero_grad()
            f_w = torch.mm(w.t(), self.X_bias)+ self.y_bias.t()
            if objective == 'bias_opt_cf' or objective == 'bias':
                loss = torch.mm(f_w, f_w.t())
            elif objective == 'bias_td_opt_cf' or objective == 'bias_td':
                loss = (torch.abs(torch.mm(f_w, self.v0))+math.sqrt(self.reg*torch.mm(torch.mm(f_w, self.M_inv), f_w.t())))**2
            elif objective == 'bias_td_var_opt_cf' or objective == 'bias_td_var':
                bias = (torch.abs(torch.mm(f_w, self.v0))+math.sqrt(self.reg*torch.mm(torch.mm(f_w, self.M_inv), f_w.t())))**2
                variance = 1/2*torch.var(torch.mm(w.t(), average_reward_feature)) #/ self.horizon_normalization**2
                # variance = 1.0/k*((torch.mm(w.t(), average_reward_feature) - torch.mean(torch.mm(w.t(), trajectory_reward_feature)))**2).sum()
                loss = bias + variance
            loss.backward()
            return loss
        # pdb.set_trace()
        for i in range(n_iterations):
            L = optimizer_w.step(closure)
            trailing_objective = 1/(i+1)*L + i / (i+1)*trailing_objective
            if L<L_min: L_min = L; w_min = w.clone().detach(); i_min=i
            trailing_grad_norm = 1/(i+1)*torch.norm(w.grad) + i/(i+1)*trailing_grad_norm
            w_estimator = self.w_estimator(w)
            if i%100 ==0 and self.config.print_progress:
                print('\n')
                print('opt objective', objective)
                print('iteration ', i)
                print('trailing objective:', trailing_objective)
                print('current w estimator: ', w_estimator)
                print('reg:', self.reg)
                print('current objective:', L)
                print('min objective:', L_min)
                print('min iteration:', i_min)
                print('w min estimator:', self.w_estimator(w_min))
        return self.w_estimator(w_min)

    def optimize(self, objective):
        if objective == 'bias_opt_cf' or objective == 'bias':
            w_estimator = self.minimize_lbfgs(objective)
            return w_estimator
        elif objective == 'bias_scipy_cf':
            w_estimator = self.minimize_bias_cf_scipy()
            return w_estimator
        elif objective == 'bias_cf':
            w = torch.mm(torch.inverse(torch.mm(self.X_bias, self.X_bias.t()) + self.config.eps_matrix_invert*self.I_w), torch.mm(self.X_bias, -self.y_bias))
            w_estimator = self.w_estimator(w)
            return w_estimator
        elif objective == 'bias_td_opt_cf' or objective == 'bias_td':
            try:
                w_estimator = self.minimize_lbfgs(objective)
            except:
                # pdb.set_trace()
                print('lbfgs error in pytorch')
                w_estimator = self.minimize_bias_td_cf_scipy()
            return w_estimator
        elif objective == 'bias_td_scipy_cf':
            w_estimator = self.minimize_bias_td_cf_scipy()
            return w_estimator
        elif objective == 'bias_td_var_opt_cf' or objective == 'bias_td_var':
            try:
                w_estimator = self.minimize_lbfgs(objective)
            except:
                # pdb.set_trace()
                print('lbfgs error in pytorch')
                w_estimator = self.minimize_bias_td_var_cf_scipy()
            return w_estimator
        elif objective == 'bias_td_var_scipy_cf':
            w_estimator = self.minimize_bias_td_var_cf_scipy(objective)
            return w_estimator
        elif objective == 'TD-ball center':
            estimator = self.v_estimator(self.v0)
            return estimator
        elif objective == 'LSTD':
            estimator = self.v_estimator(self.v_lstd)
            return estimator
        elif objective == 'LSTDQ':
            estimator = self.lstdq()
            return estimator
        elif objective == 'MWL':
            estimator = self.mwl()
            return estimator

    def model_based(self):
        Rpi = (self.R*self.pi).sum(dim=1).view(-1,1)
        # Rpi = torch.sum(self.R*self.pi, dim=1)
        v_s = torch.zeros(self.n_states,1, dtype=dtype)
        for s in range(self.n_states):
            dt = torch.zeros(self.n_states,1,dtype=dtype)
            dt[s,0] = 1.0
            ds = torch.zeros(self.n_states,1, dtype=dtype)
            discounted_t = 1.0
            for h in range(self.horizon):
                ds += dt*discounted_t
                dt = torch.mm(self.P_pi.T, dt)
                discounted_t *=self.gamma
            v_s[s,0] += torch.mm(ds.t(), Rpi)[0,0]
        
        self.v_mb = v_s
        return v_s
    def mwl(self):
        # implement mwl for tabular specific case
        dim_sa = self.n_states*self.n_actions
        X = torch.zeros(dim_sa, dim_sa)        
        y = torch.zeros(dim_sa,1)
        reward = torch.zeros(dim_sa,1)
        for episode in range(self.n_trajectories):
            t = 0
            s0 = self.s[episode][0]
            for (s,a,sn,r) in zip(self.s[episode], self.a[episode], self.sn[episode], self.r[episode]):
                X[s*self.n_actions+a, s*self.n_actions+a] += self.gamma**t
                X[sn*self.n_actions:(sn+1)*self.n_actions, s*self.n_actions+a] -= self.gamma**(t+1)*self.pi[sn,:]
                reward[s*self.n_actions+a,0] += self.gamma**t *r
                t+=1
            y[s0*self.n_actions:(s0+1)*self.n_actions,0] += self.pi[s0,:]
            y[sn*self.n_actions:(sn+1)*self.n_actions,0] -= self.gamma**self.horizon*self.pi[sn,:]
        X /= (self.horizon_normalization*self.n_trajectories)
        y /= (self.horizon_normalization*self.n_trajectories)
        reward/= self.n_trajectories
        w = torch.mm(torch.inverse(X+self.config.eps_matrix_invert*torch.eye(dim_sa)), y)
        # w = torch.mm(torch.inverse(torch.mm(X.t(), X)+EPS_MATRIX_INVERT*torch.eye(dim_sa)), torch.mm(X.t(), y))
        w_estimator = torch.mm(w.t(), reward)[0,0]
        return w_estimator
    
    def lstdq(self):
        dim_sa = self.n_states*self.n_actions
        X = torch.zeros(dim_sa, dim_sa)        
        y = torch.zeros(dim_sa,1)
        d0 = torch.zeros(dim_sa,1)
        for episode in range(self.n_trajectories):
            t = 0
            s0 = self.s[episode][0]
            for (s,a,sn,r) in zip(self.s[episode], self.a[episode], self.sn[episode], self.r[episode]):
                X[s*self.n_actions+a, s*self.n_actions+a] += self.gamma**t # *torch.ger(self.phi_q(s,a), self.phi_q(s,a) - self.gamma*self.phi_q_s(sn))
                X[s*self.n_actions+a, sn*self.n_actions:(sn+1)*self.n_actions] -= self.gamma**(t+1)*self.pi[sn,:]
                y[s*self.n_actions+a,0] += self.gamma**t *r
                t+=1
            d0[s0*self.n_actions:(s0+1)*self.n_actions,0] += self.pi[s0,:]
        X /= (self.horizon_normalization*self.n_trajectories)
        y /= (self.horizon_normalization*self.n_trajectories)
        d0/= self.n_trajectories
        q_lstd = torch.mm(torch.inverse(X+self.config.eps_matrix_invert*torch.eye(dim_sa)),y)
        q_estimator = torch.mm(d0.t(), q_lstd)[0,0]
        return q_estimator
        
        
