import numpy as np
import torch
import pdb
import math
import torch.nn.functional as F
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from torch.optim import  LBFGS, RMSprop
from optimizer import ModifiedAdam, OptimisticAdam
import quadprog
from scipy import optimize

dtype = torch.double
USE_KNOWN_D0 = False
EPS_CLOSED_FORM_W = 1e-8
class Tabular_State_MVM_Estimator(object):
    def __init__(self,behavior_data, config, logging=None, ground_truth = None):
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
        self.r_discounted = self.r*self.discount
        self.I = torch.eye(self.n_states, dtype=dtype)
        self.Phi = torch.eye(self.n_states, dtype = dtype)
        if logging: self.writer = SummaryWriter('log_expt/')
        self.estimate_empirical_distribution()
    # function to generate features
    def phi(self,s):
        return self.Phi[s,:]

    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)        

    def estimate_empirical_distribution(self):
        self.d_mu_count = torch.zeros(self.n_states, dtype=dtype)
        P_count = torch.zeros(self.n_states,self.n_actions, self.n_states, dtype=dtype)
        d0_count = torch.zeros(self.n_states)
        R_sum = torch.zeros(self.n_states, self.n_actions, dtype=dtype)
        R_count = torch.zeros(self.n_states, self.n_actions)
        mu_state_action_count = torch.zeros(self.n_states, self.n_actions)
        time_indexing_frequency = torch.zeros(self.n_states)
        self.max_rewards_to_go = torch.zeros(self.n_states, dtype=dtype)
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
                if sn == self.n_states-1:
                    reward_to_go_sn = 0
                else:
                    reward_to_go_sn = (1-self.gamma**(self.horizon-t-1))/(1-self.gamma) if self.gamma<1 else self.horizon-t-1
                self.max_rewards_to_go[sn] += reward_to_go_sn
                if s == self.n_states-1:
                    reward_to_go_s = 0
                else:
                    reward_to_go_s = (1-self.gamma**(self.horizon-t))/(1-self.gamma) if self.gamma<1 else self.horizon-t
                self.max_rewards_to_go[s] += reward_to_go_s
                time_indexing_frequency[s] += 1
                time_indexing_frequency[sn] += 1
                t += 1
        self.max_rewards_to_go = self.max_rewards_to_go / time_indexing_frequency        
        self.max_rewards_to_go[self.max_rewards_to_go != self.max_rewards_to_go] = 0 # assign nan to 0
        self.max_rewards_to_go[torch.isinf(self.max_rewards_to_go)] = 0 # assign inf to 0
        assert torch.sum(d0_count) == self.n_trajectories, 'Wrong count on initial state distribution'        
        assert torch.sum(P_count) == self.n_trajectories*self.horizon

        self.d_mu = F.normalize(self.d_mu_count, p=1, dim = 0).type(dtype)
        self.P = F.normalize(P_count,p=1,dim=2).type(dtype)
        #! speifying that terminal states will self-loop
        self.P[-1,:,-1] = 1
        #! assign uniform random for unseen state, action pairs (be careful here)
        unseen_sa = torch.where(self.P.sum(dim=2)==0)
        for (s,a) in zip(unseen_sa[0], unseen_sa[1]):
            self.P[s,a] = torch.ones(self.n_states)/self.n_states
        # pdb.set_trace()
        # if USE_KNOWN_D0:
        #     self.d0 = self.true.d0
        # else:
        self.d0 = F.normalize(d0_count,p=1,dim=0).type(dtype)
        self.R = R_sum / R_count
        self.R[self.R != self.R] = 0
        self.mu_hat = F.normalize(mu_state_action_count, p=1, dim=1).type(dtype)
        #! for absorbing state, just set estimated mu to the true mu
        # self.mu_hat[-1,:] = self.mu[-1,:].clone()

        #* setting rho and setting inf to zero?
        if self.config.estimate_mu:
            self.rho = self.pi / self.mu_hat
            self.P_mu = torch.einsum('sa,san->sn', (self.mu_hat, self.P))
        else:
            self.rho = self.pi / self.mu
            self.P_mu = torch.einsum('sa,san->sn', (self.mu, self.P))
        self.rho[torch.isinf(self.rho)] = 0 # assign inf to 0
        self.P_pi = torch.einsum('sa,san->sn', (self.pi, self.P))
        v_upper_bound = (self.R.max()*self.max_rewards_to_go).data.numpy().astype(np.float64)
        v_lower_bound = (self.R.min()*self.max_rewards_to_go).data.numpy().astype(np.float64)
        self.v_bounds = tuple( [ ( v_lower_bound[i], v_upper_bound[i]) for i in range(len(v_upper_bound)) ] )
        self.w_bounds = tuple( [ ( 0, 1e6) for i in range(self.n_states) ] )
        true_P_pi = torch.einsum('sa,san->sn', (self.pi, self.true.P))
        self.true_R_adjusted = torch.matmul(self.I - torch.matrix_power(self.gamma*true_P_pi, self.horizon) , (self.pi*self.true.R).sum(dim=1))
        self.w_pi = self.true.d_pi/self.d_mu; self.w_pi[torch.isinf(self.w_pi)]=0
        self.w_star = self.true.d_pi/self.true.d_mu; self.w_star[torch.isinf(self.w_star)]=0
        #* aggregating P_rho
        self.P_rho = torch.zeros(self.n_states, self.n_states, dtype=dtype)
        rho_count = torch.zeros(self.n_states, self.n_states, dtype=dtype)
        mu_count = torch.zeros(self.n_states, dtype=dtype)
        for episode in range(self.n_trajectories):
            for (s,a,sn,r) in zip(self.s[episode], self.a[episode], self.sn[episode], self.r[episode]):
                self.P_rho[s,sn] += self.rho[s,a]
                rho_count[s,sn] += 1
                mu_count[s] += 1
        # self.P_rho = self.P_rho / rho_count; self.P_rho[self.P_rho!=self.P_rho] = 0; self.P_rho[torch.isinf(self.P_rho)] = 0
        # for s in range(self.n_states):
        #     self.P_rho[s,:] = self.P_rho[s,:]/mu_count[s]
        # # self.P_rho = self.P_rho / mu_count[:,None]; 
        # self.P_rho[self.P_rho!=self.P_rho] = 0; self.P_rho[torch.isinf(self.P_rho)] = 0
        self.P_rho = torch.einsum('san,sa->sn', self.P, self.pi)
        self.D_mu = torch.diag(self.d_mu)
        self.Rpi = (self.R*self.pi).sum(dim=1)
        # empirical_r_pi_adjusted = np.dot((np.identity(n_states) - np.linalg.matrix_power(gamma*empirical_P_pi, horizon)),empirical_r_pi)
        self.Rpi_adjusted = torch.matmul(self.I-torch.matrix_power(self.gamma*self.P_pi,self.horizon), self.Rpi)

        # Calculate d_pi_H
        if USE_KNOWN_D0:
            d_pi_t = self.true.d0.clone().detach()
        else:
            d_pi_t = self.d0.clone().detach()
        for h in range(self.horizon):
            d_pi_t = torch.matmul(self.P_pi.t(), d_pi_t)
        self.d_pi_H = d_pi_t


    def bias_check(self):
        r_pi = torch.zeros(self.n_states, dtype=dtype)
        td = torch.zeros(self.n_states, dtype=dtype)
        count = torch.zeros(self.n_states, dtype=dtype)
        for episode in range(self.n_trajectories):
            t=0
            for (s,a,sn,r) in zip(self.s[episode], self.a[episode], self.sn[episode], self.r[episode]):
                r_pi[s] += self.rho[s,a]*r
                td[s] += self.true.v_pi[s] - self.gamma*self.rho[s,a]*self.true.v_pi[sn]
                count[s] += 1
        pdb.set_trace()
        
    def model_based(self):
        Rpi = (self.R*self.pi).sum(dim=1)
        v_s = torch.zeros(self.n_states,dtype=dtype)
        for s in range(self.n_states):
            dt = torch.zeros(self.n_states,dtype=dtype)
            dt[s] = 1.0
            ds = torch.zeros(self.n_states, dtype=dtype)
            discounted_t = 1.0
            for h in range(self.horizon):
                ds += dt*discounted_t
                dt = torch.matmul(self.P_pi.T, dt)
                discounted_t *=self.gamma
            v_s[s] += torch.matmul(ds, Rpi)

        return v_s

    def w_estimator(self,w):
        trajectory_reward = (w[self.s]*self.rho[self.s, self.a]*self.r_discounted).sum(dim=1)
        w_based_estimate = torch.mean(trajectory_reward)
        return w_based_estimate

    # def w_estimator_alternative(self,w):

    def self_normalized_w_estimator(self,w):
        trajectory_reward = (w[self.s]*self.rho[self.s, self.a]*self.r_discounted).sum()
        normalization = (w[self.s]*self.rho[self.s, self.a]*(torch.ones_like(self.r)*self.discount)).sum()
        w_based_estimate = trajectory_reward / normalization*self.horizon_normalization
        return w_based_estimate

    def v_estimator(self,v):
        if USE_KNOWN_D0:
            v_based_estimate = torch.matmul(v, self.true.d0)
        else:
            s0 = self.s[:,0]
            v_based_estimate = torch.mean(v[s0])
        return v_based_estimate

    def bias_squared_population(self,w,v):
        bias = 0
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for sn in range(self.n_states):
                    bias += w[s]*(-v[s] + self.gamma*self.pi[s,a]/self.mu[s,a] * v[sn])*self.true.d_mu[s]*self.mu[s,a]*self.true.P[s,a,sn]
        
        bias += 1/self.horizon_normalization* torch.matmul(self.true.d0,v)
        bias -= 1/self.horizon_normalization * self.gamma**self.horizon*torch.matmul(self.true.d_pi_H,v)
        # pdb.set_trace()
        return bias**2

    def bias_squared_mb(self, w, v):
        bias = 0
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for sn in range(self.n_states):
                    if self.mu_hat[s,a] > 0:
                        bias += w[s]*(-v[s] + self.gamma*self.pi[s,a]/self.mu_hat[s,a] * v[sn])*self.d_mu[s]*self.mu_hat[s,a]*self.P[s,a,sn]
        bias += 1/self.horizon_normalization* torch.matmul(self.d0,v)
        bias -= 1/self.horizon_normalization * self.gamma**self.horizon*torch.matmul(self.true.d_pi_H,v)
        pdb.set_trace()
        return bias**2

    def f(self, w):
        weighted_vector = torch.matmul(w, torch.mm(self.D_mu, self.gamma*self.P_rho-self.I)) + self.d0/self.horizon_normalization
        weighted_vector += -1/self.horizon_normalization * self.gamma**self.horizon*self.d_pi_H
        return weighted_vector

    def bias_squared(self, w, v):
        self.f_w = self.f(w)
        bias_squared = torch.matmul(self.f_w, v)**2
        return bias_squared

    def bias_squared_vectorize_v(self, w, v_stack):
        self.f_w = self.f(w)
        bias_squared = torch.matmul(self.f_w, v_stack)**2
        return bias_squared

    def bias_squared_pool(self, w,v):
        # weighted_vector = torch.matmul(torch.mm(self.gamma*self.P_rho.t()-self.I, self.D_mu), w) + self.true.d0/self.horizon_normalization
        # weighted_vector = torch.matmul(torch.mm(self.gamma*self.P_rho-self.I, self.D_mu), w) + self.d0/self.horizon_normalization
        weighted_vector = torch.matmul(w, torch.mm(self.D_mu, self.gamma*self.P_rho-self.I)) + self.d0/self.horizon_normalization
        # weighted_vector += -1/self.horizon_normalization * self.gamma**self.horizon*self.true.d_pi_H
        weighted_vector += -1/self.horizon_normalization * self.gamma**self.horizon*self.d_pi_H
        bias_squared = torch.matmul(weighted_vector, v)**2
        return bias_squared

    def bias_squared_incremental(self,w,v):
        w_comp = (self.discount*w[self.s]*(self.gamma*self.rho[self.s, self.a]*v[self.sn] - v[self.s])).sum()
        if USE_KNOWN_D0:
            v_comp = torch.matmul(self.true.d0, v) * self.n_trajectories
        else:
            v_comp = v[self.s[:,0]].sum()
            # v_comp = torch.matmul(self.d0, v) * self.n_trajectories
        # extra_term = -1/self.horizon_normalization * self.gamma**self.horizon*torch.matmul(self.true.d_pi_H,v)
        extra_term = -1/self.horizon_normalization * self.gamma**self.horizon*torch.matmul(self.d_pi_H,v)
        # extra_term = 0
        bias = (w_comp + v_comp)  / self.n_trajectories / self.horizon_normalization + extra_term
        return bias**2


    # ##** This is for reference
    # def bias_squared(self, w,q):
    #     weighted_vector = torch.matmul(torch.mm(self.gamma*self.P_pi.t()-self.I, self.D_mu), w) + self.p0_pi/self.horizon_normalization
    #     bias_squared = torch.matmul(weighted_vector.t(), q)**2
    #     return bias_squared 

    # def f(self,w):
    #     w_comp = (self.discount*w[self.s]*(self.gamma*self.rho[self.s, self.a]*self.Phi[:,self.sn] - self.Phi[:,self.s])).sum(dim=(1,2))
    #     v_comp = self.Phi[:,self.s[:,0]].sum(dim=1)
    #     # extra_term = -1/self.horizon_normalization * self.gamma**self.horizon*self.true.d_pi_H
    #     extra_term = (-self.gamma**self.horizon*w[self.sn[:,-1]]*self.Phi[:,self.sn[:,-1]]).sum(dim=1)
    #     # extra_term = 0
    #     f_w = (w_comp + v_comp) + extra_term
    #     return f_w /self.n_trajectories

    # def bias_squared_vectorize(self, w, v_stack):
    #     w_comp = (self.discount[:,None]*w[self.s,None]*(self.gamma*self.rho[self.s, self.a,None]*v_stack[self.sn] - v_stack[self.s])).sum(dim=(0,1))
    #     if USE_KNOWN_D0:
    #         v_comp = torch.matmul(self.true.d0, v_stack)* self.n_trajectories
    #     else:
    #         v_comp = v_stack[self.s[:,0]].sum(dim=0)
    #     # pdb.set_trace()
    #     # extra_term = 0
    #     extra_term = -1/self.horizon_normalization * self.gamma**self.horizon*torch.matmul(self.true.d_pi_H,v_stack)
        
    #     bias = (w_comp + v_comp)  / self.n_trajectories / self.horizon_normalization+ extra_term
    #     return bias**2

    # def bias_squared_vectorize_w(self, w_stack, v):
    #     w_comp = (self.discount[:,None]*w_stack[self.s]*(self.gamma*self.rho[self.s, self.a,None]*v[self.sn, None] - v[self.s,None])).sum(dim=(0,1))
    #     if USE_KNOWN_D0:
    #         v_comp = torch.matmul(self.true.d0, v)* self.n_trajectories
    #     else:
    #         v_comp = v[self.s[:,0]].sum()
    #     extra_term = -1/self.horizon_normalization * self.gamma**self.horizon*torch.matmul(self.true.d_pi_H,v)
        
    #     bias = (w_comp + v_comp)  / self.n_trajectories / self.horizon_normalization+ extra_term
    #     return bias**2


    def mb_bellman_error(self, v):
        Rpi = (self.R*self.pi).sum(dim=1)
        bellman_error = (self.discount*(self.gamma*torch.matmul(self.P_pi[self.s], v) + Rpi[self.s] - v[self.s])**2).sum()/self.n_trajectories/self.horizon_normalization
        return bellman_error

    def td_error(self,v):
        td_error = v - self.Rpi_adjusted - self.gamma*torch.matmul(self.P_rho, v)
        return torch.matmul(td_error, torch.matmul(self.D_mu, td_error))        
    def td_error_vectorize(self, v_stack):
        td_error = v_stack - self.Rpi_adjusted[:,None] - self.gamma*torch.matmul(self.P_rho, v_stack)
        return torch.diagonal(torch.matmul(td_error.t(), torch.matmul(self.D_mu, td_error)))

    # def td_error(self, v):
    #     td_error = (self.discount*self.rho[self.s, self.a]*(self.gamma*v[self.sn] +self.R[self.s, self.a]- v[self.s])**2).sum()/self.n_trajectories/self.horizon_normalization
    #     return td_error

    # def td_error_vectorize(self, v_stack):
    #     td_error = (self.discount[:,None]*self.rho[self.s, self.a,None]*(self.gamma*v_stack[self.sn] +self.R[self.s, self.a,None]- v_stack[self.s])**2).sum(dim=(0,1))/self.n_trajectories/self.horizon_normalization
    #     return td_error

    def generate_random_v_class(self, cardinality):
        self.v_candidate = torch.zeros(self.n_states, cardinality, dtype = dtype)
        self.v_candidate[:,0] = self.true.v_pi.clone().detach()
        for i in range(1,cardinality):
            self.v_candidate[:,i] = (2*torch.rand(self.n_states,dtype=dtype)-1)+self.true.v_pi
        return self.v_candidate

    def generate_random_w_class(self, cardinality):
        self.w_candidate = torch.zeros(self.n_states, cardinality, dtype=dtype)
        self.w_candidate[:,0] = self.w_star.clone().detach()
        # delta = (self.w_bias_cf - self.w_pi)/cardinality
        for i in range(1, cardinality):
            w = torch.rand(self.n_states, dtype=dtype)*self.w_star.max()
            w = w / torch.matmul(w, self.true.d_mu)
            self.w_candidate[:,i] = w.clone().detach()
        return self.w_candidate

    def form_restricted_v_class(self, cardinality):
        td_error_vector = self.td_error_vectorize(self.v_candidate)
        topk = torch.topk(td_error_vector,k=cardinality,largest=False)[1]
        self.v_candidate_narrow = self.v_candidate[:, topk].clone().detach()
        # pdb.set_trace()

    def optimize_finite_class(self, objective, td_penalty = None):
        # self.generate_random_v_class(100)
        self.form_restricted_v_class(100)
        self.w = torch.rand(self.n_states, dtype=dtype)
        n_iterations = self.config.n_iterations
        lr_decay = self.config.lr_decay
        self.lr_w = self.config.lr_w; self.beta_w = self.config.beta; self.eps = self.config.eps
        optimizer_w = ModifiedAdam([self.w], lr = self.lr_w, betas = self.beta_w, eps = self.eps, amsgrad=False)
        self.w_collection = deque([])
        # self.td_penalty = td_penalty
        self.td_penalty = 1e-4
        running_average_w_estimator = 0
        for i in range(n_iterations):
            optimizer_w.zero_grad()
            if objective == 'bias_no_td_no_regW' or objective == 'bias_no_td_regW':
                L_v = self.bias_squared_vectorize_v(self.w, self.v_candidate)
            elif objective == 'bias_td_no_regW' or objective == 'bias_td_regW':
                L_v = self.bias_squared_vectorize_v(self.w, self.v_candidate) - self.td_penalty*self.td_error_vectorize(self.v_candidate)
            elif objective == 'bias_restricted_td_regW' or objective == 'bias_restricted_td_no_regW':
                L_v = self.bias_squared_vectorize_v(self.w, self.v_candidate_narrow)
            max_idx = torch.argmax(L_v)
            if objective == 'bias_restricted_td_regW' or objective == 'bias_restricted_td_no_regW':
                self.v = self.v_candidate_narrow[:, max_idx].clone().detach()
            else:
                self.v = self.v_candidate[:,max_idx].clone().detach()
            self.v.requires_grad = False
            self.w.requires_grad = True
            if objective == 'bias_no_td_no_regW' or objective == 'bias_td_no_regW' or objective == 'bias_restricted_td_no_regW':
                self.L_w = self.bias_squared(self.w, self.v)
            elif objective == 'bias_no_td_regW' or objective == 'bias_td_regW' or objective == 'bias_restricted_td_regW':
                if self.config.reg_w == 'l2':
                    self.L_w = self.bias_squared(self.w, self.v) + self.config.coeff_reg_w*torch.norm(self.w)**2#+1*((self.w*self.d_mu).sum()-1)**2
                elif self.config.reg_w == 'linfty':
                    self.L_w = self.bias_squared(self.w, self.v)
            self.L_w.backward()
            copy = self.w.clone().detach()
            if lr_decay:
                lr_w = self.lr_w / (math.sqrt(i+1))
                for param_group in optimizer_w.param_groups:
                    param_group['lr'] = lr_w
            optimizer_w.step()
            self.w.requires_grad = False
            if self.config.reg_w == 'linfty':
                torch.clamp_(self.w, min=0, max=self.config.coeff_reg_w)
            else:
                torch.clamp_(self.w, min=0, max=1e6)
            # torch.clamp_(self.w, min=0, max=4)
            w_grad_norm = torch.norm(self.w-copy,p=1)/(lr_w)
            w_estimator = self.w_estimator(self.w)
            w_estimator_sn = self.self_normalized_w_estimator(self.w)
            v_estimator = self.v_estimator(self.v)
            running_average_w_estimator = i / (i+1)*running_average_w_estimator + 1/(i+1)*w_estimator
            self.w_collection.append(w_estimator)
            if len(self.w_collection)> int(self.config.tail_fraction*i + 1):
                self.w_collection.popleft()

            if i %1000 ==0:
                print('\n')
                print('iteration ', i)
                print('max idx', max_idx)
                print('w projected grad norm: ', w_grad_norm)
                print('v estimator: ', v_estimator)
                print('self normalized w estimator: ', w_estimator_sn)
                print('w estimator: ', w_estimator)
                print('running average w estimator:', running_average_w_estimator)
                print('tail average w estimator', sum(self.w_collection)/len(self.w_collection))
                # pdb.set_trace()
        return sum(self.w_collection)/len(self.w_collection)

    def optimize_discrete(self,objective, td_penalty = None):
        # self.form_restricted_v_class(1)
        # self.w = torch.rand(self.n_states, dtype=dtype)
        # self.generate_random_w_class()
        n_iterations = self.config.n_iterations
        lr_decay = self.config.lr_decay
        self.w_collection = deque([])
        self.td_penalty = td_penalty
        running_average_w_estimator = 0
        # bias_matrix = torch.zeros(self.config.v_class_cardinality, self.config.v_class_cardinality, dtype=dtype)
        # bias_matrix_population = torch.zeros(self.config.v_class_cardinality, self.config.v_class_cardinality, dtype=dtype)
        # # bias_matrix_vectorize = 
        # for i in range(self.config.v_class_cardinality):
        #     for j in range(self.config.v_class_cardinality):
        #         bias_matrix[i,j] = self.bias_squared_incremental(self.w_candidate[:,i], self.v_candidate[:,j])
        #         bias_matrix_population[i,j] = self.bias_squared_population(self.w_candidate[:,i], self.v_candidate[:,j])
        # pdb.set_trace()
        self.w = self.w_candidate[:, np.random.randint(0, self.config.v_class_cardinality)]
        # pdb.set_trace()
        max_idx = torch.zeros(self.config.v_class_cardinality, dtype=torch.long)

        for i in range(self.config.v_class_cardinality):
            # optimizer_w.zero_grad()
            if objective == 'bias_no_td_no_regW' or objective == 'bias_no_td_regW':
                L_v = self.bias_squared_vectorize_v(self.w_candidate[:,i], self.v_candidate)
            elif objective == 'bias_td_no_regW' or objective == 'bias_td_regW':
                L_v = self.bias_squared_vectorize_v(self.w_candidate[:,i], self.v_candidate) - self.td_penalty*self.td_error_vectorize(self.v_candidate)
            elif objective == 'bias_restricted_td_regW' or objective == 'bias_restricted_td_no_regW':
                L_v = self.bias_squared_vectorize_v(self.w_candidate[:,i], self.v_candidate_narrow)
            max_idx[i] = torch.argmax(L_v)
            # if objective == 'bias_restricted_td_regW' or objective == 'bias_restricted_td_no_regW':
            #     self.v = self.v_candidate_narrow[:, max_idx].clone().detach()
            # else:
            #     self.v = self.v_candidate[:,max_idx].clone().detach()
        L_w = torch.zeros(self.config.v_class_cardinality)
        for i in range(self.config.v_class_cardinality):
            if objective == 'bias_no_td_no_regW' or objective == 'bias_td_no_regW' or objective == 'bias_restricted_td_no_regW':
                # self.L_w = self.bias_squared_incremental(self.w, self.v)
                L_w[i] = self.bias_squared(self.w_candidate[:,i], self.v_candidate[:,max_idx[i]])
            elif objective == 'bias_no_td_regW' or objective == 'bias_td_regW' or objective == 'bias_restricted_td_regW':
                if self.config.reg_w == 'l2':
                    # self.L_w = self.bias_squared_incremental(self.w, self.v) + self.config.coeff_reg_w*torch.norm(self.w)**2#+1*((self.w*self.d_mu).sum()-1)**2
                    L_w[i] = self.bias_squared(self.w_candidate[:,i], self.v_candidate[:,max_idx[i]]) + self.config.coeff_reg_w*torch.norm(self.w_candidate[:,i])**2#+1*((self.w*self.d_mu).sum()-1)**2
                elif self.config.reg_w == 'linfty':
                    L_w[i] = self.bias_squared(self.w_candidate[:,i], self.v_candidate[:,max_idx[i]])
            # self.L_w.backward()
        min_idx = torch.argmin(L_w)
        # copy = self.w.clone().detach()
        # if lr_decay:
        #     lr_w = self.lr_w / (math.sqrt(i+1))
        #     for param_group in optimizer_w.param_groups:
        #         param_group['lr'] = lr_w
        # optimizer_w.step()
        # self.w.requires_grad = False
        # if self.config.reg_w == 'linfty':
        #     torch.clamp_(self.w, min=0, max=self.config.coeff_reg_w)
        # else:
        #     torch.clamp_(self.w, min=0, max=1e6)
        # torch.clamp_(self.w, min=0, max=4)
        # w_grad_norm = torch.norm(self.w-copy,p=1)/(lr_w)
        self.w = self.w_candidate[:, min_idx].clone().detach()
        w_estimator = self.w_estimator(self.w)
        w_estimator_sn = self.self_normalized_w_estimator(self.w)
        self.v = self.v_candidate[:, max_idx[min_idx]]
        v_estimator = self.v_estimator(self.v)
        # running_average_w_estimator = i / (i+1)*running_average_w_estimator + 1/(i+1)*w_estimator
        # self.w_collection.append(w_estimator)
        # if len(self.w_collection)> int(self.config.tail_fraction*i + 1):
        #     self.w_collection.popleft()

        # if i %1000 ==0:
        print('\n')
        # print('iteration ', i)
        print('max idx', max_idx[min_idx])
        print('min idx', min_idx)
        # print('w projected grad norm: ', w_grad_norm)
        print('v estimator: ', v_estimator)
        print('self normalized w estimator: ', w_estimator_sn)
        print('w estimator: ', w_estimator)
        # print('running average w estimator:', running_average_w_estimator)
        # print('tail average w estimator', sum(self.w_collection)/len(self.w_collection))
        # pdb.set_trace()
        return w_estimator
        # return sum(self.w_collection)/len(self.w_collection)

    def optimize(self, objective, td_penalty):
        if objective == 'bias_cf':
            self.w = torch.matmul(torch.inverse(torch.mm(self.A_bias_cf.t(), self.A_bias_cf)), torch.matmul(self.A_bias_cf.t(), self.b_bias_cf))
            w_estimator = self.w_estimator(self.w) 
            return w_estimator
        elif objective == 'bias_opt_cf':
            w_estimator = self.optimize_closed_form()
            return w_estimator
        elif objective == 'bias_minmax':
            w_estimator = self.optimize_optimistic('bias', td_penalty=None)
            return w_estimator
        elif objective == 'bias_td_minmax':
            w_estimator = self.optimize_optimistic('bias_td', td_penalty)
            return w_estimator
        
    def optimize_closed_form(self):
        self.w = torch.rand(self.n_states, dtype=dtype)
        # self.w_cf = torch.matmul(torch.inverse(self.A_bias_cf), self.b_bias_cf)
        self.w_cf = torch.matmul(torch.inverse(torch.mm(self.A_bias_cf.t(), self.A_bias_cf)), torch.matmul(self.A_bias_cf.t(), self.b_bias_cf))
        w_estimator_cf = self.w_estimator(self.w_cf)
        self.lr_w = self.config.lr_w
        self.beta_w = self.config.beta
        self.eps = self.config.eps
        self.optimizer_w = OptimisticAdam([self.w], lr = self.lr_w, betas = self.beta_w, eps = self.eps, amsgrad=False)
        # self.optimizer_w = RMSprop([self.w], lr = self.lr_w)
        n_iterations = self.config.n_iterations
        self.w_collection = deque([])
        lr_decay = self.config.lr_decay
        

        beta1 = self.beta_w[0]; beta2 = self.beta_w[1]; eps = self.eps
        
        self.zero = torch.zeros(self.n_states, dtype=dtype)
        self.w_e = self.zero.clone().detach()
        self.w_e_hat = [self.zero.clone().detach(),self.zero.clone().detach()]
        self.w_s = self.zero.clone().detach()
        self.w_s_hat = [self.zero.clone().detach(),self.zero.clone().detach()]

        for i in range(n_iterations):
            self.w.requires_grad = True
            self.optimizer_w.zero_grad()
            f_w = self.f(self.w)
            L = torch.matmul(f_w.t(), f_w)
            L.backward()

            copy = self.w.clone().detach()
            if lr_decay:
                lr_w = self.lr_w / (math.sqrt(i+1))
                for param_group in self.optimizer_w.param_groups:
                    param_group['lr'] = lr_w
            else: lr_w = self.lr_w
            # self.optimizer_w.step()

            w_grad_term = self.optimizer_w.step()
            # w_grad = self.w.grad.clone().detach()
            # self.w_e = beta1 * self.w_e + (1-beta1)*w_grad
            # self.w_s = beta2 * self.w_s +(1-beta2)* (w_grad**2)
            # self.w_e_hat[0] = self.w_e_hat[1]
            # self.w_e_hat[1] = self.w_e / (1-beta1**(i+1))
            # self.w_s_hat[0] = self.w_s_hat[1]
            # self.w_s_hat[1] = self.w_s / (1-beta2**(i+1))
            # self.w.requires_grad = False
            # if lr_decay:
            #     lr_w = self.lr_w / (math.sqrt(i+1))
            # else:
            #     lr_w = self.lr_w
            # self.w = self.w - 2*lr_w*self.w_e_hat[1]/(self.w_s_hat[1].sqrt().add_(eps)) + lr_w*self.w_e_hat[0]/(self.w_s_hat[0].sqrt().add_(eps))

            self.w.requires_grad = False

            torch.clamp_(self.w, min=0, max=1e6)
            # w_grad_norm = torch.norm(self.w-copy,p=1)/(lr_w)

            w_estimator = self.w_estimator(self.w)
            w_estimator_sn = self.self_normalized_w_estimator(self.w)
            self.w_collection.append(w_estimator)
            if len(self.w_collection)> int(self.config.tail_fraction*i + 1):
                self.w_collection.popleft()

            if i%1000 == 0:
                print('\n')
                print('iteration ', i)
                # print('w grad norm', w_grad_norm)
                print('w estimator: ', w_estimator)
                print('w estimator cf:', w_estimator_cf)
                print('self normalized w estimator: ', w_estimator_sn)
                print('tail average: ', sum(self.w_collection)/len(self.w_collection))
                # if i %10000 == 0:
                #     pdb.set_trace()
        return sum(self.w_collection)/len(self.w_collection)

    def optimize_optimistic(self, objective, td_penalty=None):
        self.w_cf = torch.matmul(torch.inverse(torch.mm(self.A_bias_cf.t(), self.A_bias_cf)), torch.matmul(self.A_bias_cf.t(), self.b_bias_cf))
        w_estimator_cf = self.w_estimator(self.w_cf)
        # self.w = torch.rand(self.n_states, dtype=dtype)
        # self.v = torch.rand(self.n_states, dtype=dtype)
        # self.w = self.w_pi.clone().detach(); self.v = self.true.v_pi.clone().detach()
        self.w = torch.ones(self.n_states, dtype=dtype); self.v = torch.rand(self.n_states, dtype=dtype)
        self.lr_w = self.config.lr_w #0.1
        self.lr_v = -self.config.lr_w #-0.1
        self.beta = self.config.beta
        self.eps = 1e-8
        self.optimizer_w = OptimisticAdam([self.w], lr = self.lr_w, betas = self.beta, eps = self.eps, amsgrad=False)
        self.optimizer_v = OptimisticAdam([self.v], lr = self.lr_v, betas = self.beta, eps = self.eps, amsgrad=False)
        n_iterations = self.config.n_iterations
        self.w_collection = deque([])
        for i in range(n_iterations):
            self.optimizer_w.zero_grad()
            self.optimizer_v.zero_grad()
            self.w.requires_grad = True
            self.v.requires_grad = True
            if objective == 'bias':
                L = self.bias_squared(self.w, self.v)#-0.1*self.td_error(self.v)
            elif objective == 'bias_td':
                L = self.bias_squared(self.w, self.v)-td_penalty*self.td_error(self.v)
            L.backward()
            copy = self.v.clone().detach()
            # lr_v = self.lr_v /(math.sqrt(i+1)**1.1)
            lr_v = self.lr_v #/(math.sqrt(i+1))
            for param_group in self.optimizer_v.param_groups:
                param_group['lr'] = lr_v
            v_grad_term = self.optimizer_v.step()
            self.v.requires_grad = False
            v_grad_norm = torch.norm(self.v-copy,p=1)/(-lr_v)

            copy = self.w.clone().detach()
            lr_w = self.lr_w / (math.sqrt(i+1))
            # lr_w = self.lr_w*(i+1)**(-0.51)
            for param_group in self.optimizer_w.param_groups:
                param_group['lr'] = lr_w
            w_grad_term = self.optimizer_w.step()
            self.w.requires_grad = False
            torch.clamp_(self.w, min=0, max=1e6)
            w_grad_norm = torch.norm(self.w-copy,p=1)/(lr_w)

            w_estimator = self.w_estimator(self.w)
            w_estimator_sn = self.self_normalized_w_estimator(self.w)
            v_estimator = self.v_estimator(self.v)
            self.w_collection.append(w_estimator)
            if len(self.w_collection)> int(self.config.tail_fraction*i + 1):
                self.w_collection.popleft()

            if i%1000 ==0:
                print('\n')
                print('iteration ', i)
                print('w grad norm', w_grad_norm)
                print('v grad norm', v_grad_norm)
                print('v estimator: ', v_estimator)
                print('w estimator: ', w_estimator)
                print('w estimator cf:', w_estimator_cf)
                print('self normalized w estimator: ', w_estimator_sn)
                print('tail average: ', sum(self.w_collection)/len(self.w_collection))
                # if i %10000 == 0:
                #     pdb.set_trace()
        return sum(self.w_collection)/len(self.w_collection)


    def optimize_optimistic_adam(self, objective, td_penalty = None):
        # self.initialize_w_v(objective, input_penalty)
        self.w_cf = torch.matmul(torch.inverse(torch.mm(self.A_bias_cf.t(), self.A_bias_cf)), torch.matmul(self.A_bias_cf.t(), self.b_bias_cf))
        w_estimator_cf = self.w_estimator(self.w_cf)
        self.w = torch.rand(self.n_states, dtype=dtype)
        self.v = torch.rand(self.n_states, dtype=dtype)
        # self.v = self.v_td[:,0].clone().detach().type(dtype)
        if td_penalty is not None:
            self.penalty = td_penalty
        n_iterations = 100001
        lr_decay = True; lr_decay_v =True
        self.lr_w = 0.1 #0.1
        self.lr_v = 0.1
        self.beta_w = (0, 0.999)
        self.beta_v = (0, 0.999)
        self.eps = 1e-8
        self.optimizer_w = OptimisticAdam([self.w], lr = self.lr_w, betas = self.beta_w, eps = self.eps, amsgrad=False)
        self.optimizer_v = OptimisticAdam([self.v], lr = self.lr_v, betas = self.beta_v, eps = self.eps, amsgrad=False)
        Rmax = self.R.max()
        Rmin = self.R.min()
        for i in range(n_iterations):
            self.optimizer_w.zero_grad()
            self.optimizer_v.zero_grad()
            self.w.requires_grad = False
            self.v.requires_grad = True
            # L = -self.bias_squared_incremental(self.w, self.v) + self.penalty * self.td_error(self.v)
            # L = -self.bias_squared_incremental(self.w, self.v) + self.penalty * self.bellman_error(self.v)
            L = -self.bias_squared(self.w, self.v)+self.td_error(self.v)
            L.backward()
            copy = self.v.clone().detach()
            if lr_decay_v:
                lr_v = self.lr_v /(math.sqrt(i+1))
                for param_group in self.optimizer_v.param_groups:
                    param_group['lr'] = lr_v
            v_grad_term = self.optimizer_v.step()
            self.v.requires_grad = False
            # torch.clamp_(self.v, min = 0, max = (self.horizon_normalization-1)/self.config.normalizing_factor)
            # torch.min(self.v, Rmax*self.max_rewards_to_go, out=self.v)
            # torch.max(self.v, Rmin*self.max_rewards_to_go, out=self.v)
            v_grad_norm = torch.norm(self.v-copy,p=1)/(lr_v)

            self.w.requires_grad = True
            L = self.bias_squared(self.w, self.v)
            L.backward()
            copy = self.w.clone().detach()
            if lr_decay:
                lr_w = self.lr_w / (math.sqrt(i+1))
                for param_group in self.optimizer_w.param_groups:
                    param_group['lr'] = lr_w
            w_grad_term = self.optimizer_w.step()
            self.w.requires_grad = False
            torch.clamp_(self.w, min=0, max=1e6)
            w_grad_norm = torch.norm(self.w-copy,p=1)/(lr_w)

            w_estimator = self.w_estimator(self.w)
            w_estimator_sn = self.self_normalized_w_estimator(self.w)
            v_estimator = self.v_estimator(self.v)
            print('\n')
            print('iteration ', i)
            print('w grad norm', w_grad_norm)
            print('v grad norm', v_grad_norm)
            print('v estimator: ', v_estimator)
            print('w estimator: ', w_estimator)
            print('w estimator cf:', w_estimator_cf)
            print('self normalized w estimator: ', w_estimator_sn)
            if i %10000 == 0:
                pdb.set_trace()
        
        return w_estimator, w_estimator_sn

    def optimize_manual(self, objective, td_penalty=None):
        self.w = torch.ones(self.n_states, dtype=dtype)
        self.v = torch.rand(self.n_states, dtype=dtype)
        n_iterations = self.config.n_iterations
        lr_decay = self.config.lr_decay
        self.lr_w = self.config.lr_w
        self.beta_w = self.config.beta
        self.eps = self.config.eps
        beta1 = self.beta_w[0]; beta2 = self.beta_w[1]; eps = self.eps

        self.w_e = self.zero.clone().detach()
        self.w_e_hat = [self.zero.clone().detach(),self.zero.clone().detach()]
        self.w_s = self.zero.clone().detach()
        self.w_s_hat = [self.zero.clone().detach(),self.zero.clone().detach()]
        if objective == 'bias_td_delay':
            self.v_target = self.v.clone().detach()
        for i in range(n_iterations):
            x,f,d = self.maximize_v()
            if d['warnflag'] != 0:
                print('did not converge')
            self.v = torch.tensor(x, dtype = dtype, requires_grad=False)
            self.w.requires_grad = True
            if self.w.grad is not None:
                self.w.grad.zero_()
            L = self.bias_squared(self.w, self.v)+1*((self.w*self.d_mu).sum()-1)**2
            L.backward()
            w_grad = self.w.grad.clone().detach()
            self.w_e = beta1 * self.w_e + (1-beta1)*w_grad
            self.w_s = beta2 * self.w_s +(1-beta2)* (w_grad**2)
            self.w_e_hat[0] = self.w_e_hat[1]
            self.w_e_hat[1] = self.w_e / (1-beta1**(i+1))
            self.w_s_hat[0] = self.w_s_hat[1]
            self.w_s_hat[1] = self.w_s / (1-beta2**(i+1))
            self.w.requires_grad = False
            if lr_decay:
                lr_w = self.lr_w / (math.sqrt(i+1))
            else:
                lr_w = self.lr_w
            self.w = self.w - 2*lr_w*self.w_e_hat[1]/(self.w_s_hat[1].sqrt().add_(eps)) + lr_w*self.w_e_hat[0]/(self.w_s_hat[0].sqrt().add_(eps))
            torch.clamp_(self.w, min=0, max=1e6)

            w_estimator = self.w_estimator(self.w)
            w_estimator_sn = self.self_normalized_w_estimator(self.w)
            v_estimator = self.v_estimator(self.v)
            if objective == 'bias_td_delay':
                self.v_target = self.v.clone().detach()
            if i%1 == 0:
                print('\n')
                print('iteration ', i)
                # print('projected grad norm', w_grad_norm)
                print('w grad norm', torch.norm(w_grad))
                # print('v grad norm', v_grad_norm)
                print('v estimator: ', v_estimator)
                print('w estimator: ', w_estimator)
                print('self normalized w estimator: ', w_estimator_sn)
            if i %1000 == 0:
                pdb.set_trace()
        return w_estimator, w_estimator_sn
    def maximize_v(self):
        max_v = optimize.minimize(self.objective_v_scipy, self.v.data.numpy().astype(np.float64), method="L-BFGS-B", jac = self.jac_v_scipy, bounds = self.v_bounds)
        # max_v = optimize.fmin_l_bfgs_b(self.objective_v_scipy, x0 = self.v.data.numpy().astype(np.float64), fprime = self.jac_v_scipy, bounds = self.v_bounds, pgtol = 1e-8,maxiter = 20)
        # max_v = optimize.fmin_l_bfgs_b(self.objective_v_scipy, x0 = self.v.data.numpy().astype(np.float64), fprime = self.jac_v_scipy, bounds = self.v_bounds, pgtol = 1e-8,maxiter = 100)
        # max_v = optimize.minimize(self.objective_v_scipy, x0=self.v.data.numpy().astype(np.float64), method="L-BFGS-B", jac = self.jac_v_scipy, bounds = self.v_bounds, options ={'maxfun':1500, 'maxiter':1500})
        # max_v = optimize.minimize(self.objective_v_scipy, self.v.data.numpy().astype(np.float64), method="L-BFGS-B", bounds = self.v_bounds)
        # max_v = optimize.minimize(self.objective_v_scipy, self.v.data.numpy().astype(np.float64), method="SLSQP", constraints={"fun": self.l2_norm_constraint, "type": "ineq"}, bounds = self.v_bounds)
        # max_v = optimize.minimize(self.objective_v_scipy, self.v.data.numpy().astype(np.float64), method="L-BFGS-B", constraints={"fun": self.l2_norm_constraint, "type": "ineq"}, bounds = self.v_bounds)
        return max_v
    def objective_v_scipy(self,v):
        v = torch.tensor(v, dtype = dtype)
        # pdb.set_trace()
        # return (-self.bias_squared_d(self.d.detach(), v)+ self.penalty*self.td_error(v) + self.reg_v*torch.norm(v)**2 ).data.numpy().astype(np.float64) 
        # return (-self.bias_squared_d(self.d.detach(), v)).data.numpy().astype(np.float64) 
        # return (-self.bias_squared_incremental(self.w, v) + 0.0001*torch.norm(v)**2).data.numpy().astype(np.float64) 
        if self.objective == 'bias_td':
            return (-self.bias_squared_incremental(self.w, v) + self.penalty * self.td_error(v)).data.numpy().astype(np.float64) 
        elif self.objective == 'bias':
            return (-self.bias_squared_incremental(self.w, v)).data.numpy().astype(np.float64) 
        elif self.objective == 'true_bellman':
            return (-self.bias_squared_incremental(self.w, v) + self.penalty * self.bellman_error(v)).data.numpy().astype(np.float64) 
        elif self.objective == 'bias_td_delay':
            return (-self.bias_squared_incremental(self.w, v) + self.penalty * self.td_error_delay(v)).data.numpy().astype(np.float64) 
        elif self.objective == 'bias_l2':
            return (-self.bias_squared_incremental(self.w, v) + 0.01 * torch.var(v)).data.numpy().astype(np.float64) 

        # return (-self.bias_squared_incremental(self.w, v)).data.numpy().astype(np.float64) 
        # return (-self.bias_squared(self.w, v)+ self.penalty*self.td_error(v) + self.reg_v*torch.norm(v)**2 ).data.numpy().astype(np.float64) 


    def jac_v_scipy(self,v):
        # self.w.requires_grad = False
        # self.d.requires_grad = False
        v_pytorch = torch.tensor(v, dtype = torch.float64, requires_grad=True)
        # v_pytorch = v.clone().detach(); v_pytorch.requires_grad = True
        self.w.requires_grad = False
        # L = -self.bias_squared(self.w, v_pytorch)+ self.penalty*self.td_error(v_pytorch)+self.reg_v*torch.norm(v_pytorch)**2
        # L = -self.bias_squared_d(self.d.detach(), v_pytorch)+ self.penalty*self.td_error(v_pytorch)+self.reg_v*torch.norm(v_pytorch)**2
        # L = -self.bias_squared_d(self.d.detach(), v_pytorch)
        # L = -self.bias_squared_incremental(self.w, v_pytorch)#+0.0001*torch.norm(v_pytorch)**2
        if self.objective == 'bias_td':
            L = -self.bias_squared_incremental(self.w, v_pytorch)+self.penalty * self.td_error(v_pytorch)
        elif self.objective == 'bias':
            L = -self.bias_squared_incremental(self.w, v_pytorch)
        elif self.objective == 'true_bellman':
            L = -self.bias_squared_incremental(self.w, v_pytorch) + self.penalty* self.bellman_error(v_pytorch)
        elif self.objective == 'bias_td_delay':
            L = -self.bias_squared_incremental(self.w, v_pytorch)+self.penalty * self.td_error_delay(v_pytorch)
        elif self.objective == 'bias_l2':
            L = -self.bias_squared_incremental(self.v, v_pytorch) + 0.01*torch.var(v_pytorch)
        L.backward()
        return v_pytorch.grad.data.numpy().astype(np.float64)


    def solve_closed_form_bias(self):
        self.A_bias_cf = torch.mm(self.D_mu, self.gamma*self.P_rho-self.I).t()
        self.b_bias_cf = -self.d0/self.horizon_normalization+1/self.horizon_normalization * self.gamma**self.horizon*self.d_pi_H
        # weighted_vector = torch.matmul(w, torch.mm(self.D_mu, self.gamma*self.P_rho-self.I)) + self.d0/self.horizon_normalization
        # weighted_vector += -1/self.horizon_normalization * self.gamma**self.horizon*self.d_pi_H
        #* Now we should have the min norm problem min \norm{A\omega -b}
        #* can impose additional equality constraint and still get closed form solution
        
        #* we will get closed form solution to the following quadratic program
        #* min 1/2 x'Qx - c'x s.t Ex = d
        #* the closed form will use matrix inversion
        eps = EPS_CLOSED_FORM_W
        # W = torch.diag(1/self.d_mu); W[W!=W] = 1; W[torch.isinf(W)]=1
        #* we will first attempt a closed form solution to this quadratic problem with linear equality constraint
        #* if we encounter negative component of the solution, then we will invoke scipy solver for positivity constraint
        W = self.I.clone().detach()
        Q = torch.matmul(self.A_bias_cf.t(), torch.matmul(W,self.A_bias_cf))+eps*self.I
        c = torch.matmul(self.A_bias_cf.t(),torch.matmul(W,self.b_bias_cf))

        #* append matrix to reflect normalization constraint
        X = torch.zeros(self.n_states+1, self.n_states+1, dtype=dtype)
        X[:self.n_states, :self.n_states] = Q
        X[-1,:-1] = self.d_mu.clone().detach()
        X[:-1,-1] = self.d_mu.clone().detach()
        y = torch.zeros(self.n_states+1, dtype=dtype)
        y[:-1] = c
        y[-1] = 1.0

        w = torch.matmul(torch.inverse(X),y)[:-1]
        if (w<0).any():
            print('Encounter negative element in closed form, relaxed problem, will use scipy solver instead')
            #* alternatively, with positivity constraint on w, we can use some optimization solver
            d_mu = self.d_mu.data.numpy().astype(np.float64)
            Q = Q.data.numpy().astype(np.float64)
            c = c.data.numpy().astype(np.float64)
            def objective_scipy(w):
                return np.dot(w.T, np.dot(Q,w))-2*np.dot(c,w)
            def jac_scipy(w):
                return np.dot(Q,w) - c
            def constraint_scipy(w):
                return np.dot(w, d_mu)-1
            
            cons = {'type':'eq', 'fun': constraint_scipy}
            min_w = optimize.minimize(objective_scipy, np.random.rand(self.n_states), jac= jac_scipy, bounds = self.w_bounds, constraints=cons, tol=1e-9, options={'ftol':1e-10})
            self.w_bias_cf = torch.tensor(min_w.x, dtype=dtype)
        else:
            self.w_bias_cf = torch.tensor(w, dtype=dtype)
    def solve_closed_form_bias_old(self):
        self.A_bias_cf = torch.zeros(self.n_states, self.n_states, dtype=dtype)
        
        A = torch.zeros(self.n_states, self.n_states, dtype=dtype)
        initial = torch.zeros(self.n_states, dtype=dtype)
        for episode in range(self.n_trajectories):
            discounted_t = 1
            s0 = self.s[episode][0]
            initial += self.phi(s0)
            for (s,a,sn,r) in zip(self.s[episode], self.a[episode], self.sn[episode], self.r[episode]):
                self.A_bias_cf += discounted_t*torch.ger(self.gamma*self.rho[s,a]*self.phi(sn)-self.phi(s), self.phi(s))
                discounted_t *= self.gamma
            self.A_bias_cf -= self.gamma**self.horizon*torch.ger(self.phi(sn), self.phi(sn))
        self.b_bias_cf = -initial
        #* Now we should have the min norm problem min \norm{A\omega -b}
        #* can impose additional equality constraint and still get closed form solution
        
        #* we will get closed form solution to the following quadratic program
        #* min 1/2 x'Qx - c'x s.t Ex = d
        #* the closed form will use matrix inversion
        eps = EPS_CLOSED_FORM_W
        # W = torch.diag(1/self.d_mu); W[W!=W] = 1; W[torch.isinf(W)]=1
        #* we will first attempt a closed form solution to this quadratic problem with linear equality constraint
        #* if we encounter negative component of the solution, then we will invoke scipy solver for positivity constraint
        W = self.I.clone().detach()
        Q = torch.matmul(self.A_bias_cf.t(), torch.matmul(W,self.A_bias_cf))+eps*self.I
        c = torch.matmul(self.A_bias_cf.t(),torch.matmul(W,self.b_bias_cf))

        #* append matrix to reflect normalization constraint
        X = torch.zeros(self.n_states+1, self.n_states+1, dtype=dtype)
        X[:self.n_states, :self.n_states] = Q
        X[-1,:-1] = self.d_mu.clone().detach()
        X[:-1,-1] = self.d_mu.clone().detach()
        y = torch.zeros(self.n_states+1, dtype=dtype)
        y[:-1] = c
        y[-1] = 1.0

        w = torch.matmul(torch.inverse(X),y)[:-1]
        if (w<0).any():
            print('Encounter negative element in closed form, relaxed problem, will use scipy solver instead')
            #* alternatively, with positivity constraint on w, we can use some optimization solver
            d_mu = self.d_mu.data.numpy().astype(np.float64)
            Q = Q.data.numpy().astype(np.float64)
            c = c.data.numpy().astype(np.float64)
            def objective_scipy(w):
                return np.dot(w.T, np.dot(Q,w))-2*np.dot(c,w)
            def jac_scipy(w):
                return np.dot(Q,w) - c
            def constraint_scipy(w):
                return np.dot(w, d_mu)-1
            
            cons = {'type':'eq', 'fun': constraint_scipy}
            min_w = optimize.minimize(objective_scipy, np.random.rand(self.n_states), jac= jac_scipy, bounds = self.w_bounds, constraints=cons, tol=1e-9, options={'ftol':1e-10})
            self.w_bias_cf = torch.tensor(min_w.x, dtype=dtype)
        else:
            self.w_bias_cf = torch.tensor(w, dtype=dtype)