import numpy as np
import torch
import pdb
import math
import torch.nn.functional as F
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from torch.optim import  LBFGS, RMSprop
from optimizer import SGD,Adam, ExtraAdam, ModifiedAdam, ModifiedExtraAdam, ACGD, OptimisticAdam
from manual_optimizer import AMSGrad, OptimisticAMSGrad
import quadprog
from scipy import optimize

dtype = torch.double
USE_KNOWN_D0 = True
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
        pdb.set_trace()
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

    def bias_squared_incremental(self,w,v):
        w_comp = (self.discount*w[self.s]*(self.gamma*self.rho[self.s, self.a]*v[self.sn] - v[self.s])).sum()
        if USE_KNOWN_D0:
            v_comp = torch.matmul(self.true.d0, v) * self.n_trajectories
        else:
            v_comp = v[self.s[:,0]].sum()
            # v_comp = torch.matmul(self.d0, v) * self.n_trajectories
        extra_term = -1/self.horizon_normalization * self.gamma**self.horizon*torch.matmul(self.true.d_pi_H,v)
        # extra_term = 0
        bias = (w_comp + v_comp)  / self.n_trajectories / self.horizon_normalization + extra_term
        return bias**2

    def bias_squared_vectorize(self, w, v_stack):
        w_comp = (self.discount[:,None]*w[self.s,None]*(self.gamma*self.rho[self.s, self.a,None]*v_stack[self.sn] - v_stack[self.s])).sum(dim=(0,1))
        if USE_KNOWN_D0:
            v_comp = torch.matmul(self.true.d0, v_stack)* self.n_trajectories
        else:
            v_comp = v_stack[self.s[:,0]].sum(dim=0)
        # pdb.set_trace()
        # extra_term = 0
        extra_term = -1/self.horizon_normalization * self.gamma**self.horizon*torch.matmul(self.true.d_pi_H,v_stack)
        
        bias = (w_comp + v_comp)  / self.n_trajectories / self.horizon_normalization+ extra_term
        return bias**2

    def mb_bellman_error(self, v):
        Rpi = (self.R*self.pi).sum(dim=1)
        bellman_error = (self.discount*(self.gamma*torch.matmul(self.P_pi[self.s], v) + Rpi[self.s] - v[self.s])**2).sum()/self.n_trajectories/self.horizon_normalization
        return bellman_error

    def td_error(self, v):
        td_error = (self.discount*self.rho[self.s, self.a]*(self.gamma*v[self.sn] +self.R[self.s, self.a]- v[self.s])**2).sum()/self.n_trajectories/self.horizon_normalization
        return td_error

    def td_error_vectorize(self, v_stack):
        td_error = (self.discount[:,None]*self.rho[self.s, self.a,None]*(self.gamma*v_stack[self.sn] +self.R[self.s, self.a,None]- v_stack[self.s])**2).sum(dim=(0,1))/self.n_trajectories/self.horizon_normalization
        return td_error

    def generate_random_v_class(self, cardinality):
        self.v_candidate = torch.zeros(self.n_states, cardinality, dtype = dtype)
        self.v_candidate[:,0] = self.true.v_pi.clone().detach()
        for i in range(1,cardinality):
            self.v_candidate[:,i] = (2*torch.rand(self.n_states,dtype=dtype)-1)+self.true.v_pi
        return self.v_candidate
    def narrow_v_class(self, cardinality):
        td_error_vector = self.td_error_vectorize(self.v_candidate)
        topk = torch.topk(td_error_vector,k=cardinality,largest=False)[1]
        self.v_candidate_narrow = self.v_candidate[:, topk].clone().detach()
        # pdb.set_trace()
    # def optimize_g
    def optimize_finite_class(self, objective, td_penalty = None):
        # self.generate_random_v_class(100)
        self.narrow_v_class(10)
        self.w = torch.rand(self.n_states, dtype=dtype)
        n_iterations = self.config.n_iterations
        lr_decay = self.config.lr_decay
        self.lr_w = self.config.lr_w; self.beta_w = self.config.beta; self.eps = self.config.eps
        optimizer_w = ModifiedAdam([self.w], lr = self.lr_w, betas = self.beta_w, eps = self.eps, amsgrad=False)
        self.w_collection = deque([])
        self.td_penalty = td_penalty
        running_average_w_estimator = 0
        for i in range(n_iterations):
            optimizer_w.zero_grad()
            if objective == 'bias_no_td_no_regW' or objective == 'bias_no_td_regW':
                L_v = self.bias_squared_vectorize(self.w, self.v_candidate)
            elif objective == 'bias_td_no_regW' or objective == 'bias_td_regW':
                L_v = self.bias_squared_vectorize(self.w, self.v_candidate) - self.td_penalty*self.td_error_vectorize(self.v_candidate)
            elif objective == 'bias_restricted_td_regW' or objective == 'bias_restricted_td_no_regW':
                L_v = self.bias_squared_vectorize(self.w, self.v_candidate_narrow)
            max_idx = torch.argmax(L_v)
            if objective == 'bias_restricted_td_regW' or objective == 'bias_restricted_td_no_regW':
                self.v = self.v_candidate_narrow[:, max_idx].clone().detach()
            else:
                self.v = self.v_candidate[:,max_idx].clone().detach()
            self.v.requires_grad = False
            self.w.requires_grad = True
            if objective == 'bias_no_td_no_regW' or objective == 'bias_td_no_regW' or objective == 'bias_restricted_td_no_regW':
                self.L_w = self.bias_squared_incremental(self.w, self.v)
            elif objective == 'bias_no_td_regW' or objective == 'bias_td_regW' or objective == 'bias_restricted_td_regW':
                if self.config.reg_w == 'l2':
                    self.L_w = self.bias_squared_incremental(self.w, self.v) + 1e-4*torch.norm(self.w)**2#+1*((self.w*self.d_mu).sum()-1)**2
                elif self.config.reg_w == 'linfty':
                    self.L_w = self.bias_squared_incremental(self.w, self.v)
            self.L_w.backward()
            copy = self.w.clone().detach()
            if lr_decay:
                lr_w = self.lr_w / (math.sqrt(i+1))
                for param_group in optimizer_w.param_groups:
                    param_group['lr'] = lr_w
            optimizer_w.step()
            self.w.requires_grad = False
            if self.config.reg_w == 'linfty':
                torch.clamp_(self.w, min=0, max=2)
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



    def solve_closed_form_bias(self):
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