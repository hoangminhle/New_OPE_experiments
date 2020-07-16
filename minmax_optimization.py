import numpy as np
import torch
import pdb
import math
from collections import deque
from scipy import optimize
from optimizer import ModifiedAdam, OptimisticAdam, ACGD
import argparse
from torch import autograd
from torch.utils.tensorboard import SummaryWriter

dtype = torch.double
logging = True
# def objective_function(w,v):
#     return torch.matmul(w,v)

# def objective_v_scipy(v):
#     return -objective_function(w,v).data.numpy().astype(np.float64) 

# def maximize_v():
#     max_v = optimize.minimize(objective_v_scipy, v.data.numpy().astype(np.float64), method="L-BFGS-B", bounds = ((-1,1),))
#     return max_v

class MinMax_Optimizer(object):
    def __init__(self, w, v, optimization_params):
        self.w = w.clone().detach()
        self.v = v.clone().detach()
        self.param = optimization_params
        # unpack some optimization parameters
        self.n_iterations = self.param.n_iterations
        self.zero = torch.zeros_like(self.w, dtype  =dtype)
        self.I = torch.eye(len(self.w), dtype=dtype)
        self.optimization_method = self.param.method
        self.objective = self.param.objective
        if self.objective == 2 or self.objective == 3 or self.objective == 4:
            assert len(self.w) == 1
            assert len(self.v) == 1
        if logging:
            self.writer = SummaryWriter('toy_opt/ogda')
        if self.optimization_method == 'no_regret':
            self.lr_w = self.param.lr_w
            self.beta_w = (0,0.9)
            self.eps = 1e-4
            self.optimizer_w = ModifiedAdam([self.w], lr = self.lr_w, betas = self.beta_w, eps = self.eps, amsgrad=True)
            self.w_avg = self.w.clone().detach()
            self.v_avg = self.v.clone().detach()
            self.tail_fraction = self.param.tail_fraction
            self.w_tail = deque([])
            self.v_tail = deque([])
        elif self.optimization_method == 'ogda':
            self.lr_w = self.param.lr_w
            self.lr_v = self.param.lr_v
            self.beta_w = (0,0.9)
            self.beta_v = (0,0.9)
            self.eps = 1e-8
            self.optimizer_w = OptimisticAdam([self.w], lr = self.lr_w, betas = self.beta_w, eps = self.eps, amsgrad=False)
            self.optimizer_v = OptimisticAdam([self.v], lr = self.lr_v, betas = self.beta_v, eps = self.eps, amsgrad=False)
        elif self.optimization_method == 'gda':
            self.lr_w = self.param.lr_w
            self.lr_v = self.param.lr_v
            self.beta_w = (0,0.9)
            self.beta_v = (0,0.9)
            self.eps = 1e-8
            self.optimizer_w = ModifiedAdam([self.w], lr = self.lr_w, betas = self.beta_w, eps = self.eps, amsgrad=True)
            self.optimizer_v = ModifiedAdam([self.v], lr = self.lr_v, betas = self.beta_v, eps = self.eps, amsgrad=True)
        elif self.optimization_method == 'cgd':
            self.lr_w = self.param.lr_w
            self.lr_v = self.param.lr_v
        elif self.optimization_method == 'acgd':
            self.lr_w = self.param.lr_w
            self.lr_v = self.param.lr_v
            self.optimizer = ACGD(max_params = self.v, min_params = self.w, lr_max = self.lr_v, lr_min = self.lr_w)



    #* implements the objective of min-max optimization
    def objective_function(self):
        if self.objective == 1:
            return torch.matmul(self.w, self.v)
        elif self.objective == 2:
            return -3*self.w**2 - self.v**2 +4*self.w*self.v
        elif self.objective == 3:
            return 3*self.w**2 + self.v**2 + 4*self.w*self.v
        elif self.objective == 4:
            return (4*self.w**2 -(self.v-3*self.w+0.05*self.w**3)**2 - 0.1*self.v**4)*torch.exp(-0.01*(self.w**2 + self.v**2))
        elif self.objective == 5:
            pass

    #* converts the objective into numpy in case using scipy optimizer (only needed for full optimization of v)
    def objective_function_v(self, v):
        # pdb.set_trace()
        v=  torch.tensor(v, dtype=dtype)
        if self.objective == 1:
            return -torch.matmul(self.w, torch.tensor(v, dtype=dtype)).data.numpy().astype(np.float64)
        elif self.objective == 2:
            return -(-3*self.w**2 - v**2 +4*self.w*v).data.numpy().astype(np.float64)
        elif self.objective == 3:
            return -(3*self.w**2 + v**2 + 4*self.w*v).data.numpy().astype(np.float64)
        elif self.objective == 4:
            return -(4*self.w**2 -(v-3*self.w+0.05*self.w**3)**2 - 0.1*v**4)*torch.exp(-0.01*(self.w**2 + v**2)).data.numpy().astype(np.float64)
        elif self.objective == 5:
            pass

    def optimize(self):
        if self.optimization_method == 'no_regret':
            self.optimize_no_regret()
        elif self.optimization_method == 'ogda':
            self.optimize_ogda()
        elif self.optimization_method == 'gda':
            self.optimize_gda()
        elif self.optimization_method == 'cgd':
            self.optimize_cgd()
        elif self.optimization_method == 'acgd':
            self.optimize_acgd()
        elif self.optimization_method == 'conopt':
            pass
        elif self.optimization_method == 'fr':
            pass

    def maximize_v(self):
        # max_v = optimize.minimize(self.objective_function_v, self.v.data.numpy().astype(np.float64), method="L-BFGS-B", bounds = self.v_bounds)
        max_v = optimize.minimize(self.objective_function_v, self.v, method="L-BFGS-B", bounds = self.v_bounds)
        return max_v

    def optimize_acgd(self):
        for i in range(self.n_iterations):
            self.w.requires_grad = True
            self.v.requires_grad = True
            loss = self.objective_function()
            self.optimizer.zero_grad()
            self.optimizer.step(loss=loss)
            self.w.requires_grad = False
            self.v.requires_grad = False
            torch.clamp_(self.w, min=-1, max=1)
            torch.clamp_(self.v, min=-1, max=1)
            print('\n')
            print('Iteration :', i)
            print('current w: ', self.w)
            print('current v: ', self.v)
            if i %1000 == 0:
                pdb.set_trace()

    def optimize_acgd_old(self):
        sq_avg_w = torch.zeros_like(self.w)
        sq_avg_v = torch.zeros_like(self.v)
        beta = 0.99
        eps = 1e-5

        for i in range(self.n_iterations):
            if self.w.grad is not None:
                self.w.grad.zero_()
            if self.v.grad is not None:
                self.v.grad.zero_()
            self.w.requires_grad = True
            self.v.requires_grad = True
            L_w = self.objective_function()
            L_v = -self.objective_function()
            grad_w = autograd.grad(L_w, self.w, create_graph=True, retain_graph=True)
            grad_w_vec = torch.cat([g.contiguous().view(-1) for g in grad_w])
            grad_v = autograd.grad(L_v, self.v, create_graph=True, retain_graph=True)
            grad_v_vec = torch.cat([g.contiguous().view(-1) for g in grad_v])
            grad_w_vec_d = grad_w_vec.clone().detach()
            grad_v_vec_d = grad_v_vec.clone().detach()

            sq_avg_w.mul_(beta).addcmul_(1 - beta, grad_w_vec_d, grad_w_vec_d)
            sq_avg_v.mul_(beta).addcmul_(1 - beta, grad_v_vec_d, grad_v_vec_d)
            bias_correction = 1 - beta ** (i+1)

            lr_w = math.sqrt(bias_correction) * self.lr_w / sq_avg_w.sqrt().add(eps)
            lr_v = math.sqrt(bias_correction) * self.lr_v / sq_avg_v.sqrt().add(eps)
            
            scaled_grad_w = torch.mul(lr_w, grad_w_vec_d)
            scaled_grad_v = torch.mul(lr_v, grad_v_vec_d)

            hvp_w_vec = Hvp_vec(grad_v_vec, self.w, scaled_grad_v,
                                retain_graph=True)  # h_xy * d_y
            hvp_v_vec = Hvp_vec(grad_w_vec, self.v, scaled_grad_w,
                                retain_graph=True)  # h_yx * d_x
            p_w = torch.add(grad_w_vec_d, - hvp_w_vec)
            p_v = torch.add(grad_v_vec_d, - hvp_v_vec)
            
            pdb.set_trace()

    def optimize_cgd(self):
        # self.w = self.w / 1000
        # self.v = self.v / 1000
        for i in range(self.n_iterations):
            if self.w.grad is not None:
                self.w.grad.zero_()
            if self.v.grad is not None:
                self.v.grad.zero_()
            
            self.w.requires_grad = True
            self.v.requires_grad = False
            L_w = self.objective_function()
            L_w.backward()
            w_grad = self.w.grad.clone().detach()

            self.w.requires_grad = False
            self.v.requires_grad = True
            L_v = -self.objective_function()
            L_v.backward()
            v_grad = self.v.grad.clone().detach()

            #* update
            self.v.requires_grad = False
            self.w = self.w - self.lr_w / (1+self.lr_w**2*16) * (w_grad - self.lr_w*4*v_grad)
            self.v = self.v - self.lr_v / (1+self.lr_v**2*16) * (v_grad + self.lr_v*4*w_grad)
            
            torch.clamp_(self.w, min=-1, max=1)
            torch.clamp_(self.v, min=-1, max=1)
            print('\n')
            print('Iteration :', i)
            print('current w: ', self.w)
            print('current v: ', self.v)
            if i %1000 == 0:
                pdb.set_trace()


    def optimize_no_regret(self):
        self.v_bounds = tuple( [ ( self.param.v_bound[0], self.param.v_bound[1]) for _ in range(len(self.v)) ] )
        for i in range(1, self.n_iterations):
            max_v = self.maximize_v()
            self.v = torch.tensor(max_v.x, dtype = dtype)
            self.v_avg = (1-1/i)*self.v_avg + (1/i)*self.v.clone().detach()
            self.v_tail.append(self.v.clone().detach())
            copy = self.w.clone().detach()
            self.optimizer_w.zero_grad()
            self.w.requires_grad = True
            L = self.objective_function()
            L.backward()
            lr = self.lr_w /(math.sqrt(i))
            # lr = self.lr_w
            for param_group in self.optimizer_w.param_groups:
                param_group['lr'] = lr
            w_grad_term = self.optimizer_w.step()
            self.w.requires_grad = False
            torch.clamp_(self.w, min=-1, max = 1)
            w_grad_norm = torch.norm(self.w-copy,p=1)/(lr)
            self.w_avg = (1-1/i)*self.w_avg + (1/i)*self.w.clone().detach()
            self.w_tail.append(self.w.clone().detach())
            while len(self.w_tail) > self.tail_fraction*i and len(self.w_tail)>1:
                self.w_tail.popleft()
                self.v_tail.popleft()
                print('kick one out')
            
            # self.w_avg = (1-(eta+1)/(i+eta))*w_avg + (eta+1)/(i+eta)*w.clone().detach()
            if i %10 == 0:
                print('\n')
                print('Iteration :', i)
                print('Current learning rate:', lr)
                print('grad norm w: ', w_grad_norm)
                print('current w: ', self.w)
                print('current v: ', self.v)
                print('average w: ', self.w_avg)
                print('average v: ', self.v_avg)
                print('tail average w: ', sum(self.w_tail) / len(self.w_tail))
                print('tail average v: ', sum(self.v_tail) / len(self.v_tail))
                print('objective:', L)
                if logging:
                    self.writer.add_scalar('Summary/1.current w', self.w,i)
                    self.writer.add_scalar('Summary/2.current v', self.v,i)
                    self.writer.add_scalar('Summary/3.tail average w', sum(self.w_tail) / len(self.w_tail), i)
                    self.writer.add_scalar('Summary/4.tail average v', sum(self.v_tail) / len(self.v_tail), i)
                    self.writer.add_scalar('Summary/5.average w', self.w_avg,i)
                    self.writer.add_scalar('Summary/6.average v', self.v_avg,i)
                    # self.writer.add_scalar('Summary/5.squared error wrt true v pi', (sum(self.w_collection)/len(self.w_collection)-v_star)**2, i)
                    # self.writer.add_scalar('Summary/6.l1 norm w - w mu hat', torch.norm(self.w - w_star),i)
                    # self.writer.add_scalar('Summary/7.grad norm w', torch.norm(self.w.grad),i)
                    # self.writer.add_scalar('Summary/8.td error v', self.td_error(self.v),i)

            if i % 1000 == 0:
                pdb.set_trace()
            
        pass

    def optimize_ogda(self):
        beta1 = 0
        beta2 = 0.999
        eps = 1e-1
        self.eta = self.lr_w
        self.w_e = self.zero.clone().detach()
        self.w_e_hat = [self.zero.clone().detach(),self.zero.clone().detach()]
        self.w_s = self.zero.clone().detach()
        self.w_s_hat = [self.zero.clone().detach(),self.zero.clone().detach()]
        self.v_e = self.zero.clone().detach()
        self.v_e_hat = [self.zero.clone().detach(),self.zero.clone().detach()]
        self.v_s = self.zero.clone().detach()
        self.v_s_hat = [self.zero.clone().detach(),self.zero.clone().detach()]

        for i in range(self.n_iterations):
            self.w.requires_grad = True
            if self.w.grad is not None:
                self.w.grad.zero_()
            # L = torch.matmul(torch.matmul(self.w, self.I), self.v)
            L = self.objective_function()
            L.backward()
            self.w_e = beta1 * self.w_e + (1-beta1)*self.w.grad
            self.w_s = beta2 * self.w_s +(1-beta2)* (self.w.grad**2)
            self.w_e_hat[0] = self.w_e_hat[1]
            self.w_e_hat[1] = self.w_e / (1-beta1**(i+1))
            self.w_s_hat[0] = self.w_s_hat[1]
            self.w_s_hat[1] = self.w_s / (1-beta2**(i+1))
            self.w.requires_grad = False
            # pdb.set_trace()
            lr_w = self.eta / math.sqrt(i+1)
            self.w = self.w - 2*self.eta*self.w_e_hat[1]/(self.w_s_hat[1].sqrt().add_(eps)) + self.eta*self.w_e_hat[0]/(self.w_s_hat[0].sqrt().add_(eps))
            # self.w = self.w - 2*lr_w*self.w_e_hat[1]/(self.w_s_hat[1].sqrt().add_(eps)) + lr_w*self.w_e_hat[0]/(self.w_s_hat[0].sqrt().add_(eps))
            
            
            torch.clamp_(self.w, min=-1, max=1)

            self.v.requires_grad = True
            if self.v.grad is not None:
                self.v.grad.zero_()
            # L = - torch.matmul(torch.matmul(self.w, self.I),self.v)
            L = -self.objective_function()
            L.backward()
            self.v_e = beta1 * self.v_e + (1-beta1)*self.v.grad
            self.v_s = beta2 * self.v_s +(1-beta2)* (self.v.grad**2)
            self.v_e_hat[0] = self.v_e_hat[1]
            self.v_e_hat[1] = self.v_e / (1-beta1**(i+1))
            self.v_s_hat[0] = self.v_s_hat[1]
            self.v_s_hat[1] = self.v_s / (1-beta2**(i+1))
            self.v.requires_grad = False
            lr_v = self.eta / math.sqrt(i+1)
            # self.v = self.v - 2*lr_v*self.v_e_hat[1]/(self.v_s_hat[1].sqrt().add_(eps)) + lr_v*self.v_e_hat[0]/(self.v_s_hat[0].sqrt().add_(eps))
            self.v = self.v - 2*self.eta*self.v_e_hat[1]/(self.v_s_hat[1].sqrt().add_(eps)) + self.eta*self.v_e_hat[0]/(self.v_s_hat[0].sqrt().add_(eps))
            # self.v = self.v - 2*self.eta*self.v_e_hat[1]/(math.sqrt(self.v_s_hat[1])+eps) + self.eta*self.v_e_hat[0]/(math.sqrt(self.v_s_hat[0])+eps)
            
            
            torch.clamp_(self.v, min=-1, max=1)

            print('\n')
            print('Iteration :', i)
            print('current w: ', self.w)
            print('current v: ', self.v)
            if i %1000 == 0:
                pdb.set_trace()
    
    def optimize_optimistic(self):
        for i in range(self.n_iterations):
            self.optimizer_w.zero_grad()
            self.optimizer_v.zero_grad()
            self.v.requires_grad = True
            L = -self.objective_function()
            L.backward()
            
            copy = self.v.clone().detach()
            if self.param.lr_decay:
                lr_v = self.lr_v /(math.sqrt(i+1))
                for param_group in self.optimizer_v.param_groups:
                    param_group['lr'] = lr_v
            v_grad_term = self.optimizer_v.step()
            self.v.requires_grad = False
            torch.clamp_(self.v, min=-1, max = 1)
            v_grad_norm = torch.norm(self.v-copy,p=1)/(self.lr_v)
            self.w.requires_grad = True
            L = self.objective_function()
            L.backward()
            
            copy = self.w.clone().detach()
            if self.param.lr_decay:
                lr_w = self.lr_w / (math.sqrt(i+1))
                for param_group in self.optimizer_w.param_groups:
                    param_group['lr'] = lr_w
            w_grad_term = self.optimizer_w.step()
            self.w.requires_grad = False
            torch.clamp_(self.w, min=-1, max = 1)
            w_grad_norm = torch.norm(self.w-copy,p=1)/(self.lr_w)
            print('\n')
            print('Iteration :', i)
            # print('Current learning rate:', lr_w)
            print('grad norm w: ', w_grad_norm)
            print('grad norm v: ', v_grad_norm)
            print('current w: ', self.w)
            print('current v: ', self.v)
            # print('average w: ', self.w_avg)
            # print('average v: ', self.v_avg)
            # print('tail average w: ', sum(self.w_tail) / len(self.w_tail))
            # print('tail average v: ', sum(self.v_tail) / len(self.v_tail))
            if i % 100 == 0:
                pdb.set_trace()
    
    def optimize_gda(self):
        for i in range(self.n_iterations):
            self.optimizer_w.zero_grad()
            self.optimizer_v.zero_grad()
            # self.w.requires_grad = False
            self.v.requires_grad = True
            L = -self.objective_function()
            L.backward()
            lr_v = self.lr_v /(math.sqrt(i+1))
            copy = self.v.clone().detach()
            for param_group in self.optimizer_v.param_groups:
                param_group['lr'] = lr_v
            v_grad_term = self.optimizer_v.step()
            self.v.requires_grad = False
            torch.clamp_(self.v, min=-1, max = 1)
            v_grad_norm = torch.norm(self.v-copy,p=1)/(lr_v)
            if i%10 ==0:
                self.w.requires_grad = True
                L = self.objective_function()
                L.backward()
                lr_w = self.lr_w / (math.sqrt(i+1))
                copy = self.w.clone().detach()
                for param_group in self.optimizer_w.param_groups:
                    param_group['lr'] = lr_w
                w_grad_term = self.optimizer_w.step()
                self.w.requires_grad = False
                torch.clamp_(self.w, min=-1, max = 1)
                w_grad_norm = torch.norm(self.w-copy,p=1)/(lr_w)
            print('\n')
            print('Iteration :', i)
            print('Current learning rate:', lr_w)
            print('grad norm w: ', w_grad_norm)
            print('grad norm v: ', v_grad_norm)
            print('current w: ', self.w)
            print('current v: ', self.v)
            # print('average w: ', self.w_avg)
            # print('average v: ', self.v_avg)
            # print('tail average w: ', sum(self.w_tail) / len(self.w_tail))
            # print('tail average v: ', sum(self.v_tail) / len(self.v_tail))
            if i % 1000 == 0:
                pdb.set_trace()




class AttrDict(object):
    def __init__(self, init=None):
        if init is not None:
            self.__dict__.update(init)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)

# def conjugate_gradient(grad_x, grad_y,
#                        x_params, y_params,
#                        b, x=None, nsteps=10,
#                        tol=1e-10, atol=1e-16,
#                        lr_x=1.0, lr_y=1.0,
#                        device=torch.device('cpu')):
#     """
#     :param grad_x:
#     :param grad_y:
#     :param x_params:
#     :param y_params:
#     :param b: vec
#     :param nsteps: max number of steps
#     :param residual_tol:
#     :return: A ** -1 * b
#     h_1 = D_yx * p
#     h_2 = D_xy * D_yx * p
#     A = I + lr_x * D_xy * lr_y * D_yx
#     """
#     if x is None:
#         x = torch.zeros(b.shape[0], device=device)
#         r = b.clone().detach()
#     else:
#         h1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=x, retain_graph=True).detach_().mul(lr_y)
#         h2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h1, retain_graph=True).detach_().mul(lr_x)
#         Avx = x + h2
#         r = b.clone().detach() - Avx

#     p = r.clone().detach()
#     rdotr = torch.dot(r, r)
#     residual_tol = tol * rdotr

#     for i in range(nsteps):
#         # To compute Avp
#         h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=p, retain_graph=True).detach_().mul(lr_y)
#         h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h_1, retain_graph=True).detach_().mul(lr_x)
#         Avp_ = p + h_2

#         alpha = rdotr / torch.dot(p, Avp_)
#         x.data.add_(alpha * p)

#         r.data.add_(- alpha * Avp_)
#         new_rdotr = torch.dot(r, r)
#         beta = new_rdotr / rdotr
#         p = r + beta * p
#         rdotr = new_rdotr
#         if rdotr < residual_tol or rdotr < atol:
#             break
#     if i > 99:
#         warnings.warn('CG iter num: %d' % (i + 1))
#     return x, i + 1


# def Hvp_vec(grad_vec, params, vec, retain_graph=False):
#     '''
#     return Hessian vector product
#     '''
#     if torch.isnan(grad_vec).any():
#         raise ValueError('Gradvec nan')
#     if torch.isnan(vec).any():
#         raise ValueError('vector nan')
#         # zero padding for None
#     grad_grad = autograd.grad(grad_vec, params, grad_outputs=vec, retain_graph=retain_graph,
#                               allow_unused=True)
#     grad_list = []
#     for i, p in enumerate(params):
#         if grad_grad[i] is None:
#             grad_list.append(torch.zeros_like(p).view(-1))
#         else:
#             grad_list.append(grad_grad[i].contiguous().view(-1))
#     hvp = torch.cat(grad_list)
#     if torch.isnan(hvp).any():
#         raise ValueError('hvp Nan')
#     return hvp

# def general_conjugate_gradient(grad_x, grad_y,
#                                x_params, y_params, b,
#                                lr_x, lr_y, x=None, nsteps=None,
#                                tol=1e-12, atol=1e-20,
#                                device=torch.device('cpu')):
#     '''
#     :param grad_x:
#     :param grad_y:
#     :param x_params:
#     :param y_params:
#     :param b:
#     :param lr_x:
#     :param lr_y:
#     :param x:
#     :param nsteps:
#     :param residual_tol:
#     :param device:
#     :return: (I + sqrt(lr_x) * D_xy * lr_y * D_yx * sqrt(lr_x)) ** -1 * b
#     '''
#     lr_x = lr_x.sqrt()
#     if x is None:
#         x = torch.zeros(b.shape[0], device=device)
#         r = b.clone().detach()
#     else:
#         h1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * x, retain_graph=True).mul_(lr_y)
#         h2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h1, retain_graph=True).mul_(lr_x)
#         Avx = x + h2
#         r = b.clone().detach() - Avx

#     if nsteps is None:
#         nsteps = b.shape[0]

#     if grad_x.shape != b.shape:
#         raise RuntimeError('CG: hessian vector product shape mismatch')
#     p = r.clone().detach()
#     rdotr = torch.dot(r, r)
#     residual_tol = tol * rdotr
#     for i in range(nsteps):
#         # To compute Avp
#         # h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * p, retain_graph=True)
#         h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * p, retain_graph=True).mul_(lr_y)
#         # h_1.mul_(lr_y)
#         # lr_y * D_yx * b
#         # h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=lr_y * h_1, retain_graph=True)
#         h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h_1, retain_graph=True).mul_(lr_x)
#         # h_2.mul_(lr_x)
#         # lr_x * D_xy * lr_y * D_yx * b
#         Avp_ = p + h_2

#         alpha = rdotr / torch.dot(p, Avp_)
#         x.data.add_(alpha * p)
#         r.data.add_(- alpha * Avp_)
#         new_rdotr = torch.dot(r, r)
#         beta = new_rdotr / rdotr
#         p = r + beta * p
#         rdotr = new_rdotr
#         if rdotr < residual_tol or rdotr < atol:
#             break
#     return x, i + 1

def main():
    #* choose a function to test minmax optimization 
    #* option 1: bilinear objective = x^T y
    #* option 2: scalar quadratic objective = -3x^2-y^2+4xy
    #* option 3: scalar quadratic objective = 3x^2 + y^2 + 4xy
    #* option 4: higher order polynomial and exponential objective = (4x^2 - (y-3x+0.05x^3)^2 - 0.1y^4) * exp(-0.01(x^2+y^2))
    #* option 5: quadratic games
    optimization_objective = 1
    #* specify optimization method
    optimization_method = 'ogda'
    # optimization_method = 'no_regret'
    # optimization_method = 'gda'
    # optimization_method = 'cgd'
    # optimization_method = 'acgd'
    
    # initialize w and v
    dim = 1
    if optimization_objective == 2 or optimization_objective == 3 or optimization_objective == 4:
        dim = 1
    w = torch.rand(dim, dtype = dtype)*2-torch.ones(dim, dtype=dtype)
    v = torch.rand(dim, dtype = dtype)*2-torch.ones(dim, dtype=dtype)
    v_bound = (-1,1)
    w_bound = (-1,1)
    
    n_iterations = 10000
    tail_fraction = 0.1
    lr_decay = True
    # define parameters for optimistic gradient descent
    if optimization_method == 'ogda':
        lr_w = 0.01
        lr_v = 0.01
        params = {'objective': optimization_objective,
                  'method': optimization_method, 
                  'lr_w': lr_w, 
                  'lr_v': lr_v, 
                  'n_iterations': n_iterations, 
                  'v_bound': v_bound, 
                  'w_bound':w_bound, 
                  'tail_fraction': tail_fraction, 
                  'lr_decay':lr_decay}
    if optimization_method == 'cgd':
        lr_w = 0.1
        lr_v = 0.1
        params = {'objective': optimization_objective,
                  'method': optimization_method, 
                  'lr_w': lr_w, 
                  'lr_v': lr_v, 
                  'n_iterations': n_iterations, 
                  'v_bound': v_bound, 
                  'w_bound':w_bound, 
                  'tail_fraction': tail_fraction, 
                  'lr_decay':lr_decay}
    if optimization_method == 'acgd':
        lr_w = 0.1
        lr_v = 0.1
        params = {'objective': optimization_objective,
                  'method': optimization_method, 
                  'lr_w': lr_w, 
                  'lr_v': lr_v, 
                  'n_iterations': n_iterations, 
                  'v_bound': v_bound, 
                  'w_bound':w_bound, 
                  'tail_fraction': tail_fraction, 
                  'lr_decay':lr_decay}

    # define parameters for greedy - no regret optimization
    if optimization_method == 'no_regret':
        lr_w = 0.1
        # lr_v = 0.01
        # params = {'method': optimization_method, 'lr_w': lr_w, 'n_iterations': n_iterations, 'v_bound': v_bound, 'w_bound':w_bound, 'tail_fraction': tail_fraction}
        params = {'objective': optimization_objective,
                'method': optimization_method, 
                'lr_w': lr_w, 
                'n_iterations': n_iterations, 
                'v_bound': v_bound, 
                'w_bound':w_bound, 
                'tail_fraction': tail_fraction, 
                'lr_decay':lr_decay}
    if optimization_method == 'gda':
        lr_w = 0.1
        lr_v = 0.1
        tail_fraction = 0.1
        # params = {'method': optimization_method, 'lr_w': lr_w, 'lr_v': lr_v, 'n_iterations': n_iterations, 'v_bound': v_bound, 'w_bound':w_bound, 'tail_fraction': tail_fraction}
        params = {'objective': optimization_objective,
                  'method': optimization_method, 
                  'lr_w': lr_w, 
                  'lr_v': lr_v, 
                  'n_iterations': n_iterations, 
                  'v_bound': v_bound, 
                  'w_bound':w_bound, 
                  'tail_fraction': tail_fraction, 
                  'lr_decay':lr_decay}

    optimization_params = AttrDict(params)
    # optimization_params.update({'method': optimization_method, 'lr_w': lr_w, 'lr_v': lr_v, 'n_iterations': n_iterations})
    opt = MinMax_Optimizer(w,v,optimization_params)
    opt.optimize()

    
if __name__ == '__main__':
    main()