import math
import torch
import torch.autograd as autograd
import warnings
import time

class Adam(torch.optim.Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        # if not 0.0 <= lr:
            # raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class ModifiedAdam(torch.optim.Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        # if not 0.0 <= lr:
            # raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(ModifiedAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ModifiedAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                # self.state[p]['step'] = 0
                self.state[p]['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                self.state[p]['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    self.state[p]['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                # import pdb; pdb.set_trace()
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]
                # import pdb;pdb.set_trace()
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # print(bias_correction2)

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq/ bias_correction2, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt()).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                # print(bias_correction2)
                # import pdb;pdb.set_trace()

                p.addcdiv_(exp_avg, denom, value=-step_size)
        return exp_avg / denom
        # return loss

class OptimisticAdam(torch.optim.Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        # if not 0.0 <= lr:
            # raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(OptimisticAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(OptimisticAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                # self.state[p]['step'] = 0
                self.state[p]['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                self.state[p]['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    self.state[p]['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                # import pdb; pdb.set_trace()
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]
                # import pdb;pdb.set_trace()
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['num'] = [torch.zeros_like(p, memory_format=torch.preserve_format), torch.zeros_like(p, memory_format=torch.preserve_format)]
                    state['denom'] = [torch.zeros_like(p, memory_format=torch.preserve_format), torch.zeros_like(p, memory_format=torch.preserve_format)+group['eps']]
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                state['num'][0] = state['num'][1]
                state['denom'][0] = state['denom'][1]
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # print(bias_correction2)

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                num = exp_avg / bias_correction1
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq/ bias_correction2, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt()).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                state['num'][1] = num.clone()
                
                state['denom'][1] = denom.clone()
                # if state['step'] == 1:
                #     grad_term = num/denom
                # else:
                grad_term = 2*num/denom -  state['num'][0]/state['denom'][0]
                # p = p - group['lr']*grad_term
                p.add_(- group['lr']*grad_term)
                # import pdb;pdb.set_trace()
                # step_size = group['lr'] / bias_correction1
                # print(bias_correction2)
                # import pdb;pdb.set_trace()

                # p.addcdiv_(exp_avg, denom, value=-step_size)
        return grad_term
        # return exp_avg / denom
        # return loss

class ACGD(object):
    def __init__(self, max_params, min_params,
                 lr_max=1e-3, lr_min=1e-3,
                 eps=1e-5, beta=0.99,
                 tol=1e-12, atol=1e-20,
                 device=torch.device('cpu'),
                 solve_x=False, collect_info=True):
        # self.max_params = list(max_params)
        # self.min_params = list(min_params)
        self.max_params = max_params
        self.min_params = min_params
        self.state = {'lr_max': lr_max, 'lr_min': lr_min,
                      'eps': eps, 'solve_x': solve_x,
                      'tol': tol, 'atol': atol,
                      'beta': beta, 'step': 0,
                      'old_max': None, 'old_min': None,  # start point of CG
                      'sq_exp_avg_max': None, 'sq_exp_avg_min': None}  # save last update
        self.info = {'grad_x': None, 'grad_y': None,
                     'hvp_x': None, 'hvp_y': None,
                     'cg_x': None, 'cg_y': None,
                     'time': 0, 'iter_num': 0}
        self.device = device
        self.collect_info = collect_info

    def zero_grad(self):
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def get_info(self):
        if self.info['grad_x'] is None:
            print('Warning! No update information stored. Set collect_info=True before call this method')
        return self.info

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)
        print('Load state: {}'.format(state_dict))

    def set_lr(self, lr_max, lr_min):
        self.state.update({'lr_max': lr_max, 'lr_min': lr_min})
        print('Maximizing side learning rate: {:.4f}\n '
              'Minimizing side learning rate: {:.4f}'.format(lr_max, lr_min))

    def step(self, loss):
        lr_max = self.state['lr_max']
        lr_min = self.state['lr_min']
        beta = self.state['beta']
        eps = self.state['eps']
        tol = self.state['tol']
        atol = self.state['atol']
        time_step = self.state['step'] + 1
        self.state['step'] = time_step

        grad_x = autograd.grad(loss, self.max_params, create_graph=True, retain_graph=True)
        # import pdb; pdb.set_trace()
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(loss, self.min_params, create_graph=True, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])
        grad_x_vec_d = grad_x_vec.clone().detach()
        grad_y_vec_d = grad_y_vec.clone().detach()

        sq_avg_x = self.state['sq_exp_avg_max']
        sq_avg_y = self.state['sq_exp_avg_min']
        sq_avg_x = torch.zeros_like(grad_x_vec_d, requires_grad=False) if sq_avg_x is None else sq_avg_x
        sq_avg_y = torch.zeros_like(grad_y_vec_d, requires_grad=False) if sq_avg_y is None else sq_avg_y
        sq_avg_x.mul_(beta).addcmul_(1 - beta, grad_x_vec_d, grad_x_vec_d)
        sq_avg_y.mul_(beta).addcmul_(1 - beta, grad_y_vec_d, grad_y_vec_d)

        bias_correction = 1 - beta ** time_step
        lr_max = math.sqrt(bias_correction) * lr_max / sq_avg_x.sqrt().add(eps)
        lr_min = math.sqrt(bias_correction) * lr_min / sq_avg_y.sqrt().add(eps)

        scaled_grad_x = torch.mul(lr_max, grad_x_vec_d)
        scaled_grad_y = torch.mul(lr_min, grad_y_vec_d)
        hvp_x_vec = Hvp_vec(grad_y_vec, self.max_params, scaled_grad_y,
                            retain_graph=True)  # h_xy * d_y
        hvp_y_vec = Hvp_vec(grad_x_vec, self.min_params, scaled_grad_x,
                            retain_graph=True)  # h_yx * d_x
        p_x = torch.add(grad_x_vec_d, - hvp_x_vec)
        p_y = torch.add(grad_y_vec_d, hvp_y_vec)
        if self.collect_info:
            norm_px = torch.norm(hvp_x_vec, p=2).item()
            norm_py = torch.norm(hvp_y_vec, p=2).item()
            timer = time.time()

        if self.state['solve_x']:
            p_y.mul_(lr_min.sqrt())
            cg_y, iter_num = general_conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                                        x_params=self.min_params, y_params=self.max_params,
                                                        b=p_y, x=self.state['old_min'],
                                                        tol=tol, atol=atol,
                                                        lr_x=lr_min, lr_y=lr_max, device=self.device)
            old_min = cg_y.detach_()
            min_update = cg_y.mul(- lr_min.sqrt())
            hcg = Hvp_vec(grad_y_vec, self.max_params, min_update).detach_()
            hcg.add_(grad_x_vec_d)
            max_update = hcg.mul(lr_max)
            old_max = hcg.mul(lr_max.sqrt())
        else:
            p_x.mul_(lr_max.sqrt())
            cg_x, iter_num = general_conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                                        x_params=self.max_params, y_params=self.min_params,
                                                        b=p_x, x=self.state['old_max'],
                                                        tol=tol, atol=atol,
                                                        lr_x=lr_max, lr_y=lr_min, device=self.device)
            old_max = cg_x.detach_()
            max_update = cg_x.mul(lr_max.sqrt())
            hcg = Hvp_vec(grad_x_vec, self.min_params, max_update).detach_()
            hcg.add_(grad_y_vec_d)
            min_update = hcg.mul(- lr_min)
            old_min = hcg.mul(lr_min.sqrt())
        self.state.update({'old_max': old_max, 'old_min': old_min,
                           'sq_exp_avg_max': sq_avg_x, 'sq_exp_avg_min': sq_avg_y})

        if self.collect_info:
            timer = time.time() - timer
            self.info.update({'time': timer, 'iter_num': iter_num,
                              'hvp_x': norm_px, 'hvp_y': norm_py})

        index = 0
        for p in self.max_params:
            p.data.add_(max_update[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == max_update.numel(), 'Maximizer CG size mismatch'

        index = 0
        for p in self.min_params:
            p.data.add_(min_update[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == min_update.numel(), 'Minimizer CG size mismatch'

        if self.collect_info:
            norm_gx = torch.norm(grad_x_vec, p=2).item()
            norm_gy = torch.norm(grad_y_vec, p=2).item()
            norm_cgx = torch.norm(max_update, p=2).item()
            norm_cgy = torch.norm(min_update, p=2).item()
            self.info.update({'grad_x': norm_gx, 'grad_y': norm_gy,
                              'cg_x': norm_cgx, 'cg_y': norm_cgy})
        self.state['solve_x'] = False if self.state['solve_x'] else True



def conjugate_gradient(grad_x, grad_y,
                       x_params, y_params,
                       b, x=None, nsteps=10,
                       tol=1e-10, atol=1e-16,
                       lr_x=1.0, lr_y=1.0,
                       device=torch.device('cpu')):
    """
    :param grad_x:
    :param grad_y:
    :param x_params:
    :param y_params:
    :param b: vec
    :param nsteps: max number of steps
    :param residual_tol:
    :return: A ** -1 * b
    h_1 = D_yx * p
    h_2 = D_xy * D_yx * p
    A = I + lr_x * D_xy * lr_y * D_yx
    """
    if x is None:
        x = torch.zeros(b.shape[0], device=device)
        r = b.clone().detach()
    else:
        h1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=x, retain_graph=True).detach_().mul(lr_y)
        h2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h1, retain_graph=True).detach_().mul(lr_x)
        Avx = x + h2
        r = b.clone().detach() - Avx

    p = r.clone().detach()
    rdotr = torch.dot(r, r)
    residual_tol = tol * rdotr

    for i in range(nsteps):
        # To compute Avp
        h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=p, retain_graph=True).detach_().mul(lr_y)
        h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h_1, retain_graph=True).detach_().mul(lr_x)
        Avp_ = p + h_2

        alpha = rdotr / torch.dot(p, Avp_)
        x.data.add_(alpha * p)

        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol or rdotr < atol:
            break
    if i > 99:
        warnings.warn('CG iter num: %d' % (i + 1))
    return x, i + 1


def Hvp_vec(grad_vec, params, vec, retain_graph=False):
    '''
    return Hessian vector product
    '''
    if torch.isnan(grad_vec).any():
        raise ValueError('Gradvec nan')
    if torch.isnan(vec).any():
        import pdb; pdb.set_trace()
        raise ValueError('vector nan')
        # zero padding for None
    grad_grad = autograd.grad(grad_vec, params, grad_outputs=vec, retain_graph=retain_graph,
                              allow_unused=True)
    grad_list = []
    for i, p in enumerate(params):
        # import pdb; pdb.set_trace()
        if grad_grad[i] is None:
            grad_list.append(torch.zeros_like(p).view(-1))
        else:
            grad_list.append(grad_grad[i].contiguous().view(-1))
    hvp = torch.cat(grad_list)
    if torch.isnan(hvp).any():
        raise ValueError('hvp Nan')
    return hvp


def general_conjugate_gradient(grad_x, grad_y,
                               x_params, y_params, b,
                               lr_x, lr_y, x=None, nsteps=None,
                               tol=1e-12, atol=1e-20,
                               device=torch.device('cpu')):
    '''
    :param grad_x:
    :param grad_y:
    :param x_params:
    :param y_params:
    :param b:
    :param lr_x:
    :param lr_y:
    :param x:
    :param nsteps:
    :param residual_tol:
    :param device:
    :return: (I + sqrt(lr_x) * D_xy * lr_y * D_yx * sqrt(lr_x)) ** -1 * b
    '''
    lr_x = lr_x.sqrt()
    if x is None:
        x = torch.zeros(b.shape[0], device=device)
        r = b.clone().detach()
    else:
        h1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * x, retain_graph=True).mul_(lr_y)
        h2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h1, retain_graph=True).mul_(lr_x)
        Avx = x + h2
        r = b.clone().detach() - Avx

    if nsteps is None:
        nsteps = b.shape[0]

    if grad_x.shape != b.shape:
        raise RuntimeError('CG: hessian vector product shape mismatch')
    p = r.clone().detach()
    rdotr = torch.dot(r, r)
    residual_tol = tol * rdotr
    for i in range(nsteps):
        # To compute Avp
        # h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * p, retain_graph=True)
        h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * p, retain_graph=True).mul_(lr_y)
        # h_1.mul_(lr_y)
        # lr_y * D_yx * b
        # h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=lr_y * h_1, retain_graph=True)
        h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h_1, retain_graph=True).mul_(lr_x)
        # h_2.mul_(lr_x)
        # lr_x * D_xy * lr_y * D_yx * b
        Avp_ = p + h_2

        alpha = rdotr / torch.dot(p, Avp_)
        x.data.add_(alpha * p)
        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol or rdotr < atol:
            break
    return x, i + 1


def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()

class Nadam(torch.optim.Optimizer):
    """Implements Nadam algorithm (a variant of Adam based on Nesterov momentum).
    It has been proposed in `Incorporating Nesterov Momentum into Adam`__.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        schedule_decay (float, optional): momentum schedule decay (default: 4e-3)
    __ http://cs229.stanford.edu/proj2015/054_report.pdf
    __ http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
    """

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, schedule_decay=4e-3):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, schedule_decay=schedule_decay)
        super(Nadam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m_schedule'] = 1.
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                # Warming momentum schedule
                m_schedule = state['m_schedule']
                schedule_decay = group['schedule_decay']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                eps = group['eps']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                momentum_cache_t = beta1 * \
                    (1. - 0.5 * (0.96 ** (state['step'] * schedule_decay)))
                momentum_cache_t_1 = beta1 * \
                    (1. - 0.5 *
                     (0.96 ** ((state['step'] + 1) * schedule_decay)))
                m_schedule_new = m_schedule * momentum_cache_t
                m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1
                state['m_schedule'] = m_schedule_new

                # Decay the first and second moment running average coefficient
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg_sq_prime = exp_avg_sq.div(1. - bias_correction2)

                denom = exp_avg_sq_prime.sqrt_().add_(group['eps'])

                p.data.addcdiv_(-group['lr'] * (1. - momentum_cache_t) / (1. - m_schedule_new), grad, denom)
                p.data.addcdiv_(-group['lr'] * momentum_cache_t_1 / (1. - m_schedule_next), exp_avg, denom)

        return loss

required = object()
class Extragradient(torch.optim.Optimizer):
    """Base class for optimizers with extrapolation step.
        Arguments:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """
    def __init__(self, params, defaults):
        super(Extragradient, self).__init__(params, defaults)
        self.params_copy = []

    def update(self, p, group):
        raise NotImplementedError

    def extrapolation(self):
        """Performs the extrapolation step and save a copy of the current parameters for the update step.
        """
        # Check if a copy of the parameters was already made.
        is_empty = len(self.params_copy) == 0
        for group in self.param_groups:
            for p in group['params']:
                u = self.update(p, group)
                if is_empty:
                    # Save the current parameters for the update step. Several extrapolation step can be made before each update but only the parameters before the first extrapolation step are saved.
                    self.params_copy.append(p.data.clone())
                if u is None:
                    continue
                # Update the current parameters
                p.data.add_(u)
        return u #hoang: added this, previously there was no return

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if len(self.params_copy) == 0:
            raise RuntimeError('Need to call extrapolation before calling step.')

        loss = None
        if closure is not None:
            loss = closure()

        i = -1
        for group in self.param_groups:
            for p in group['params']:
                i += 1
                u = self.update(p, group)
                if u is None:
                    continue
                # Update the parameters saved during the extrapolation step
                p.data = self.params_copy[i].add_(u)


        # Free the old parameters
        self.params_copy = []
        # return loss #hoang: replace return loss by return u
        return u #hoang: added this

class SGD(torch.optim.Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}
        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the 
        parameters, gradient, velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        # if lr is not required and lr < 0.0:
            # raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss

class ExtraSGD(Extragradient):
    """Implements stochastic gradient descent with extrapolation step (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.ExtraSGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.extrapolation()
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ExtraSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def update(self, p, group):
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        if p.grad is None:
            return None
        d_p = p.grad.data
        if weight_decay != 0:
            d_p.add_(weight_decay, p.data)
        if momentum != 0:
            param_state = self.state[p]
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                buf.mul_(momentum).add_(d_p)
            else:
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(1 - dampening, d_p)
            if nesterov:
                d_p = d_p.add(momentum, buf)
            else:
                d_p = buf

        return -group['lr']*d_p

class ExtraAdam(Extragradient):
    """Implements the Adam algorithm with extrapolation step.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        # if not 0.0 <= lr:
        #  raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
         raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
         raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
         raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                     weight_decay=weight_decay, amsgrad=amsgrad)
        super(ExtraAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ExtraAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def update(self, p, group):
        if p.grad is None:
            return None
        grad = p.grad.data
        if grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
        amsgrad = group['amsgrad']

        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        if group['weight_decay'] != 0:
            grad = grad.add(group['weight_decay'], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

        return -step_size*exp_avg/denom

class ModifiedExtraAdam(Extragradient):
    """Implements the Adam algorithm with extrapolation step.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        # if not 0.0 <= lr:
        #  raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
         raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
         raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
         raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                     weight_decay=weight_decay, amsgrad=amsgrad)
        super(ModifiedExtraAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ModifiedExtraAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def update(self, p, group):
        if p.grad is None:
            return None
        grad = p.grad.data
        if grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
        amsgrad = group['amsgrad']

        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        if group['weight_decay'] != 0:
            grad = grad.add(group['weight_decay'], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            # torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            torch.max(max_exp_avg_sq, exp_avg_sq / bias_correction2, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        # step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
        step_size = group['lr']  / bias_correction1

        return -step_size*exp_avg/denom



class RMSprop(torch.optim.Optimizer):
    r"""Implements RMSprop algorithm.
    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.
    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.
    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        # if not 0.0 <= lr:
            # raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.add_(buf, alpha=-group['lr'])
                else:
                    p.addcdiv_(grad, avg, value=-group['lr'])

        return loss