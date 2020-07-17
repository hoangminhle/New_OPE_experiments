import numpy as np
import torch
import pdb

from util import roll_out
from util import *
from baselines import *
from tabular_mvm_estimator import Tabular_State_MVM_Estimator
from attrdict import AttrDict
from config import Config
import random
from random import randint

dtype = torch.double
device = 'cpu'
# initial_seed = randint(0,10000)
initial_seed = 4252
RUN_SANITY_CHECK = False
config = {
    'env': 'grid_world', #'grid_world', #'taxi', #'grid_world',
    'horizon': 32,
    'n_trajectories': 256,
    'gamma': 0.99,
    'stochastic_env': False,
    'n_experiments': 1,
    'logdir': 'debug/gridworld/',
    'initial_seed': initial_seed,
    'baseline': ['IS', 'IH', 'MB'],
    'lr_w': 1, #0.1,
    'objective': ['bias_cf', 'bias_opt_cf'],# , 'bias_minmax', 'bias_td_minmax'], #['bias_no_td_regW','bias_td_regW','bias_no_td_no_regW','bias_td_no_regW'], #['bias_no_td_no_regW'],#['bias_no_td_regW','bias_td_regW','bias_restricted_td_regW'],#['bias_no_td_regW','bias_td_regW','bias_no_td_no_regW','bias_td_no_regW'], #['bias_cf', 'bias_td_v2_cf'],#['bias_cf', 'bias_td_v1_cf', 'bias_td_v2_cf' ], #['bias_td_delay'], #['bias_td', 'bias'],
    'penalty_input': 1,
    'reg_w': 'linfty', #'linfty', #'linfty', #or 'l2'
    'coeff_reg_w': 2, #1e-4, #2, # 2 for infinity norm projection, 1e-4 for l2 regularization
    'v_class_cardinality':100,
    'normalizing_factor': 1,
    'n_iterations': 20000,
    'lr_decay': True,
    'beta': (0,0.99),
    'eps': 1e-8,
    'tail_fraction': 0.1,
    'logging': False,
    'normalize_rewards': True,
    'estimate_mu': True,
}
cfg = Config(config)

def prepare_behavior_data(behavior_data):
    behavior_data_squeezed = np.squeeze(behavior_data)
    s = torch.tensor(behavior_data_squeezed[:,:,0], dtype = torch.long)
    a = torch.tensor(behavior_data_squeezed[:,:,1], dtype = torch.long)
    sn = torch.tensor(behavior_data_squeezed[:,:,2], dtype = torch.long)
    r = torch.tensor(behavior_data_squeezed[:,:,3], dtype = dtype)
    return (s,a,sn,r)
def run_model_based_tabular_estimator(behavior_data, pi, n_states, n_actions, horizon):
    import torch.nn.functional as F
    d_mu_count = torch.zeros(n_states*n_actions)
    P_count = torch.zeros(n_states*n_actions, n_states)
    Pi = torch.zeros(n_states, n_states*n_actions)
    p0_count = torch.zeros(n_states)
    R_sum = torch.zeros(n_states*n_actions)
    R_count = torch.zeros(n_states*n_actions)
    (Ss, As, SNs, Rs) = behavior_data
    
    pi_torch = torch.tensor(pi)
    for s in range(n_states):
        Pi[s,s*n_actions:(s+1)*n_actions] = pi_torch[s,:]

    for episode in range(cfg.n_trajectories):
        time_step = 0
        for (s,a,sn,r) in zip(Ss[episode], As[episode], SNs[episode], Rs[episode]):
            d_mu_count[s*n_actions+a] += cfg.gamma**time_step #1
            P_count[s*n_actions+a,sn] += 1
            R_sum[s*n_actions+a] += r
            R_count[s*n_actions+a] += 1
            if time_step == 0:
                p0_count[s] += 1
            time_step += 1

    d_mu_hat = F.normalize(d_mu_count, p=1, dim = 0)
    P = F.normalize(P_count,p=1,dim=1)
    # support_sa = (abs(torch.sum(P[0:(n_states-1)*n_actions], axis=1) - torch.ones( (n_states-1)*n_actions))<1e-6).nonzero()
    missing_sa = (abs(torch.sum(P[0:(n_states-1)*n_actions], axis=1) - torch.ones( (n_states-1)*n_actions))>1e-6).nonzero()
    # pdb.set_trace()
    P_pi = torch.mm(P, Pi)
    p0 = F.normalize(p0_count,p=1,dim=0)
    p0_pi = torch.matmul(Pi.t(), p0)
    I = torch.eye(n_states*n_actions)
    accumulated_transition = I
    for h in range(horizon):
        accumulated_transition = I + cfg.gamma*torch.mm(P_pi, accumulated_transition)
    
    # R = R_count / d_mu_count
    R = R_sum / R_count
    R[R != R] = 0        
    q = torch.matmul(accumulated_transition, R)
    value = torch.matmul(p0_pi.t(), q)
    return q, value    

def run_q_based_estimator(behavior_data, q, pi, n_actions):
    (s, a, sn, r) = behavior_data
    s_expanded = s[:,:,None]*n_actions+torch.arange(n_actions)
    s0 = s[:,0]
    s0_expanded = s_expanded[:,0]
    pi_torch = torch.tensor(pi.prob_table)
    q_based_estimate = torch.mean((q[s0_expanded]*pi_torch[s0,:]).sum(dim=1))
    return q_based_estimate

def display(result):
    estimate, mse = result
    for key,val in mse.items():
        if key == 'IH' or key == 'IS' or key== 'MB':
            assert len(val) == cfg.n_experiments
            assert len(estimate[key]) == cfg.n_experiments

    longest = max([len(key) for key,_ in mse.items()])
    sorted_keys = []
    for key,val in mse.items():
        if len(val) >0:
            sorted_keys.append([key,sum(val)/len(val) ])
    sorted_keys = np.array(sorted_keys)

    # sorted_keys = np.array([[key,sum(val)/len(val) ] for key,val in mse.items()])
    sorted_keys = sorted_keys[np.argsort(sorted_keys[:,1].astype(float))]

    print('\n')
    stochasticity = 'stochastic env' if cfg.stochastic_env == True else 'deterministic env'
    print (f"Ordered Results over {cfg.n_experiments} Experiments, horizon {cfg.horizon}, trajectories {cfg.n_trajectories}, {stochasticity} (domain {cfg.env}):  \n")
    for key in sorted_keys[:,0]:
        if len(estimate[key])>0:
            value = sum(estimate[key])/len(estimate[key])
            error = sum(mse[key])/len(mse[key])
        else:
            value = np.nan
            error = np.nan
        label = ' '*(longest-len(key)) + key
        print("Method: {} ---- Avg Estimate: {:10.4f}. MSE: {:10.4f}".format(label, value, error))

def main(cfg):
    initial_seed = cfg.initial_seed
    random.seed(initial_seed)
    np.random.seed(initial_seed)
    gamma = cfg.gamma
    n_trajectories = cfg.n_trajectories
    horizon = cfg.horizon
    horizon_normalization = (1-gamma**horizon)/(1-gamma)
    processor = lambda x: x
    seed_list = [initial_seed + np.random.randint(0,10000)*i for i in range(cfg.n_experiments)] # generate a list of random seeds
    if cfg.env == 'grid_world':
        from gridworld import GridWorldEnv
        env = GridWorldEnv()
    elif cfg.env == 'taxi':
        from taxi import TaxiEnv
        env = TaxiEnv()

    n_states = env.nS; n_actions = env.nA; P = env.P_matrix; R = env.R_matrix.copy(); d0 = env.isd
    
    q_star_original = env.value_iteration()
    # pi_prob = gymEnv.extract_policy(q_star_original, temperature=0.05)
    # mu_prob = gymEnv.extract_policy(q_star_original, temperature=1)
    pi = env.extract_policy(q_star_original, temperature=0.1)
    # mu = env.extract_policy(q_star_original, temperature=0.3)
    # pi = env.extract_policy(q_star_original, temperature=0.15)
    # mu = env.extract_policy(q_star_original, temperature=0.3)
    # mu = pi.copy(); mu[:,1] = pi[:,2].copy(); mu[:,2] = pi[:,1].copy()
    mu = pi.copy(); mu[:,0] = pi[:,1].copy(); mu[:,1] = pi[:,2].copy(); mu[:,2] = pi[:,3].copy(); mu[:,3] = pi[:,0].copy()

    dpi, dpi_t, v_pi_s, P_pi = exact_calculation(env,pi, cfg.horizon, cfg.gamma)
    dmu, dmu_t, vmu_s, P_mu = exact_calculation(env,mu, cfg.horizon, cfg.gamma)
    #! sanity check the loss objective
    #* verify the claim that L(w*,f) = 0 for all f, where
    #* L(w,f) = \E_{(s,a,s')\sim d_mu} [ w(s) (f(s) - gamm*rho(s,a)*f(s'))] +1/h E_{s\sim d0} [f(s)] - 1/h *gamma^horizon \E_{s\sim d_pi,H}[f(s)]
    # determine w_star
    w_star = np.nan_to_num(dpi/dmu)
    v_pi = np.sum(d0*v_pi_s); v_mu = np.sum(d0*vmu_s)
    
    dpi_H = np.dot(P_pi.T, dpi_t[:,horizon-1])
    dmu_H = np.dot(P_mu.T, dmu_t[:,horizon-1])
    if RUN_SANITY_CHECK:
        def L(w,f):
            loss = 0
            for s in range(n_states):
                for a in range(n_actions):
                    for sn in range(n_states):
                        loss += w[s]*(-f[s] + gamma*pi[s,a]/mu[s,a] * f[sn])*dmu[s]*mu[s,a]*P[s,a,sn]
            
            loss += 1/horizon_normalization* np.sum(d0*f)
            loss -= 1/horizon_normalization * gamma**horizon*np.sum(dpi_H*f)
            return loss
        f = np.random.rand(n_states)
        loss = L(w_star, f)
        assert abs(loss) <1e-8

        #! sanity check bellman and td error
        R_pi = np.sum(R * pi, axis = -1)
        bellman_original = v_pi_s - R_pi - gamma * np.dot(P_pi, v_pi_s)
        bellman_new = v_pi_s - np.dot((np.identity(n_states) - np.linalg.matrix_power(gamma*P_pi, horizon)),R_pi) - gamma*np.dot(P_pi, v_pi_s)
        pdb.set_trace()

    ground_truth_info = AttrDict({}) 
    ground_truth_info.update({'d_pi': torch.tensor(dpi, dtype=dtype), 'd_mu': torch.tensor(dmu, dtype=dtype), 'v_pi': torch.tensor(v_pi_s, dtype=dtype), 'v_star': v_pi})
    ground_truth_info.update({'w_pi': w_star})
    ground_truth_info.update({'P': torch.tensor(env.P_matrix, dtype=dtype)})
    ground_truth_info.update({'pi': torch.tensor(pi, dtype=dtype), 'mu': torch.tensor(mu, dtype=dtype)})
    true_rho = torch.tensor(pi/mu, dtype = dtype)
    true_rho[true_rho!= true_rho] = 0; true_rho[torch.isinf(true_rho)] = 0
    ground_truth_info.update({'rho': true_rho})
    ground_truth_info.update({'d0': torch.tensor(env.isd, dtype = dtype)})
    ground_truth_info.update({'R': torch.tensor(env.R_matrix, dtype = dtype)})
    ground_truth_info.update({'d_pi_H': torch.tensor(dpi_H, dtype = dtype)}); ground_truth_info.update({'d_mu_H': torch.tensor(dmu_H, dtype = dtype)})
    ground_truth_info.update({'d_pi_t':torch.tensor(dpi_t, dtype=dtype),'d_mu_t':torch.tensor(dmu_t, dtype=dtype)})
    

    estimate = {}
    squared_error = {}
    estimate.update({'True pi': [float(v_pi)]}); squared_error.update({'True pi':[0]})
    estimate.update({'True mu': [float(v_mu)]}); squared_error.update({'True mu':[float(v_mu-v_pi)**2]})

    #* Generate multiple sets of behavior data from mu
    training_data = []
    training_data_processed = []
    for _ in range(cfg.n_experiments):
        print('Experiment:',_)
        print('------------------------')
        np.random.seed(seed_list[_])
        env.seed(seed_list[_])
        # behavior_data = rollout(env, mu, processor, absorbing_state, pi_e = pi, N=n_trajectories, T=horizon, frameskip=1, frameheight=1, path=None, filename='tmp',)
        behavior_data, _, _ = roll_out(env, mu, n_trajectories, horizon)
        behavior_data_processed = prepare_behavior_data(behavior_data)
        training_data.append(behavior_data)
        training_data_processed.append(behavior_data_processed)
        # pdb.set_trace()        
    estimate['IS'], estimate['STEP IS'], estimate['WIS'], estimate['STEP WIS'], estimate['Mu hat'] = [], [], [], [], []
    squared_error['IS'] = []; squared_error['STEP IS'] = []; squared_error['WIS'] = []; squared_error['STEP WIS'] = []; squared_error['Mu hat'] = []
    estimate['IH_SN'] = []; squared_error['IH_SN'] = []
    estimate['IH_no_SN'] = []; squared_error['IH_no_SN'] = []
    estimate['MB']=[]; squared_error['MB']=[]

    ###* Looping over the number of baseline experiments
    for _ in range(cfg.n_experiments):
        behavior_data = training_data[_]
        behavior_data_processed = training_data_processed[_]

        IS = importance_sampling_estimator(behavior_data, mu, pi, gamma)
        step_IS = importance_sampling_estimator_stepwise(behavior_data, mu, pi, gamma)
        WIS = weighted_importance_sampling_estimator(behavior_data, mu, pi, gamma)
        step_WIS = weighted_importance_sampling_estimator_stepwise(behavior_data, mu, pi, gamma)
        estimate['IS'].append(float(IS)); squared_error['IS'].append(float( (IS - v_pi )**2))
        estimate['STEP IS'].append(float(step_IS)); squared_error['STEP IS'].append(float( (step_IS - v_pi )**2))
        estimate['WIS'].append(float(WIS)); squared_error['WIS'].append(float( (WIS - v_pi )**2))
        estimate['STEP WIS'].append(float(step_WIS)); squared_error['STEP WIS'].append(float( (step_WIS - v_pi )**2))
        MB = model_based(n_states, n_actions, behavior_data, pi, gamma)
        estimate['MB'].append(float(MB)); squared_error['MB'].append(float((MB- v_pi)**2))
        IH, IH_unnormalized = lihong_infinite_horizon(n_states, behavior_data, mu, pi, gamma)
        estimate['IH_SN'].append(float(IH)); squared_error['IH_SN'].append(float((IH - v_pi )**2))
        estimate['IH_no_SN'].append(float(IH_unnormalized)); squared_error['IH_no_SN'].append(float((IH_unnormalized - v_pi )**2))
            
    display((estimate, squared_error))
    print('exp seed:', cfg.initial_seed)
    
    # pdb.set_trace()
    if RUN_SANITY_CHECK:
        #! Let's run some additional sanity check
        #* check to see if bias formula checks out
        v_w = 0
        normalization = 0
        for trajectory in behavior_data:
            discounted_t = 1
            for s,a,sn,r in trajectory:
                v_w += w_star[s]*pi[s,a] /mu[s,a]*r*discounted_t
                normalization += discounted_t
                discounted_t *= gamma
        v_w = v_w / normalization
        
        on_policy_data, frequency, avg_reward = roll_out(env, pi, 4096, horizon)
        # pdb.set_trace()
        empirical_v_pi = np.zeros(n_states)
        empirical_d_pi = np.zeros(n_states)
        empirical_d0 = np.zeros(n_states)
        empirical_r_pi = np.zeros(n_states)
        empirical_frequency = np.zeros(n_states)
        empirical_P = np.zeros((n_states, n_actions, n_states))
        horizon_normalization = (1-gamma**horizon)/(1-gamma)
        num_traj =len(on_policy_data)
        for trajectory in on_policy_data:
            discounted_t = 1
            for s,a,sn,r in trajectory:
                empirical_v_pi[s] += r*discounted_t
                empirical_d_pi[s] += discounted_t
                # empirical_d0[s] += 1-discounted_t
                discounted_t *= gamma
                empirical_r_pi[s] += r
                empirical_frequency[s] += 1
                empirical_P[s,a,sn] += 1
        empirical_v_pi = empirical_v_pi/num_traj
        empirical_d_pi = empirical_d_pi/horizon_normalization/num_traj
        empirical_P = np.nan_to_num(empirical_P / np.sum(empirical_P, axis = -1)[:,:,None])
        # T = np.nan_to_num(T/np.sum(T, axis = -1)[:,:,None])
        empirical_r_pi = np.nan_to_num(empirical_r_pi / empirical_frequency)
        empirical_P_pi = np.einsum('san,sa->sn', empirical_P, pi)
        
        empirical_d_mu = np.zeros(n_states)
        num_traj = len(behavior_data)
        for trajectory in behavior_data:
            discounted_t = 1
            for s,a,sn,r in trajectory:
                empirical_d_mu[s] += discounted_t
                discounted_t *= gamma
        empirical_d_mu = empirical_d_mu/horizon_normalization/num_traj

        empirical_w = np.nan_to_num(empirical_d_pi/empirical_d_mu)
        empirical_loss = L(empirical_w, empirical_v_pi)

        empirical_bellman_original = 0; empirical_bellman_new = 0
        empirical_td_error = 0
        num_traj =len(on_policy_data)
        empirical_r_pi_adjusted = np.dot((np.identity(n_states) - np.linalg.matrix_power(gamma*empirical_P_pi, horizon)),empirical_r_pi)
        for trajectory in on_policy_data:
            discounted_t = 1.0
            for s,a,sn,r in trajectory:
                empirical_bellman_original += discounted_t*(v_pi_s[s] - empirical_r_pi[s] - gamma*np.dot(empirical_P_pi[s,:], v_pi_s))**2
                empirical_bellman_new += discounted_t*(v_pi_s[s] - empirical_r_pi_adjusted[s] - gamma*np.dot(empirical_P_pi[s,:], v_pi_s))**2
                empirical_td_error += discounted_t*(v_pi_s[s] - r - gamma*v_pi_s[sn])**2
                discounted_t *= gamma
        empirical_td_error = empirical_td_error / horizon_normalization / num_traj
        empirical_bellman_original = empirical_bellman_original / horizon_normalization / num_traj
        empirical_bellman_new = empirical_bellman_new / horizon_normalization / num_traj
        # empirical_bellman_original = empirical_v_pi - empirical_r_pi - gamma*np.dot(empirical_P_pi, empirical_v_pi)

        # bellman_original = v_pi_s - R_pi - gamma * np.dot(P_pi, v_pi_s)
        # bellman_new = v_pi_s - np.dot((np.identity(n_states) - np.linalg.matrix_power(gamma*P_pi, horizon)),R_pi) - gamma*np.dot(P_pi, v_pi_s)
        pdb.set_trace()    




    for objective in cfg.objective:
        estimate[objective] =[]; squared_error[objective]=[]
        objective_sn = objective + '-SN'
        estimate[objective_sn] =[]; squared_error[objective_sn]=[] 

    for i in range(cfg.n_experiments):
        training_set = training_data_processed[i]
        fixed_terminal_value = True
        logging=cfg.logging
        mvm = Tabular_State_MVM_Estimator(training_set, cfg, logging = logging, ground_truth = ground_truth_info)
        penalty = cfg.penalty_input
        
        horizon_normalization = (1-gamma**horizon)/(1-gamma)
        # penalty_base = 1/mdp_calculator.horizon_normalization#/cfg.n_trajectories
        penalty_base = 1/horizon_normalization
        mvm.set_random_seed(seed_list[i]) #different random seed per experiment
        mvm.solve_closed_form_bias()
        mvm.generate_random_v_class(cfg.v_class_cardinality)
        mvm.generate_random_w_class(cfg.v_class_cardinality)
        # mvm.bias_check()
        for objective in cfg.objective:
            mvm.set_random_seed(seed_list[i])
            # w_estimator = mvm.optimize_finite_class(objective = objective, td_penalty=penalty*penalty_base)
            # w_estimator = mvm.optimize_discrete(objective = objective, td_penalty=penalty*penalty_base)
            w_estimator = mvm.optimize(objective, td_penalty = 0.1)
            # w_estimator, w_estimator_sn = mvm.optimize_optimistic()
            # w_estimator, w_estimator_sn = mvm.optimize_optimistic_adam(objective = objective, td_penalty=penalty*penalty_base)
            # w_estimator = mvm.optimize_closed_form()
            estimate[objective].append(float(w_estimator))
            # objective_sn = objective + '-SN'
            # estimate[objective_sn].append(float(w_estimator_sn))
            squared_error[objective].append(float(w_estimator-v_pi)**2)
            # squared_error[objective_sn].append(float(w_estimator_sn-v_pi)**2)
        display((estimate, squared_error))

    display((estimate, squared_error))
if __name__ == '__main__':
    main(cfg)

