import numpy as np
import torch
import pdb

from util import roll_out
from util import *
from baselines import *
from tabular_cf_estimator import Tabular_State_MVM_Estimator
# from sa_estimator import Tabular_State_Action_MVM_Estimator
from attrdict import AttrDict
from config import Config
import random
from random import randint
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

dtype = torch.double
device = 'cpu'
# initial_seed = randint(0,10000)
initial_seed = 1234
config = {
    'env': 'grid_world', #'taxi', #'grid_world', #'grid_world', #'taxi', #'grid_world',
    'horizon': 32,
    'n_trajectories': [128, 256, 512, 1024],#[128,256,512],
    'gamma': 0.99,
    'n_experiments': 25,
    'initial_seed': initial_seed,
    'lr_w': 0.1,
    'objective': ['TD-ball center',  'bias', 'bias_td', 'bias_td_var'],#['MWL', 'LSTDQ', 'TD-ball center',  'bias', 'bias_td', 'bias_td_var'],
    'n_iterations': 101,
    'lr_decay': True,
    'beta': (0,0.9),
    'eps': 1e-8,
    'estimate_d0': False,
    'estimate_rho': False,
    'eps_matrix_invert': 1e-8,
    'reg_true_rho': False,
    'limited_w_rep': True,
    'limited_v_rep': False,
    'print_progress': False,
    # 'pi_temp': 0.1,
    # 'mu_temp': 0.5,
}

cfg = Config(config)

def prepare_behavior_data(behavior_data):
    behavior_data_squeezed = np.squeeze(behavior_data)
    s = torch.tensor(behavior_data_squeezed[:,:,0], dtype = torch.long)
    a = torch.tensor(behavior_data_squeezed[:,:,1], dtype = torch.long)
    sn = torch.tensor(behavior_data_squeezed[:,:,2], dtype = torch.long)
    r = torch.tensor(behavior_data_squeezed[:,:,3], dtype = dtype)
    return (s,a,sn,r)

def display(result, n_trajectories):
    estimate, mse = result
    true_pi = estimate['True pi'][0]
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
    stochasticity = 'stochastic transition' if cfg.stochastic_env == True else 'deterministic transition'
    print (f"Ordered Results over {cfg.n_experiments} Experiments, horizon {cfg.horizon}, trajectories {n_trajectories}, {stochasticity} (domain {cfg.env}):  \n")
    for key in sorted_keys[:,0]:
        if len(estimate[key])>0:
            value = sum(estimate[key])/len(estimate[key])
            error = sum(mse[key])/len(mse[key])
            bias = abs(value-true_pi)
            std = np.std(np.array(estimate[key]))
        else:
            value = np.nan
            error = np.nan
        label = ' '*(longest-len(key)) + key
        print("Method: {} ---- Avg Estimate: {:10.4f}. MSE: {:10.4f}. Bias: {:10.4f}. Std: {:10.4f}".format(label, value, error, bias, std))

def main(cfg):
    initial_seed = cfg.initial_seed
    random.seed(initial_seed)
    np.random.seed(initial_seed)
    gamma = cfg.gamma
    # n_trajectories_list = cfg.n_trajectories
    # for n_trajectories in n_trajectories_list:
    # n_trajectories = cfg.n_trajectories
    horizon = cfg.horizon
    horizon_normalization = (1-gamma**horizon)/(1-gamma) if gamma<1 else horizon
    seed_list = [initial_seed + np.random.randint(0,10000)*i for i in range(cfg.n_experiments)] # generate a list of random seeds
    if cfg.env == 'grid_world':
        from gridworld import GridWorldEnv
        env = GridWorldEnv()
    elif cfg.env == 'taxi':
        from taxi import TaxiEnv
        env = TaxiEnv()

    n_states = env.nS; n_actions = env.nA; P = env.P_matrix; R = env.R_matrix.copy(); d0 = env.isd
    q_star_original = env.value_iteration()
    pi = env.extract_policy(q_star_original, temperature=0.3)
    mu = env.extract_policy(q_star_original, temperature=0.1)
    # pi = env.extract_policy(q_star_original, temperature=0.1)
    # mu = env.extract_policy(q_star_original, temperature=0.3)
    # pi = env.extract_policy(q_star_original, temperature=0.3)
    # mu = env.extract_policy(q_star_original, temperature=0.15)
    # mu = pi.copy(); mu[:,1] = pi[:,2].copy(); mu[:,2] = pi[:,1].copy()
    #* 4 swapped cyclic 
    # mu = pi.copy(); mu[:,0] = pi[:,1].copy(); mu[:,1] = pi[:,2].copy(); mu[:,2] = pi[:,3].copy(); mu[:,3] = pi[:,0].copy()
    #* D swapped with R, L swapped with U
    # mu = pi.copy(); mu[:,0] = pi[:,3].copy(); mu[:,1] = pi[:,2].copy(); mu[:,2] = pi[:,1].copy(); mu[:,3] = pi[:,0].copy()
    # mu = pi.copy(); mu[:,0] = pi[:,1].copy(); mu[:,1] = pi[:,2].copy(); mu[:,2] = pi[:,3].copy(); mu[:,3] = pi[:,4].copy();mu[:,4] = pi[:,5].copy();mu[:,5] = pi[:,0].copy() 

    dpi, dpi_t, v_pi_s, q_pi_sa, P_pi = exact_calculation(env,pi, cfg.horizon, cfg.gamma)
    dmu, dmu_t, vmu_s, qmu_sa, P_mu = exact_calculation(env,mu, cfg.horizon, cfg.gamma)
    w_star = np.nan_to_num(dpi/dmu)
    v_pi = np.sum(d0*v_pi_s); v_mu = np.sum(d0*vmu_s)
    
    dpi_H = np.dot(P_pi.T, dpi_t[:,horizon-1])
    dmu_H = np.dot(P_mu.T, dmu_t[:,horizon-1])

    ground_truth_info = AttrDict({}) 
    ground_truth_info.update({'d_pi': torch.tensor(dpi, dtype=dtype), 'd_mu': torch.tensor(dmu, dtype=dtype), 'v_pi': torch.tensor(v_pi_s, dtype=dtype),'q_pi': torch.tensor(q_pi_sa, dtype=dtype), 'v_star': v_pi})
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
    estimate.update({'True pi': [float(v_pi)]}); squared_error.update({'True pi':[0]}); 
    estimate.update({'True mu': [float(v_mu)]}); squared_error.update({'True mu':[float(v_mu-v_pi)**2]}); 

    results = {}; results['trajectories'] = [];
    results['IS'] = []; results['IH'] = []; results['MB'] = []; results['WIS'] = []; results['STEP WIS'] = [];results['STEP IS'] = [];results['True mu'] = [];
    for objective in cfg.objective: results[objective]=[]

    n_trajectories_list = cfg.n_trajectories
    for n_trajectories in n_trajectories_list:
        print('------------------------')
        #* Generate multiple sets of behavior data from mu
        training_data = []
        training_data_processed = []
        for _ in range(cfg.n_experiments):
            # print('Experiment:',_)
            # print('------------------------')
            np.random.seed(seed_list[_])
            env.seed(seed_list[_])
            behavior_data, _, _ = roll_out(env, mu, n_trajectories, horizon)
            behavior_data_processed = prepare_behavior_data(behavior_data)
            training_data.append(behavior_data)
            training_data_processed.append(behavior_data_processed)
        estimate['IS'], estimate['STEP IS'], estimate['WIS'], estimate['STEP WIS'], estimate['Mu hat'] = [], [], [], [], []
        squared_error['IS'] = []; squared_error['STEP IS'] = []; squared_error['WIS'] = []; squared_error['STEP WIS'] = []; squared_error['Mu hat'] = []
        estimate['IH_SN'] = []; squared_error['IH_SN'] = []; 
        estimate['IH_no_SN'] = []; squared_error['IH_no_SN'] = []; 
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
                
        # display((estimate, squared_error))
        # print('exp seed:', cfg.initial_seed)
        # pdb.set_trace()
        results['trajectories'].append(np.log2(n_trajectories))
        results['IH'].append(np.log2(sum(squared_error['IH_SN'])/len(squared_error['IH_SN']) /v_pi**2))
        results['MB'].append(np.log2(sum(squared_error['MB'])/len(squared_error['IH_SN']) /v_pi**2))
        results['IS'].append(np.log2(sum(squared_error['IS'])/len(squared_error['IS']) /v_pi**2))
        results['WIS'].append(np.log2(sum(squared_error['WIS'])/len(squared_error['WIS']) /v_pi**2))
        results['STEP WIS'].append(np.log2(sum(squared_error['STEP WIS'])/len(squared_error['STEP WIS']) /v_pi**2))
        results['STEP IS'].append(np.log2(sum(squared_error['STEP IS'])/len(squared_error['STEP IS']) /v_pi**2))
        results['True mu'].append(np.log2(sum(squared_error['True mu'])/len(squared_error['True mu']) /v_pi**2))

        for objective in cfg.objective:
            estimate[objective] =[]; squared_error[objective]=[]

        # for i in range(cfg.n_experiments):
        #     training_set = training_data_processed[i]
        #     mvm = Tabular_State_MVM_Estimator(training_set, cfg, ground_truth = ground_truth_info)
        #     for objective in cfg.objective:
        #         mvm.set_random_seed(seed_list[i])
        #         w_estimator = mvm.optimize(objective)
        #         estimate[objective].append(float(w_estimator))
        #         squared_error[objective].append(float(w_estimator-v_pi)**2)
        #     display((estimate, squared_error))

        for i in range(cfg.n_experiments):
            training_set = training_data_processed[i]
            mvm = Tabular_State_MVM_Estimator(training_set, cfg, ground_truth = ground_truth_info)
            for objective in cfg.objective:
                mvm.set_random_seed(seed_list[i])
                w_estimator = mvm.optimize(objective)
                estimate[objective].append(float(w_estimator))
                squared_error[objective].append(float(w_estimator-v_pi)**2)
        # display((estimate, squared_error))
        for objective in cfg.objective:
            results[objective].append(np.log2(sum(squared_error[objective])/len(squared_error[objective]) /v_pi**2))    
        display((estimate, squared_error), n_trajectories)
        print('\n')
        print('End of one set of experiments')
        
    
    # pdb.set_trace()
    df = pd.DataFrame(results)
    # plt.plot(results['trajectories'], results['IH'],marker='o', markerfacecolor='blue', markersize=12, color='blue', linewidth=4)
    # plt.plot(results['trajectories'], results['MB'],marker='o', markerfacecolor='red', markersize=12, color='red', linewidth=4)
    # plt.plot(results['trajectories'], results['STEP WIS'],marker='o', markerfacecolor='aqua', markersize=12, color='aqua', linewidth=4)
    # plt.plot(results['trajectories'], results['STEP IS'],marker='o', markerfacecolor='orange', markersize=12, color='orange', linewidth=4)
    markersize = 8
    linewidth = 4
    plt.plot('trajectories', 'STEP WIS', data=df, marker='o', markerfacecolor='slategrey', markersize=markersize, color='slategrey', linewidth=linewidth)
    plt.plot('trajectories', 'STEP IS', data=df, marker='o', markerfacecolor='rosybrown', markersize=markersize, color='rosybrown', linewidth=linewidth)
    plt.plot('trajectories', 'True mu', data=df, marker='o', markerfacecolor='black', markersize=markersize, color='black', linewidth=linewidth)
    # plt.plot('trajectories', 'MWL', data=df, marker='o', markerfacecolor='green', markersize=markersize, color='green', linewidth=linewidth)
    # plt.plot('trajectories', 'LSTDQ', data=df, marker='o', markerfacecolor='olive', markersize=markersize, color='olive', linewidth=linewidth)
    plt.plot('trajectories', 'IH', data=df, marker='o', markerfacecolor='purple', markersize=markersize, color='purple', linewidth=linewidth)
    plt.plot('trajectories', 'MB', data=df, marker='o', markerfacecolor='gold', markersize=markersize, color='gold', linewidth=linewidth)
    plt.plot('trajectories', 'TD-ball center', data=df, marker='p', markerfacecolor='cadetblue', markersize=markersize, color='cadetblue', linewidth=linewidth)
    plt.plot('trajectories', 'bias', data=df, marker='s', markerfacecolor='skyblue', markersize=markersize, color='skyblue', linewidth=linewidth)
    plt.plot('trajectories', 'bias_td', data=df, marker='s', markerfacecolor='darkred', markersize=markersize, color='darkred', linewidth=linewidth)
    plt.plot('trajectories', 'bias_td_var', data=df, marker='s', markerfacecolor='orange', markersize=markersize, color='orange', linewidth=linewidth)
    # plt.xticks(cfg.n_trajectories)
    plt.xticks(results['trajectories'])
    plt.xlabel('log number of trajectories'); plt.ylabel('log MSE')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol = 3, prop={'size': 8})
    plt.savefig('pi_03_mu_01_grid_misspecified_w.png')
    pdb.set_trace()    
if __name__ == '__main__':
    main(cfg)

