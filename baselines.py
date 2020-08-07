import numpy as np
import pdb
import quadprog

NORMALIZED_ESTIMATE = False

def quadratic_solver(n, M, regularizer):
    qp_G = np.matmul(M, M.T)
    qp_G += regularizer * np.eye(n)
    
    qp_a = np.zeros(n, dtype = np.float64)

    qp_C = np.zeros((n,n+1), dtype = np.float64)
    for i in range(n):
        qp_C[i,0] = 1.0
        qp_C[i,i+1] = 1.0
    qp_b = np.zeros(n+1, dtype = np.float64)
    qp_b[0] = 1.0
    meq = 1
    res = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)
    w = res[0]
    return w

class Density_Ratio_discounted(object):
    def __init__(self, num_state, gamma):
        self.num_state = num_state
        self.Ghat = np.zeros([num_state, num_state], dtype = np.float64)
        self.Nstate = np.zeros([num_state, 1], dtype = np.float64)
        self.initial_b = np.zeros([num_state], dtype = np.float64)
        self.gamma = gamma

    def reset(self):
        num_state = self.num_state
        self.Ghat = np.zeros([num_state, num_state], dtype = np.float64)
        self.Nstate = np.zeros([num_state, 1], dtype = np.float64)

    def feed_data(self, cur, next, initial, policy_ratio, discounted_t):
        if cur == -1:
            self.Ghat[next, next] -= discounted_t
        else:
            self.Ghat[cur, next] += policy_ratio * discounted_t
            self.Ghat[cur, initial] += (1-self.gamma)/self.gamma * discounted_t
            self.Ghat[next, next] -= discounted_t
            self.Nstate[cur] += discounted_t

    def density_ratio_estimate(self, regularizer = 0.001):
        Frequency = self.Nstate.reshape(-1)
        tvalid = np.where(Frequency >= 1e-20)
        G = np.zeros_like(self.Ghat)
        Frequency = Frequency/np.sum(Frequency)
        G[tvalid] = self.Ghat[tvalid]/(Frequency[:,None])[tvalid]		
        n = self.num_state
        x = quadratic_solver(n, G/50.0, regularizer)
        w = np.zeros(self.num_state)
        w[tvalid] = x[tvalid]/Frequency[tvalid]
        return x, w

def train_density_ratio(SASR, policy0, policy1, den_discrete, gamma):
    for sasr in SASR:
        discounted_t = 1.0
        # pdb.set_trace()
        # initial_state = sasr[0,0]
        initial_state = sasr[0][0]
        for state, action, next_state, reward in sasr:
            discounted_t *= gamma
            policy_ratio = policy1[state, action]/policy0[state, action]
            den_discrete.feed_data(state, next_state, initial_state, policy_ratio, discounted_t)
        den_discrete.feed_data(-1, initial_state, initial_state, 1, 1-discounted_t)
        
    x, w = den_discrete.density_ratio_estimate()
    return x, w

def lihong_infinite_horizon(n_state, SASR, pi0, pi1, gamma):
    den_discrete = Density_Ratio_discounted(n_state, gamma)
    x, w = train_density_ratio(SASR, pi0, pi1, den_discrete, gamma)
    x = x.reshape(-1)
    w = w.reshape(-1)
    est_DENR = off_policy_evaluation_density_ratio(SASR, pi0, pi1, w, gamma)
    return est_DENR

def off_policy_evaluation_density_ratio(SASR, policy0, policy1, density_ratio, gamma):
    total_reward = 0.0
    self_normalizer = 0.0
    self_normalizer_standard = 0.0
    
    for sasr in SASR:
        discounted_t = 1.0
        horizon_normalization = 0.0
        for state, action, next_state, reward in sasr:
            policy_ratio = policy1[state, action]/policy0[state, action]
            total_reward += density_ratio[state] * policy_ratio * reward * discounted_t
            self_normalizer += density_ratio[state] * policy_ratio * discounted_t
            self_normalizer_standard += discounted_t
            horizon_normalization += discounted_t
            discounted_t *= gamma
    # pdb.set_trace()
    if not NORMALIZED_ESTIMATE:
        return total_reward / self_normalizer *horizon_normalization, total_reward / self_normalizer_standard * horizon_normalization    
    else:
        return total_reward / self_normalizer, total_reward / self_normalizer_standard

def importance_sampling_estimator(SASR, policy0, policy1, gamma):
    mean_est_reward = 0.0
    for sasr in SASR:
        log_trajectory_ratio = 0.0
        total_reward = 0.0
        discounted_t = 1.0
        self_normalizer = 0.0
        for state, action, next_state, reward in sasr:
            log_trajectory_ratio += np.log(policy1[state, action]) - np.log(policy0[state, action])
            total_reward += reward * discounted_t
            self_normalizer += discounted_t
            discounted_t *= gamma
        if not NORMALIZED_ESTIMATE:
            self_normalizer = 1.0 # whether or not normalize the returned estimate
        avr_reward = total_reward / self_normalizer
        mean_est_reward += avr_reward * np.exp(log_trajectory_ratio)
    mean_est_reward /= len(SASR)
    return mean_est_reward

def importance_sampling_estimator_stepwise(SASR, policy0, policy1, gamma):
    mean_est_reward = 0.0
    for sasr in SASR:
        step_log_pr = 0.0
        est_reward = 0.0
        discounted_t = 1.0
        self_normalizer = 0.0
        for state, action, next_state, reward in sasr:
            step_log_pr += np.log(policy1[state, action]) - np.log(policy0[state, action])
            est_reward += np.exp(step_log_pr)*reward*discounted_t
            self_normalizer += discounted_t
            discounted_t *= gamma
        if not NORMALIZED_ESTIMATE:
            self_normalizer = 1.0 # whether or not normalize the returned estimate
        est_reward /= self_normalizer
        mean_est_reward += est_reward
    mean_est_reward /= len(SASR)
    return mean_est_reward

def weighted_importance_sampling_estimator(SASR, policy0, policy1, gamma):
    total_rho = 0.0
    est_reward = 0.0
    for sasr in SASR:
        total_reward = 0.0
        log_trajectory_ratio = 0.0
        discounted_t = 1.0
        self_normalizer = 0.0
        for state, action, next_state, reward in sasr:
            log_trajectory_ratio += np.log(policy1[state, action]) - np.log(policy0[state, action])
            total_reward += reward * discounted_t
            self_normalizer += discounted_t
            discounted_t *= gamma
        if not NORMALIZED_ESTIMATE:
            self_normalizer = 1.0 # whether or not normalize the returned estimate
        avr_reward = total_reward / self_normalizer
        trajectory_ratio = np.exp(log_trajectory_ratio)
        total_rho += trajectory_ratio
        est_reward += trajectory_ratio * avr_reward

    avr_rho = total_rho / len(SASR)
    return est_reward / avr_rho/ len(SASR)

def weighted_importance_sampling_estimator_stepwise(SASR, policy0, policy1, gamma):
    Log_policy_ratio = []
    REW = []
    for sasr in SASR:
        log_policy_ratio = []
        rew = []
        discounted_t = 1.0
        self_normalizer = 0.0
        for state, action, next_state, reward in sasr:
            log_pr = np.log(policy1[state, action]) - np.log(policy0[state, action])
            if log_policy_ratio:
                log_policy_ratio.append(log_pr + log_policy_ratio[-1])
            else:
                log_policy_ratio.append(log_pr)
            rew.append(reward * discounted_t)
            self_normalizer += discounted_t
            discounted_t *= gamma
        Log_policy_ratio.append(log_policy_ratio)
        REW.append(rew)
    if not NORMALIZED_ESTIMATE:
        self_normalizer = 1.0 # whether or not normalize the returned estimate
    est_reward = 0.0
    rho = np.exp(Log_policy_ratio)
    #print 'rho shape = {}'.format(rho.shape)
    REW = np.array(REW)
    for i in range(REW.shape[0]):
        est_reward += np.sum(rho[i]/np.mean(rho, axis = 0) * REW[i])/self_normalizer
    return est_reward/REW.shape[0]

def model_based(n_state, n_action, SASR, pi, gamma):
    T = np.zeros([n_state, n_action, n_state], dtype = np.float32)
    R = np.zeros([n_state, n_action], dtype = np.float32)
    R_count = np.zeros([n_state, n_action], dtype = np.int32)
    for sasr in SASR:
        for state, action, next_state, reward in sasr:
            T[state, action, next_state] += 1
            R[state, action] += reward
            R_count[state, action] += 1
    d0 = np.zeros([n_state, 1], dtype = np.float32)
    # pdb.set_trace()
    for trajectory in SASR:
        state = trajectory[0][0]
        d0[state,0] += 1.0
    # for state in SASR[:,0,0].flat:
    #     d0[state, 0] += 1.0
    t = np.where(R_count > 0)
    t0 = np.where(R_count == 0)
    R[t] = R[t]/R_count[t]
    # R[t0] = np.mean(R[t]) #! why should we do this smoothing? skip the smoothing for now
    # T = T + 1e-9	# smoothing #*smoothing step from Lihong's paper implementation, is it a good idea at all?
    # pdb.set_trace()
    T = np.nan_to_num(T/np.sum(T, axis = -1)[:,:,None])
    #! terminal state should self-loop?
    T[-1,:,-1] = 1.0
    Tpi = np.zeros([n_state, n_state])
    for state in range(n_state):
        for next_state in range(n_state):
            for action in range(n_action):
                Tpi[state, next_state] += T[state, action, next_state] * pi[state, action]
    dt = d0/np.sum(d0)
    dpi = np.zeros([n_state, 1], dtype = np.float32)
    # pdb.set_trace()
    # truncate_size = SASR.shape[1] 
    truncate_size = len(SASR[0])
    discounted_t = 1.0
    self_normalizer = 0.0
    for i in range(truncate_size):
        dpi += dt * discounted_t
        # if i < 50: #! wth is this? #! this is a weird artifact from Lihong's paper experiment
        #     dt = np.dot(Tpi.T,dt)
        dt = np.dot(Tpi.T,dt) #* let's just remove the weird 50 limit from Lihong paper's implementation
        self_normalizer += discounted_t
        discounted_t *= gamma
    dpi /= self_normalizer
    Rpi = np.sum(R * pi, axis = -1)
    # pdb.set_trace()
    if not NORMALIZED_ESTIMATE:
        return np.sum(dpi.reshape(-1) * Rpi) * self_normalizer
    else:
        return np.sum(dpi.reshape(-1) * Rpi)