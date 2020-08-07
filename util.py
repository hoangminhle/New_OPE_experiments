import numpy as np
import pdb
import random

FAKE_TRANSITION = False
def roll_out(env, policy, num_trajectory, truncate_size):
    # np.random.seed(seed)
    # random.seed(seed)
    SASR = []
    total_reward = 0.0
    state_num = env.nS
    frequency = np.zeros(state_num)
    for i_trajectory in range(num_trajectory):
        state = env.reset()
        sasr = []
        for i_t in range(truncate_size-1):
            #env.render()
            p_action = policy[state, :]
            # pdb.set_trace()
            action = np.random.choice(p_action.shape[0], 1, p = p_action)[0]
            next_state, reward, done, _ = env.step(action)

            sasr.append((state, action, next_state, reward))
            frequency[state] += 1
            total_reward += reward
            #print env.state_decoding(state)
            #a = input()

            state = next_state
        #! add one fake transition to the trajectory to signify the end of trajectory
        p_action = policy[state, :]
        action = np.random.choice(p_action.shape[0], 1, p = p_action)[0]
        next_state, reward, done, _ = env.step(action)
        if FAKE_TRANSITION:
            next_state = env.s_absorb
        # next_state = env.s_absorb
        # frequency[state] += 1
        # reward = 0
        # sasr.append((state, action, env.s_absorb, 0))
        sasr.append((state, action, next_state, reward))
        SASR.append(sasr)
    
    return SASR, frequency, total_reward/(num_trajectory * truncate_size)

def on_policy(SASR, gamma):
    total_reward = 0.0
    self_normalizer = 0.0
    for sasr in SASR:
        discounted_t = 1.0
        for state, action, next_state, reward in sasr:
            total_reward += reward * discounted_t
            self_normalizer += discounted_t
            discounted_t *= gamma
    return total_reward / self_normalizer

def exact_calculation(env, policy, horizon, gamma):
    P = env.P_matrix
    n_states = env.nS
    n_actions = env.nA
    d0 = env.isd
    R = env.R_matrix
    horizon_normalization = (1-gamma**horizon)/(1-gamma) if gamma<1 else horizon

    #* calculate true distribution
    P_pi = np.einsum('san, sa-> sn', P,policy)
    dpi_t = np.zeros((n_states,horizon))
    dpi = np.zeros(n_states)
    for h in range(horizon):
        if h == 0:
            dpi_t[:,h] = d0.copy()
        else:
            dpi_t[:,h] = np.dot(P_pi.T,dpi_t[:,h-1])
        dpi += gamma**h*dpi_t[:,h]
    dpi /= horizon_normalization

    Rpi = np.sum(R * policy, axis = -1)
    vpi = np.sum(dpi*Rpi) #* normalized true value

    v_s = np.zeros(n_states)
    
    for s in range(n_states):
        dt = np.zeros(n_states)
        dt[s] = 1.0
        ds = np.zeros(n_states)
        discounted_t = 1.0
        for h in range(horizon):
            ds += dt*discounted_t
            dt = np.dot(P_pi.T, dt)
            discounted_t *=gamma
        v_s[s] += np.sum(ds*Rpi)

    #* after this step, should have
    #* np.sum(d0*v_s) / horizon_normalization == vpi

    #* step-wise value function: calculate it backward
    v_t_s = np.zeros((n_states, horizon))
    for h in range(horizon-1,-1,-1):
        if h == horizon-1:
            v_t_s[:,h] = Rpi.copy()
        else:
            v_t_s[:,h] = Rpi + gamma*np.dot(P_pi, v_t_s[:,h+1])
    #* after this step, we should have v_t_s[:,0] == v_s
    # q_sa = np.zeros(n_states*n_actions)
    q_sa = (R + gamma*np.dot(P,v_t_s[:,1])).reshape(n_states*n_actions)


    #! calculate the correction terms to adjust for finite horizon

    #! sanity check the relationship for finite horizon with discount
    #* d_pi(s') = gamma *sum_{s} P_pi(s'|s) d_pi(s) +1 /h *d_0(s') - 1/h*gamma^H *sum_{s} P_pi(s'|s) d_{pi,H-1}(s) for all s'
    #*  where h is the horizon normalization factor
    assert np.linalg.norm(gamma*np.dot(P_pi.T, dpi) + 1/horizon_normalization *d0 - 1/ horizon_normalization *gamma**horizon *np.dot(P_pi.T, dpi_t[:,-1]) - dpi)<1e-8

    #! make sure 'bellman-like' equation checks out
    #* the usual bellman equation will become
    #* V = (I - (gamma*P_pi)^H))R + gamma* dot(P_pi, V)
    assert np.linalg.norm(v_s - np.dot((np.identity(n_states) - np.linalg.matrix_power(gamma*P_pi, horizon)),Rpi) - gamma*np.dot(P_pi, v_s)) <1e-8

    #! sanity-check the flow equation
    #* in particular, we want to see that
    #* \E_{(s,a,s')\sim d_pi}[ gamma * f(s') - f(s)] + 1/h \E_{s\sim d0}[f(s)] - 1/h*gamma^horizon \E_{s'\sim d_pi,H} [f(s')] = 0 for all function f
    f = np.random.rand(n_states)
    flow = 0
    for s in range(n_states):
        for sn in range(n_states):
            flow += (gamma*f[sn] - f[s])*dpi[s]*P_pi[s,sn]
    flow += 1/horizon_normalization*np.sum(d0*f)
    d_pi_H = np.dot(P_pi.T, dpi_t[:,horizon-1])
    flow -= 1/horizon_normalization*gamma**horizon*np.sum(d_pi_H*f)
    assert abs(flow) < 1e-8
    # pdb.set_trace()

    #! the min-max objective loss function would be a bit different from the average and infinite horizon case
    #* L(w,f) = \E_{(s,a,s')\sim d_mu} [ w(s) (f(s) - gamm*rho(s,a)*f(s'))] +1/h E_{s\sim d0} [f(s)] - 1/h *gamma^horizon \E_{s\sim d_pi,H}[f(s)]
    
    return dpi, dpi_t, v_s, q_sa, P_pi



def sanity_check(env,policy, horizon, gamma):
    P = env.P_matrix
    n_states = env.nS
    n_actions = env.nA
    d0 = env.isd
    R = env.R_matrix

    horizon_normalization = (1-gamma**horizon)/(1-gamma)
    
    #!value function original problem
    P_pi = np.einsum('san, sa-> sn', P,policy)
    dpi_t = np.zeros((n_states,horizon))
    dpi = np.zeros(n_states)
    for h in range(horizon):
        if h == 0:
            dpi_t[:,h] = d0.copy()
        else:
            dpi_t[:,h] = np.dot(P_pi.T,dpi_t[:,h-1])
        dpi += gamma**h*dpi_t[:,h]
    dpi /= horizon_normalization
    
    Rpi = np.sum(R * policy, axis = -1)
    vpi = np.sum(dpi*Rpi) #* normalized true value

    v_s = np.zeros(n_states)
    for s in range(n_states):
        dt = np.zeros(n_states)
        dt[s] = 1.0
        ds = np.zeros(n_states)
        discounted_t = 1.0
        for h in range(horizon):
            ds += dt*discounted_t
            dt = np.dot(P_pi.T, dt)
            discounted_t *=gamma
        v_s[s] += np.sum(ds*Rpi)
    #* after this step, should have
    #* np.sum(d0*v_s) / horizon_normalization == vpi

    #* step-wise value function: calculate it backward
    v_t_s = np.zeros((n_states, horizon))
    for h in range(horizon-1,-1,-1):
        if h == horizon-1:
            v_t_s[:,h] = Rpi.copy()
        else:
            v_t_s[:,h] = Rpi + gamma*np.dot(P_pi, v_t_s[:,h+1])
    #* after this step, we should have v_t_s[:,0] == v_s
    #! problem here, is with non-stationarity of the value fucntion, which will case issues with the notion of bellman equation

    #! modify the transition matrix
    P_mod = P.copy()
    for s in range(n_states-1):
        for a in range(n_actions):
            send_to_absorbing = 0
            for sn in range(n_states):
                P_mod[s,a,sn] = gamma*P[s,a,sn]
                send_to_absorbing += (1-gamma)*P[s,a,sn]
            P_mod[s,a,env.s_absorb] += send_to_absorbing
            # pdb.set_trace()
    for a in range(n_actions):
        P_mod[env.s_absorb,a,:] = d0.copy()

    #* Let's recalculate the distribution and value with this new transition
    P_pi_mod = np.einsum('san, sa-> sn', P_mod,policy)
    dpi_t_mod = np.zeros((n_states,horizon))
    dpi_mod = np.zeros(n_states)
    for h in range(horizon):
        if h == 0:
            dpi_t_mod[:,h] = d0.copy()
        else:
            dpi_t_mod[:,h] = np.dot(P_pi_mod.T,dpi_t_mod[:,h-1])
        dpi_mod += gamma**h*dpi_t_mod[:,h]
    dpi_mod /= horizon_normalization

    vpi_mod = np.sum(dpi_mod*Rpi) #* normalized true value

    v_s_mod = np.zeros(n_states)
    for s in range(n_states):
        dt = np.zeros(n_states)
        dt[s] = 1.0
        ds = np.zeros(n_states)
        discounted_t = 1.0
        for h in range(horizon):
            ds += dt*discounted_t
            dt = np.dot(P_pi_mod.T, dt)
            discounted_t *=gamma
        v_s_mod[s] += np.sum(ds*Rpi)
    #* after this step, should have
    #* np.sum(d0*v_s_mod) / horizon_normalization == vpi_mod

    v_t_s_mod_inf = np.dot(np.linalg.inv(np.identity(n_states) - gamma*P_pi_mod), Rpi)
    #* step-wise value function: calculate it backward
    v_t_s_mod = np.zeros((n_states, horizon))
    for h in range(horizon-1,-1,-1):
        if h == horizon-1:
            v_t_s_mod[:,h] = Rpi.copy()
        else:
            v_t_s_mod[:,h] = Rpi + np.dot(P_pi_mod, v_t_s_mod[:,h+1])
    #* after this step, we should have v_t_s[:,0] == v_s

    pdb.set_trace()


def exact_value_finite_horizon(env,policy, horizon, gamma):
    P = env.P_matrix
    n_states = env.nS
    n_actions = env.nA
    d0 = env.isd.reshape((env.nS,1))

    #! what happens if we modify P
    P_mod = P.copy()
    for s in range(n_states):
        for a in range(n_actions):
            for sn in range(n_states):
                P_mod[s,a,sn] = gamma*P[s,a,sn]
                P_mod[s,a,env.s_absorb] += (1-gamma)*P[s,a,sn]
            P_mod[env.s_absorb,a,:] = d0[:,0].copy()

    P = P_mod.copy()

    R = env.R_matrix
    P_pi = np.einsum('san, sa-> sn', P,policy)
    horizon_normalization = (1-gamma**horizon)/(1-gamma)
    discounted_t = 1.0
    # horizon_normalization = 0.0
    

    dt = d0.copy()
    dpi = np.zeros((env.nS,1))
    for h in range(horizon):
        dpi += dt* discounted_t
        dt = np.dot(P_pi.T,dt)
        # horizon_normalization += discounted_t
        discounted_t *= gamma
    dpi /= horizon_normalization
    Rpi = np.sum(R * policy, axis = -1)
    v_pi_s = dpi.reshape(-1) * Rpi
    v_pi = np.sum(v_pi_s)
    
    #! calculate exact value function
    v_s = np.zeros(n_states)
    for s in range(n_states):
        dt = np.zeros(n_states)
        dt[s] = 1.0
        ds = np.zeros(n_states)
        discounted_t = 1.0
        for h in range(horizon):
            ds += dt*discounted_t
            dt = np.dot(P_pi.T, dt)
            discounted_t *=gamma
        v_s[s] += np.sum(ds*Rpi)
    # pdb.set_trace()


    #! sanity check the relationship for finite horizon with discount
    #* d_pi(s') = gamma *sum_{s} P_pi(s'|s) d_pi(s) +1 /h *d_0(s') - 1/h*gamma^H *sum_{s} P_pi(s'|s) d_{pi,H-1}(s) for all s'
    #*  where h is the horizon normalization factor

    d_pi_t = np.zeros((env.nS, horizon))
    d_pi_t[:,0:1] = d0.copy()
    for h in range(horizon-1):
        d_pi_t[:,h+1:h+2] = np.dot(P_pi.T,d_pi_t[:,h:h+1])
    d_pi = np.zeros(env.nS)
    horizon_normalization = 0
    for h in range(horizon):
        d_pi += gamma**h*d_pi_t[:,h]
        horizon_normalization += gamma**h
    d_pi /= horizon_normalization
    #* sanity check to deduce adjusted d0
    #* in this case adjusted_d0(s') = 1/h d_0(s') -1/h*gamma^H \sum_{s} P_pi(s' |s) d_{pi,H-1}(s)
    #* this formulation of adjusted_d0 can contain negative number
    n_states = env.nS
    lhs = np.zeros(n_states)
    for sn in range(n_states):
        for s in range(n_states):
            lhs[sn] += gamma*P_pi[s,sn]*dpi[s,0]
        lhs[sn] -=  dpi[sn,0]
    rhs = 1/horizon_normalization*d0[:,0]
    residual = 1/horizon_normalization*gamma**horizon*np.dot(P_pi.T, d_pi_t[:,-1])
    adjusted_d0 = d0[:,0]/horizon_normalization - 1/horizon_normalization*gamma**horizon*np.dot(P_pi.T, d_pi_t[:,-1])

    # check bellman equation
    #! V_pi(s) - gamma *E_{s',a|s-sim d_pi}[V_pi(s')] = \E_{a|s\sim pi}[r(s,a)]
    lhs = np.zeros(n_states)
    rhs = np.zeros(n_states)
    for s in range(n_states):
        rhs[s] = np.dot(R[s,:], policy[s,:])
        # pdb.set_trace()
        lhs[s] = v_s[s] - gamma*np.dot(P_pi[s,:], v_s)
        # lhs[s] = v_s[s] - (1-1/horizon_normalization)*np.dot(P_pi[s,:], v_s)
    pdb.set_trace()
    return dpi, v_pi_s, v_pi


