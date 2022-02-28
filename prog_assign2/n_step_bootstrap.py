from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################
    V = initV
    for traj in trajs:
        G = 0
        T = len(traj)
        for t, state_tuple in enumerate(traj):
            s, a, r, s_p = state_tuple
            tau = t - n + 1
            G += pow(env_spec.gamma, t%n) * r
            if tau >=0:
                if tau + n < T:
                    G = G + pow(env_spec.gamma, n) * V[s_p]
                V[traj[tau][0]] += alpha * (G - V[traj[tau][0]]) # V(S_tau)
                G = 0

    return V

class OptimalPolicy(Policy):
    def __init__(self, env: EnvSpec, Q ) -> None:
        super().__init__()
        # self.policy = np.full(env.nS, -1) 
        self.policy = np.zeros((env.nS, env.nA))
        # self.policy_value = np.full(env.nS, 0)
    
    def update_policy_value(self, state, action, value):
        # self.policy[state] = action
        self.policy[state][action] = value
        # self.policy_value[state] = value # store for easier use later

    # def has_policy_for_state(self, state):
    #     return self.policy[state] != -1

    def get_policy_value(self, state):
        return max(self.policy[state])

    def action_prob(self,state:int,action:int) -> float:
        """
        input:
            state, action
        return:
            \pi(a|s)
        """
        val = self.get_policy_value(state)
        
        if self.policy[state][action] == val:
            return 1.0
        else:
            return 0.0

    def action(self,state:int) -> int:
        """
        input:
            state
        return:
            action
        """
        return np.argmax(self.policy[state])

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################
    Q = initQ
    pi = OptimalPolicy(env_spec, Q)
    idx = 0
    for traj in trajs:
        idx += 1

        T = len(traj)
        G = 0
        rho = 1
        for t, state_tuple in enumerate(traj):
            s, a, r, s_p = state_tuple
            tau = t - n + 1
            G += pow(env_spec.gamma, t%n) * r
            if t <= T-1 and t%n != 0:
                # print(pi.action_prob(s,a))
                rho *= pi.action_prob(s, a)/bpi.action_prob(s, a)
            # assert Q[0][0] == 0.0, " " + str(Q) + "\n" + str(t) + "\n" + str(idx)
            if tau >= 0:
                # G = traj[tau]
                # G = 0
                # rho = 1
                # for i in range(tau+1, min(tau + n, T-1)+1):
                #     G += pow(env_spec.gamma, i-tau-1) * traj[i-1][2]
                #     assert traj[i-1][2] == 0
                #     if i <= T-1:
                #         rho *= pi.action_prob(traj[i][0], traj[i][1])/bpi.action_prob(traj[i][0], traj[i][1])
                if tau + n < T: 
                    next_state = traj[t+1][0] # because tau = (t-n+1), so tau + n = t-n+n + 1 = t+1
                    next_action = traj[t+1][1]
                    G = G + pow(env_spec.gamma, n) * Q[next_state][next_action]
                tau_state = traj[tau][0]
                tau_action = traj[tau][1]
                Q[tau_state][tau_action] += alpha*rho*(G - Q[tau_state][tau_action])
                # print(Q)
                # now, adjust policy
                # if pi.get_policy_value
                pi.update_policy_value(tau_state, tau_action, Q[tau_state][tau_action])
                
                G = 0
                rho = 1



    return Q, pi
