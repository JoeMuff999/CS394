# from cmath import nan
from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using ordinary importance
    # sampling (Hint: Sutton Book p. 109, every-visit implementation is fine)
    #####################
    Q = initQ
    T = np.zeros((env_spec.nS, env_spec.nA))

    # T = [0 for i in range(env_spec.nS)]
    for trajectory_idx in range(len(trajs)):
    # for trajectory_idx in range(100):
        trajectory = trajs[trajectory_idx]
        p = np.ones((env_spec.nS, env_spec.nA))
        G = 0
        length = len(trajectory)
        for t in range(length-1, -1,-1):
            s, a, r, s_p = trajectory[t]
            if p[s][a] == 0:
                break
            # s, a, r, s_p = state_tuple
            G = env_spec.gamma * G + r
            T[s][a] += 1
            p[s][a] *= (pi.action_prob(s, a)/bpi.action_prob(s,a))
            # assert T[s][a] != 0
            Q[s][a] = Q[s][a] + (p[s][a]/T[s][a])*(G - Q[s][a])
                
        # print(T)
        # print(p)
    return Q

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """
    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using weighted importance
    # sampling (Hint: Sutton Book p. 110, every-visit implementation is fine)
    #####################
    Q = initQ
    C = np.zeros((env_spec.nS, env_spec.nA))
    for trajectory in trajs:
        G = 0
        W = 1
        T = len(trajectory)
        for t in range(T-1, -1,-1):
            if W == 0:
                break
            s, a, r, s_p = trajectory[t]
            G = env_spec.gamma * G + r
            C[s][a] += W
            Q[s][a] = Q[s][a] + (W/C[s][a])*(G - Q[s][a])
            W *= (pi.action_prob(s, a)/bpi.action_prob(s,a))

    return Q
