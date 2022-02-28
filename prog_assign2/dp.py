from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Value Prediction Algorithm (Hint: Sutton Book p.75)
    #####################
    # print(initV)
    V = np.zeros(env.spec.nS)
    Q = np.zeros((env.spec.nS,env.spec.nA))
    

    v_delta = np.inf
    q_delta = np.inf
    while v_delta >= theta and q_delta >= theta:
        v_delta = 0
        q_delta = 0
        for state in range(len(V)):
            init_v = V[state]
            new_V = 0
            for action in range(env.spec.nA):
                init_q = Q[state][action]
                # q_sum_prob = 0
                sum_prob = 0
                # go over all possible successor states
                for state_p in range(env.spec.nS):
                    pr = env.TD[state][action][state_p]
                    reward = env.R[state][action][state_p]
                    value_succ = V[state_p]
                    # q_sum_prob += pr * (reward + env.spec.gamma * value_succ)
                    sum_prob += pr * (reward + env.spec.gamma * value_succ)
            
                new_V += pi.action_prob(state, action) * sum_prob
                Q[state][action] = sum_prob
                if(state == 0 and action == 0):
                    print(pi.action_prob(state,action))
                    print(sum_prob)
                q_delta = max(q_delta, abs(init_q - Q[state][action]))
            V[state] = new_V
            v_delta = max(v_delta, abs(init_v - V[state]))
        print(Q)

    # env.step()
    # pass
    return V, Q

# # returns the max Q value for a given state
# # checks the Q value for each action possible from this state
# # returns the max of these values
# def max_q_value(env : EnvWithModel, Q, pi:Policy,  state):
#     max_q_val = -9999
#     for action in range(env.spec.nA):
#         max_q_val = max(max_q_val, Q[state][action]) 
#     return max_q_val

class OptimalPolicy(Policy):
    def __init__(self, env: EnvWithModel) -> None:
        super().__init__()
        self.policy = np.zeros(env.spec.nS)
    
    def set_optimal_action(self, state, action):
        self.policy[state] = action

    def action_prob(self,state:int,action:int) -> float:
        """
        input:
            state, action
        return:
            \pi(a|s)
        """
        if self.policy[state] == action:
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
        return self.policy[state]

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################
    # pass
    V = np.zeros(env.spec.nS)
    delta = np.inf
    # value evaluation
    while delta >= theta:
        delta = 0
        for state in range(env.spec.nS):
            init_v = V[state]
            max_v = -np.inf
            for action in range(env.spec.nA):
                sum_succ_values = 0
                for successor in range(env.spec.nS):
                    # print(env.TD)
                    # L = env.TD[state][action][successor]
                    # X = env.R[state][action][successor]
                    sum_succ_values += env.TD[state][action][successor] * (env.R[state][action][successor] + env.spec.gamma * V[successor])
                max_v = max(max_v, sum_succ_values)
            V[state] = max_v
            delta = max(delta, abs(init_v - V[state]))
    # policy improvement
    # for each state in the policy, set its action to be the highest valued action
    pi = OptimalPolicy(env)
    for state in range (env.spec.nS):
        max_action_val = 0
        max_action_index = 0
        #for all possible actions (from this state since we check the dynamics)
        # get the action with the highest total value (probability of transition occuring * (reward of transition + value of successor))
        for action in range(env.spec.nA):
            action_val = 0
            for successor in range(env.spec.nS):
                action_val += env.TD[state][action][successor] * (env.R[state][action][successor] + env.spec.gamma * V[successor])
            if max_action_val < action_val:
                max_action_index = action
                max_action_val = action_val
        
        pi.set_optimal_action(state, max_action_index)
                
    return V, pi
