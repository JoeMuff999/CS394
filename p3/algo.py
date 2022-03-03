import numpy as np
import gym
from policy import Policy

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()
testing_states = np.array([[-.5, 0], [-0.2694817,  0.014904 ], [-1.2,  0. ], [-0.51103601,  0.06101282], [ 0.48690072,  0.04923175]])

#page 231 of pdf, 209 of textbook
def semi_gradient_n_step_td(
    env:gym.Env , #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    #TODO: implement this function
    # loop for each episode
    for _ in range(num_episode):
        print([V(i) for i in testing_states])
        T = np.inf
        print("episode num " + str(_))
        env.reset() # reset environment at the start of the episode
        s = env.observation_space.sample()
        # reward_store = [0 for i in range(n)]
        reward_store = []
        state_store = [s]
        t = 0
        while True:
            if t < T:
                a = pi.action(s)
                s, r, done, info = env.step(a)
                reward_store.append(r)
                state_store.append(s)
                # env.render()
                if done:
                    T = t+1
            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau+1, min(tau+n, T)+1):
                    G += pow(gamma, i-tau-1) * reward_store[i-1]
                if tau + n < T:
                    G += pow(gamma, n) * V(state_store[tau+n]) # time step of state = tau + 1 = tau + n
                V.update(alpha, G, state_store[tau])
            t += 1
            if tau == T-1:
                break

