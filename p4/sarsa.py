import numpy as np
from math import ceil


class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement here
        dimensions = [num_tilings]
        for idx in range(len(state_low)):
            dimensions.append(ceil((state_high[idx]-state_low[idx])/tile_width[idx]) + 1)
        dimensions.append(num_actions)
        self.state_widths = [tile_width[idx] for idx in range(len(state_low))]
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.dimensions = dimensions
        self.indexes = [np.prod(dimensions[idx+1:]) for idx in range(len(dimensions))]
        self.weights = np.zeros(tuple(dimensions)) # dimensions = (num_tilings, state_dim_1, state_dim_2, ..., state_dim_n, num_actions)

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # TODO: implement this method
        return np.prod(self.weights.shape)

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        # TODO: implement this method
        to_return = np.zeros((self.feature_vector_len()))
        if not done:
            for i in range(self.num_tilings):
                index = i * self.indexes[0] #offset into the tile number
                for j in range(1, len(self.dimensions) - 1): # state dimensions
                    state_index = (s[j-1] - self.state_low[j-1]) // self.state_widths[j-1]
                    # print((s[j-1] - self.state_low[j-1])//self.state_widths[j-1])
                    index += state_index * self.indexes[j]
                index += a #action index
                index = int(index)
                # print(index)
                to_return[index] = 1
        # print("...")
        return to_return

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))
    for _ in range(num_episode):
        env.reset()
        s = env.observation_space.sample()
        a = epsilon_greedy_policy(s, False, w, epsilon=.0005)
        x = X(s, False, a)
        z = np.zeros((X.feature_vector_len()))
        Q_old = 0
        done = False
        while not done:
            s_p, r, done, info = env.step(a)
            a_p = epsilon_greedy_policy(s_p, done, w, epsilon=.0005)
            x_p = X(s_p, done, a_p)
            Q = np.dot(w, x)
            Q_p = np.dot(w, x_p)
            delta = r + (gamma * Q_p) - Q
            z = (lam * gamma * z) + (1 - (alpha * lam * gamma * np.dot(z, x))) * x
            w = w + (alpha * ((delta + Q - Q_old) * z)) - (alpha*(Q - Q_old) * x)
            Q_old = Q_p
            x = x_p
            a = a_p
        # print(_)
    print(w)
    return w




