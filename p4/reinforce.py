from re import T
from typing import Iterable
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import math

class PiNN(torch.nn.Module):
    # n_hidden = number of nodes in hidden layer
    # n_layers = number of hidden layers
    # n_feature = number of inputs
    # n_output = number of outputs
    def __init__(self, n_feature, n_layers, n_hidden : int, n_output):
        super(PiNN, self).__init__()
        # self.hi
        self.hidden0 = (torch.nn.Linear(n_feature, n_hidden))
        self.hidden1 = (torch.nn.Linear(n_hidden, n_hidden))
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, input):
        # print("1")
        layer_output = F.relu(self.hidden0(input))
        # print(layer_output)
        # for i in range(1, len(self.hidden)):
        layer_output = F.relu(self.hidden1(layer_output))      # activation function for hidden layer
        # print(layer_output)
        # print(self.predict(layer_output))
        actions = self.predict(layer_output)
        network_output = F.softmax(actions)    # linear output
        # print(network_output)
        # print("3")
        # log_prob_action = torch.log(actions.sqeueze(0))[network_output]
        return network_output

class VNN(torch.nn.Module):
    def __init__(self, n_feature, n_layers, n_hidden : int, n_output):
        super(VNN, self).__init__()
        self.hidden0 = (torch.nn.Linear(n_feature, n_hidden))
        self.hidden1 = (torch.nn.Linear(n_hidden, n_hidden))
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, input):
        layer_output = F.relu(self.hidden0(input))
        layer_output = F.relu(self.hidden1(layer_output))      # activation function for hidden layer
        network_output = self.predict(layer_output)    # linear output
        return network_output


class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here

        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
            # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
            # a_t = tf.constant([1, 2])
            # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
            # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size
        self.nn = PiNN(state_dims, 2, 32, num_actions)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), betas=[.9, .999], lr=alpha)
        self.probabilities = dict()

    def get_action_prob(self, s, a):
        self.nn.eval()
        action_probs = self.nn(torch.from_numpy(s).float())
        return action_probs[a]

    def __call__(self,s) -> int:
        # TODO: implement this method
        self.nn.eval()
        action_probs = self.nn(torch.from_numpy(s).float())
        # print(action_probs)
        m = Categorical(action_probs)
        action = (m.sample()).item()
        # print(action)
        # self.probabilities[(tuple(s), action)] = action_probs[action].item()
        self.probabilities[(tuple(s), action)] = torch.log(action_probs.squeeze(0)[action])

        return action

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.nn.train()
        # print(self.probabilities[(tuple(s), a)])
        loss = torch.tensor(gamma_t) * torch.tensor(delta) * -1*self.probabilities[(tuple(s), a)]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return None

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.alpha = alpha
        self.nn = VNN(state_dims, 2, 32, 1) # num_layers not used :)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), betas=[.9, .999], lr=alpha)
        

    def __call__(self,s) -> float:
        # TODO: implement this method
        self.nn.eval()
        val = self.nn(torch.from_numpy(s).float())
        # print(val.item())
        # assert val.item() != np.NAN, val
        return val.item()

    def update(self,s,G):
        # TODO: implement this method
        self.nn.train()
        self.optimizer.zero_grad()
        prediction = self.nn(torch.from_numpy(s).float())
        # print(prediction)
        # print(type(prediction))
        loss_function = torch.nn.MSELoss()
        # print(next(self.nn.parameters()))

        # return
        loss = loss_function(prediction, torch.tensor([G]))
        loss.backward()
        self.optimizer.step()



def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    to_return = []
    for ep_num in range(num_episodes):
        Episode = []
        s = env.reset()
        a = pi(s)
        done = False
        
        while not done:
            s_p, r, done, info = env.step(a)
            Episode.append((s, a, r))
            s = s_p
            a = pi(s_p)
            # if done:
                # print(s_p)
                # env.render()

        T = len(Episode)
        for t, step in enumerate(Episode):
            s, a, r = step 
            G = sum([pow(gamma, idx)*Episode[idx][2] for idx in range(t, T)])
            if t == 0:
                to_return.append(G)
            # print("state : " + str(s))
            delta = G - V(s)
            # print(delta)
            V.update(s, G)
            pi.update(s, a, pow(gamma, t), delta)
        
        print("ep_num " + str(ep_num))
    return to_return

    

