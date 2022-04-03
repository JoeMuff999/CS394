import numpy as np
from algo import ValueFunctionWithApproximation

# import tensorflow as tf
import torch
import torch.nn.functional as F

class NN(torch.nn.Module):
    # n_hidden = number of nodes in hidden layer
    # n_layers = number of hidden layers
    # n_feature = number of inputs
    # n_output = number of outputs
    def __init__(self, n_feature, n_layers, n_hidden : int, n_output):
        super(NN, self).__init__()
        # self.hi
        self.hidden0 = (torch.nn.Linear(n_feature, n_hidden))
        self.hidden1 = (torch.nn.Linear(n_hidden, n_hidden))
        # for _ in range(n_layers):
            # self.hidden.append(torch.nn.Linear(n_feature, n_hidden))

        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, input):
        # print("1")
        layer_output = F.relu(self.hidden0(input))
        # print(layer_output)
        # for i in range(1, len(self.hidden)):
        layer_output = F.relu(self.hidden1(layer_output))      # activation function for hidden layer
        # print(layer_output)
        network_output = self.predict(layer_output)                 # linear output
        # print("3")
        return network_output

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        # print(state_dims)
        self.nn = NN(state_dims, 2, 32, 1)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), betas=[.9, .999], lr=.001)

    
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        self.nn.eval()
        # print(torch.from_numpy(s))
        # print(self.nn)
        return self.nn(torch.from_numpy(s).float()).item()

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
        # TODO: implement this method
        self.nn.train()
        self.optimizer.zero_grad()
        # print(self.nn)
        # print("UPDATE")
        prediction = self.nn(torch.from_numpy(s_tau).float())
        loss_function = torch.nn.MSELoss()
        # print(prediction)
        val = 1
        # prediction.backward()
        # if prediction.grad() != None:
        #     val = prediction.grad()
        # print(G-prediction)
        loss = loss_function(prediction, torch.tensor([G]))
        loss.backward()
        print(loss)
        # print(loss.grad)
        self.optimizer.step()

        return None

