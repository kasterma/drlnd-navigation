# The Neural Network Model

import logging.config

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("model")


class Model(nn.Module):
    """
    Model with two hidden layers and relu or tanh activation, exact sizes of layers and activation function are
    specified in the spec that is passed in.
    """
    activation = {"relu": F.relu, "tanh": torch.tanh}

    @classmethod
    def activation_fn(cls, activation_spec):
        return cls.activation[activation_spec]

    def __init__(self, spec):
        super().__init__()
        self.spec = spec

        hidden1 = self.spec['hidden_1_size']
        self.activation1 = Model.activation_fn(self.spec['activation_1'])
        hidden2 = self.spec['hidden_2_size']
        self.activation2 = Model.activation_fn(self.spec['activation_2'])
        self.wts1 = nn.Linear(self.spec['input_dim'], hidden1)
        self.wts2 = nn.Linear(hidden1, hidden2)
        self.wts3 = nn.Linear(hidden2, spec['output_dim'])

    def forward(self, x):
        z1 = self.wts1(x)
        h1 = self.activation1(z1)
        z2 = self.wts2(h1)
        h2 = self.activation2(z2)
        z2 = self.wts3(h2)
        return z2

    def get_copy(self):
        copy = Model(self.spec)
        for copy_param, self_param in zip(copy.parameters(), self.parameters()):
            copy_param.data.copy_(self_param.data)
        return copy
