# The Neural Network Model

import logging.config
import yaml

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("model")


class Model(nn.Module):
    """
    Model with two hidden layers and relu or tanh activation, exact sizes of layers and activation function are
    specified in the spec that is passed in.
    """
    activation = {"relu": F.relu, "tanh": F.tanh}

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


class ModelTests(unittest.TestCase):
    """
    In this collection of tests we don't just test the functioning of the model but also in several ways our
    understanding of the functioning of PyTorch.
    """
    config1 = {'input_dim': 4,
               'hidden_1_size': 5,
               'activation_1': "tanh",
               'hidden_2_size': 5,
               'activation_2': "tanh",
               'output_dim': 2}
    config2 = {'input_dim': 37,
               'hidden_1_size': 64,
               'activation_1': "relu",
               'hidden_2_size': 64,
               'activation_2': "relu",
               'output_dim': 4}

    @staticmethod
    def random_input_tensor(dim):
        return torch.as_tensor(np.random.random(dim), dtype=torch.float)

    def test_create_model(self):
        """Test that we can create a model from the config.yaml in the project and both config in this test"""
        with open("config.yaml") as conf_file:
            config = yaml.load(conf_file)

        log.info(config)

        input = ModelTests.random_input_tensor(config['network_spec']['input_dim'])
        Model(config["network_spec"]).forward(input)
        Model(self.config1)
        Model(self.config2)

    def test_effect_randomseed(self):
        """Test that our current understanding of torch.manual_seed is appropriate for the runs we intend to do"""
        torch.manual_seed(42)
        m = Model(self.config1)
        torch.manual_seed(42)
        m2 = Model(self.config1)
        # Check that in the first layer the weights and bias are the same
        self.assertTrue(torch.all(torch.eq(m.wts1.weight, m.wts1.weight)))
        self.assertTrue(torch.all(torch.eq(m.wts1.bias, m.wts1.bias)))
        torch.manual_seed(0)
        m3 = Model(self.config1)
        # Check that in the first layer the weights and bias are not the same
        self.assertFalse(torch.all(torch.eq(m3.wts1.weight, m.wts1.weight)))
        self.assertFalse(torch.all(torch.eq(m3.wts1.bias, m.wts1.bias)))

    def test_model_sanity1(self):
        """Test all parameters affect the output"""
        for _ in range(200):
            m1 = Model(self.config1)
            wts_before = torch.tensor(m1.wts1.weight)
            input1 = ModelTests.random_input_tensor(self.config1['input_dim'])
            output1 = m1.forward(input1)
            idx1 = np.random.randint(m1.wts1.weight.shape[0])
            idx2 = np.random.randint(m1.wts1.weight.shape[1])
            m1.wts1.weight[idx1, idx2] = m1.wts1.weight[idx1, idx2] + 1.0   # change a wt
            output2 = m1.forward(input1)
            if True: # torch.all(torch.eq(output1, output2)):
                log.info("idx1 %d idx2 %d", idx1, idx2)
                log.info(wts_before)
                log.info(m1.wts1.weight)
            self.assertFalse(torch.all(torch.eq(output1, output2)))

    def test_model_copy(self):
        m = Model(self.config1)
        m_copy = m.get_copy()
        input = ModelTests.random_input_tensor(self.config1['input_dim'])
        y = m.forward(input)
        y_copy = m_copy.forward(input)
        self.assertTrue(torch.all(torch.eq(y, y_copy)))


if __name__ == "__main__":
    log.info("Running tests on model, note this is NOT testing the effectiveness of the model, only its functioning")
    unittest.main()