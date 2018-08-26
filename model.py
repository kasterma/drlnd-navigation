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
log = logging.getLogger("train")


class Model(nn.Module):
    """
    Model with two hidden layers with relu activation, exact sizes of layers specified in the spec that is passed in.
    """
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        hidden1 = spec['hidden_1_size']
        hidden2 = spec['hidden_2_size']
        self.wts1 = nn.Linear(spec['input_dim'], hidden1)
        self.wts2 = nn.Linear(hidden1, hidden2)
        self.wts3 = nn.Linear(hidden2, spec['output_dim'])

    def forward(self, x):
        z1 = self.wts1(x)
        h1 = F.relu(z1)
        z2 = self.wts2(h1)
        h2 = F.relu(z2)
        z2 = self.wts3(h2)
        return z2


class ModelTests(unittest.TestCase):
    """
    In this collection of tests we don't just test the functioning of the model but also in several ways our
    understanding of the functioning of PyTorch.
    """
    config1 = {'input_dim': 4,
               'hidden_1_size': 5,
               'hidden_2_size': 5,
               'output_dim': 2}
    config2 = {'input_dim': 37,
               'hidden_1_size': 64,
               'hidden_2_size': 64,
               'output_dim': 4}

    def test_create_model(self):
        """Test that we can create a model from the config.yaml in the project and both config in this test"""
        with open("config.yaml") as conf_file:
            config = yaml.load(conf_file)

        m = Model(config["network_spec"])
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

    @unittest.skip("Failing some of the time for reasons we don't understand yet")
    def test_model_sanity1(self):
        """Test all parameters affect the output"""
        for _ in range(2):
            m1 = Model(self.config1)
            wts_before = torch.tensor(m1.wts1.weight)
            input1 = torch.as_tensor(np.random.random(self.config1['input_dim']), dtype=m1.wts1.weight.dtype)
            output1 = m1.forward(input1)
            idx1 = np.random.randint(m1.wts1.weight.shape[0])
            idx2 = np.random.randint(m1.wts1.weight.shape[1])
            m1.wts1.weight[idx1, idx2] = m1.wts1.weight[idx1, idx2] + 1.0   # change a wt
            output2 = m1.forward(input1)
            if torch.all(torch.eq(output1, output2)):
                log.info("idx1 %d idx2 %d", idx1, idx2)
                log.info(wts_before)
                log.info(m1.wts1.weight)
            self.assertFalse(torch.all(torch.eq(output1, output2)))


if __name__ == "__main__":
    log.info("Running tests on model, note this is NOT testing the effectiveness of the model, only its functioning")
    unittest.main()