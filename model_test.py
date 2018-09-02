# Testing/Interacting with the Neural Network Model
#
# Here we check that we can build models as desired, and run some general interaction with the model to learn to
# understand PyTorch model code better.

import logging.config
import unittest

import numpy as np
import torch
import yaml

from model import Model

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("model")


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