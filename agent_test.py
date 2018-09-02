import logging.config
import random
import unittest

import numpy as np
import torch
import yaml

from agent import *

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("agent")


class AgentTest(unittest.TestCase):
    RANDOM_SEED = 42

    def test_experiences(self):
        """Test the raw sampling behavior"""
        e = Experiences(memory_size=4, batch_size=3)

        e.add(np.arange(3), 2, 1.0, np.random.rand(3), False)
        self.assertRaises(NotEnoughExperiences, e.raw_sample)

        e.add(np.arange(3), 2, 2.0, np.random.rand(3), False)
        self.assertRaises(NotEnoughExperiences, e.raw_sample)

        e.add(np.arange(3), 2, 3.0, np.random.rand(3), False)
        s = e.raw_sample()
        self.assertEqual(3, len(s))
        self.assertEqual([1.0, 2.0, 3.0], sorted([x.reward for x in s]))

        s2 = e.raw_sample()
        self.assertEqual(3, len(s2))
        self.assertEqual([1.0, 2.0, 3.0], sorted([x.reward for x in s2]))

        e.add(np.arange(3), 2, 1.0, np.random.rand(3), False)
        e.add(np.arange(3), 2, 1.0, np.random.rand(3), False)
        e.add(np.arange(3), 2, 1.0, np.random.rand(3), False)
        e.add(np.arange(3), 2, 1.0, np.random.rand(3), False)
        s3 = e.raw_sample()
        self.assertEqual(3, len(s3))

    def test_experiences_sample(self):
        """Test the conversion to tensors of sample"""
        e = Experiences(memory_size=4, batch_size=3)

        random.seed(self.RANDOM_SEED)  # the random source that is used in random.sample

        e.add(np.array([1.0, 2.0, 3.0]), 0, 1.0, np.random.rand(3), False)
        e.add(np.array([4.0, 5.0, 6.0]), 1, 2.0, np.random.rand(3), False)
        e.add(np.array([7.0, 8.0, 9.0]), 2, 3.0, np.random.rand(3), False)
        e.add(np.array([10.0, 11.0, 12.0]), 3, 4.0, np.random.rand(3), False)
        e.add(np.array([13.0, 14.0, 15.0]), 0, 5.0, np.random.rand(3), False)
        e.add(np.array([16.0, 17.0, 18.0]), 1, 6.0, np.random.rand(3), False)
        e.add(np.array([19.0, 20.0, 21.0]), 2, 7.0, np.random.rand(3), False)
        s3 = e.raw_sample()
        print(s3)
        log.info("s3 %s", s3)
        self.assertEqual(3, len(s3))

        random.seed(self.RANDOM_SEED)  # reset the seed so we get the same sample as in the raw_sample call just now
        print(e.torch_sample())

    config1 = {'network_spec':
                   {'input_dim': 4,
                    'hidden_1_size': 5,
                    'activation_1': "tanh",
                    'hidden_2_size': 5,
                    'activation_2': "tanh",
                    'output_dim': 2},
               'experience_memory':
                   {'size': 100000},
               'train': {
                   'batch_size': 64,
                   'update_every': 4,
                   'tau': 0.001,
                   'gamma': 0.99,
                   'learning_rate': 0.0005
               }}
    config2 = {'network_spec':
                   {'input_dim': 37,
                    'hidden_1_size': 64,
                    'activation_1': "relu",
                    'hidden_2_size': 64,
                    'activation_2': "relu",
                    'output_dim': 4},
               'experience_memory':
                   {'size': 100000},
               'train': {
                   'batch_size': 64,
                   'update_every': 4,
                   'tau': 0.001,
                   'gamma': 0.99,
                   'learning_rate': 0.0005
               }}

    @staticmethod
    def random_input_tensor(dim):
        return torch.as_tensor(np.random.random(dim), dtype=torch.float)

    def test_act(self):
        """Test if the unsqueeze from the example had any function"""
        input = AgentTest.random_input_tensor(4)
        a = Agent(self.config1)
        log.info(a.get_action(input.numpy(), 0.1))


if __name__ == "__main__":
    log.info("Testing the agent")
    unittest.main()
