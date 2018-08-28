# Agent for mediating interactino between the environment and the model

import logging.config
import yaml

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque
import numpy as np
import random

from typing import List

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("agent")

from model import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)     # for now set the random seed here


class Agent:
    def __init__(self, config):
        self.model = Model(config['network_spec'])
        self.memory = Experiences(config=config)


class NotEnoughExperiences(Exception):
    """Exception to throw when Experiences is asked for a sample but doesn't have enough data yet"""
    pass


class Experience:
    """Wrapper class for an experience as received from the environment"""

    def __init__(self, state, action, reward, next_state, done):
        self.state: np.ndarray = state
        self.action: int = action
        self.reward: float = reward
        self.next_state: np.ndarray = next_state
        self.done: bool = done

    def __repr__(self):
        return "Experience(" + str(self.state) + "," + str(self.action) + "," + str(self.reward) + "," +\
               str(self.next_state) + "," + str(self.done) + ")"


class Experiences:
    """
    Fixed-size buffer to store experience tuples.

    Store the data as received from the Unity environment, but before returning the sample prepare it for use by the
    agent for learning (the agent can ignore that it seems to store numpy data and needs torch tensors).
    """

    def __init__(self, memory_size, batch_size, *, config=None):
        if config:
            assert memory_size is None
            assert batch_size is None
            memory_size = config['experience_memory']['size']
            batch_size = config['train']['batch_size']
        self.memory = deque(maxlen=memory_size)
        self.sample_size = batch_size      # in the current learning method we return a sample the size of a batch

    def add(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def raw_sample(self):
        if len(self.memory) < self.sample_size:
            raise NotEnoughExperiences()
        return random.sample(self.memory, k=self.sample_size)

    def torch_sample(self):
        sample = self.raw_sample()

        # convert sample for use in PyTorch:
        # 1) reshape to not have a sequence of experiences, but sequences of experience parts (sequence of states,
        #    sequence of rewards, ....)
        # 2) convert to PyTorch tensors

        def select(f):
            return [f(e) for e in sample]

        states: List[np.ndarray] = select(lambda e: e.state)
        actions: List[int] = select(lambda e: e.action)
        rewards: List[float] = select(lambda e: e.reward)
        next_states: List[np.ndarray] = select(lambda e: e.next_state)
        dones: List[bool] = select(lambda e: e.done)

        def to_tensor(dat):
            return torch.from_numpy(np.vstack(dat))

        def to_float_tensor(dat):
            return to_tensor(dat).float().to(device)

        states_tensor = to_float_tensor(states)
        actions_tensor = to_tensor(actions).long().to(device)
        rewards_tensor = to_float_tensor(rewards)
        next_states_tensor = to_float_tensor(next_states)
        dones_tensor = to_float_tensor(np.array(dones, dtype=np.uint8))   # need to preprocess the booleans

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor


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

        random.seed(self.RANDOM_SEED)   # the random source that is used in random.sample

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

        random.seed(self.RANDOM_SEED)    # reset the seed so we get the same sample as in the raw_sample call just now
        print(e.torch_sample())


if __name__ == "__main__":
    log.info("Testing the agent")
    unittest.main()
