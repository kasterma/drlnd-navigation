# Agent for mediating interactino between the environment and the model

import logging.config
import yaml

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("train")

from model import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)     # for now set the random seed here


class Agent:
    def __init__(self, config, env):
        self.model = Model(config['network_spec'])
        self.env = env


class AgentTest(unittest.TestCase):
    def test_pass(self):
        pass


if __name__ == "__main__":
    log.info("Testing the agent")
    unittest.main()
