# Agent for mediating interactino between the environment and the model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Model


class Agent:
    def __init__(self):
        self.model = Model()
