# Agent for mediating interactino between the environment and the model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)     # for now set the random seed here


class Agent:
    def __init__(self):
        self.model = Model()
