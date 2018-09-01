# train the agent

import logging.config
import yaml

from unityagents import UnityEnvironment
import numpy as np

from agent import Agent, AgentInterface

from collections import deque

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("train")

with open("config.yaml") as conf_file:
    config = yaml.load(conf_file)


class Train:
    def __init__(self, episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        self.eps = EpsExponential(eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay)

        self.max_t = max_t
        self.episodes = episodes

        self.agent: AgentInterface = Agent()
        self.scores = []
        self.scores_windo = deque(maxlen=100)

    def train(self):
        # iterate over episodes, every episode take at most max_t steps in the environment


class EpsInterface:
    def next_eps(self):
        raise NotImplementedError()


class EpsExponential(EpsInterface):
    """Exponentially decaying epsilon"""

    def __init__(self, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.eps_next = self.eps_start

    def next_eps(self):
        rv = self.eps_next
        if self.eps_next > self.eps_end:
            self.eps_next *= self.eps_decay
        return rv


class EpsLinear(EpsInterface):
    """Linearly decaying epsilon"""

    def __init__(self, eps_start=1.0, eps_end=0.01, eps_decay_steps=1000):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.eps_decay = (self.eps_start - self.eps_end) / self.eps_decay_steps

        self.eps_next = self.eps_start

    def next_eps(self):
        rv = self.eps_next
        if self.eps_next > self.eps_end:
            self.eps_next -= self.eps_decay
        return rv


if __name__ == "__main__":
    log.info("Starting to train")