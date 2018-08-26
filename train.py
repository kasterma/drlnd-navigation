# train the agent

import logging.config
import yaml

from unityagents import UnityEnvironment
import numpy as np

from agent import Agent

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("train")

with open("config.yaml") as conf_file:
    config = yaml.load(conf_file)


class Train:
    def __init__(self):
        self.agent = Agent()


if __name__ == "__main__":
    log.info("Starting to train")