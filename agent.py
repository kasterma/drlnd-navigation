# Agent for mediating interactino between the environment and the model

import logging.config
import random
import unittest
from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

from model import Model

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("agent")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random_seed = 42
torch.manual_seed(random_seed)


class AgentInterface:
    def record_step(self, state, action, reward, next_state, done):
        """Record the step and possilby perform a learning step"""
        raise NotImplementedError()

    def get_action(self, state, eps=0.0):
        """Use the contained model to get the next action to perform in the environment"""
        raise NotImplementedError()

    def save(self, id):
        """Save the model variables"""
        raise NotImplementedError()

    def load_model(self, filename):
        """Load model coefficients from the passed file"""
        raise NotImplementedError()


class Agent(AgentInterface):

    def __init__(self, config):
        self.action_size = config['network_spec']['output_dim']
        self.update_every = config['train']['update_every']
        self.tau = config['train']['tau']
        self.gamma = config['train']['gamma']
        self.learning_rate = config['train']['learning_rate']

        self.local_model = Model(config['network_spec']).to(device)
        self.target_model = self.local_model.get_copy().to(device)

        self.memory = Experiences(config=config)

        self.optimizer = optim.Adam(self.local_model.parameters(), lr=self.learning_rate)

        self.step_count = 0

    def get_action(self, state, eps):
        """Select an action epsilon greedy from the local_model"""
        if random.random() > eps:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)   # TODO: test if we needed unsqueeze after all
            self.local_model.train(False)
            with torch.no_grad():
                action_values = self.local_model.forward(state_tensor)
            return int(np.argmax(action_values))     # last type error
        else:
            return random.randint(0, self.action_size-1)

    def record_step(self, state, action, reward, next_state, done):
        self.step_count += 1
        self.memory.add(state, action, reward, next_state, done)

        if self.step_count % self.update_every == 0 and self.memory.big_enough():
            sample = self.memory.torch_sample()
            self.learn(sample)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        log.debug(states.size())
        log.debug(next_states.size())

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        log.debug(Q_targets_next.size())
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        log.debug(Q_targets.size())
        log.debug(rewards.size())

        # Get expected Q values from local model
        val = self.local_model(states)
        log.debug(val.size())
        log.debug("actions %s", actions.size())
        log.debug("action values %s", actions)
        Q_expected = val.gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.local_model, self.target_model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, id="missing_model_id"):
        torch.save(self.local_model.state_dict(), "trained_model-{id}.pth".format(id=id))

    def load_model(self, filename):
        self.local_model.load_state_dict(torch.load(filename))


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

    def __init__(self, memory_size=None, batch_size=None, *, config=None):
        if config:
            assert memory_size is None
            assert batch_size is None
            memory_size = config['experience_memory']['size']
            batch_size = config['train']['batch_size']
        self.memory = deque(maxlen=memory_size)
        self.sample_size = batch_size      # in the current learning method we return a sample the size of a batch

    def add(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def big_enough(self):
        return len(self.memory) > self.sample_size

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
            return [f(e) for e in sample if e is not None]

        states: List[np.ndarray] = select(lambda e: e.state)
        actions: List[int] = select(lambda e: e.action)
        rewards: List[float] = select(lambda e: e.reward)
        next_states: List[np.ndarray] = select(lambda e: e.next_state)
        dones: List[bool] = select(lambda e: e.done)

        def to_tensor(dat):
            return torch.from_numpy(np.vstack(dat))

        def to_float_tensor(dat):
            return to_tensor(np.vstack(dat)).float().to(device)

        states_tensor = to_float_tensor(states)
        actions_tensor = to_tensor(actions).long().to(device)
        rewards_tensor = to_float_tensor(rewards)
        next_states_tensor = to_float_tensor(next_states)
        dones_tensor = to_float_tensor(np.array(dones, dtype=np.uint8))   # need to preprocess the booleans

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

