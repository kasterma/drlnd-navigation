# train the agent

import logging.config
from collections import deque

import numpy as np
import yaml
from unityagents import UnityEnvironment

from agent import Agent, AgentInterface

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("train")


class Train:
    def __init__(self, config, episodes_ct=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        self.eps = EpsExponential(eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay)

        self.max_t = max_t
        self.episodes_ct = episodes_ct

        self.agent: AgentInterface = Agent(config)
        self.scores = Scores(filename=config['scores_filename'])
        self.eps = EpsExponential()

        self.env = UnityEnvironment(file_name="Banana.app")
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size

    def run_random(self):
        """Run the environment with random actions.

        Note: purely for testing the environment interaction.
        """
        env_info = self.env.reset(train_mode=False)[self.brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score
        while True:
            action = np.random.randint(self.action_size)  # select an action
            env_info = self.env.step(action)[self.brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                break

        print("Score: {}".format(score))

    def train(self):
        """Run the DQN training with the parameters as set up in the constructor"""

        for i_episode in range(self.episodes_ct):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            score = 0
            state = env_info.vector_observations[0]
            eps = self.eps.next_eps()

            for i_step in range(self.max_t):
                action = self.agent.get_action(state, eps)
                #log.info("action %s (ct %d %s %f)", action, self.agent.step_count, self.brain_name, eps)

                #log.info("env %s", self.env)
                result = self.env.step(action)
                #log.info("result %s", result)
                env_info = result[self.brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
                #log.info("done %s", done)

                self.agent.record_step(state, action, reward, next_state, done)

                state = next_state
                score += reward
                if done:
                    #log.info("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
                    break

            self.scores.add(score)

        log.info("Done training, saving scores and model")
        self.scores.save()
        self.agent.save()

    def test(self):
        """Run the model for 200 episodes (no learning so really testing the model, not a mixture of the model at
        different stages of learning"""

        scores = Scores(print_every=1, filename="testrun")
        for i_episode in range(200):
            env_info = self.env.reset(train_mode=False)[self.brain_name]
            score = 0
            state = env_info.vector_observations[0]

            for i_step in range(self.max_t):
                action = self.agent.get_action(state, 0.0)
                log.info(action)

                env_info = self.env.step(action.numpy())[self.brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                state = next_state
                score += reward
                if done:
                    break

            scores.add(score)

        log.info("Done training, saving scores and model")
        scores.save()

    def close(self):
        self.env.close()


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


class Scores:
    """Storage of scores and printing of progress"""
    log = logging.getLogger("scores")

    def __init__(self, window_length=100, print_every=50, filename=None):
        self.filename = filename
        self.print_every = print_every
        self.scores = []
        self.scores_window = deque(maxlen=window_length)
        self.ct = 0

    def add(self, score):
        self.ct += 1
        self.scores.append(score)
        self.scores_window.append(score)
        if self.ct % self.print_every == 0:
            log.info("After {ct} scores have mean in last {window_length} of {mean}"
                     .format(ct=self.ct,
                             window_length=self.scores_window.maxlen,
                             mean=np.mean(self.scores_window)))

    def save(self):
        """Save the scores in numpy binary file format."""
        if self.filename:
            np.save(self.filename, np.array(self.scores))


if __name__ == "__main__":
    log.info("Starting to train")
    with open("config.yaml") as conf_file:
        conf = yaml.load(conf_file)

    log.info("Using config: %s", conf)
    t = Train(conf)
    #t.run_random()
    t.train()
    t.test()
    t.close()
