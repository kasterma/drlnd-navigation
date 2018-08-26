# Use the code from the example notebook to get the interaction with the environment working/check that it is working
#
# This script is set up to run repeatedly with minor changes to see how the environment reacts.
# When done, don't forget to run env.close() to close the environment.

import logging.config
import yaml

from unityagents import UnityEnvironment
import numpy as np

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("interact")

with open("config.yaml") as conf_file:
    config = yaml.load(conf_file)

log.info("Current config: {}".format(yaml.dump(config)))

# on other than first runs of the script the env already exists, but on the first run the environment needs to be set up
try:
    env
except NameError:
    env = UnityEnvironment(file_name=config["environment"])

# reset the environment
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]


# get the default brain
assert len(env.brain_names) == 1  # This environment has only one brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# number of agents in the environment
log.info('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
log.info('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
log.info('States look like:', state)
state_size = len(state)
log.info('States have length:', state_size)

env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
state = env_info.vector_observations[0]  # get the current state
score = 0  # initialize the score
steps = 0
while True:
    steps += 1
    action = np.random.randint(action_size)  # select an action
    log.info("action %d", action)
    env_info = env.step(action)[brain_name]  # send the action to the environment
    next_state = env_info.vector_observations[0]  # get the next state
    log.info("state %s", next_state)
    reward = env_info.rewards[0]  # get the reward
    log.info("reward %d", reward)
    done = env_info.local_done[0]  # see if episode has finished
    score += reward  # update the score
    state = next_state  # roll over the state to next time step
    if done:  # exit loop if episode finished
        break

log.info("Score: %d, steps %d", score, steps)

# env.close()   # since we want to run this script repeatedly with different logging we keep the env open
