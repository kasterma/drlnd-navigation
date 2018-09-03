# DRLND: Navigation Project

## Introduction

This is a solution to the first project in the deep reinforcement
learning nanadegree from Udacity.  At this point we learned some of
the basics about reinforcement learning, and are using the ideas from
the deepmind paper (Human-level control through deep reinforcement
learning), in particular the experience replay and not updating the
target in every step; both for stability of the Q function we are
learning.  See `Report.md` a description of the architecture and
results obtained.  Here we'll describe how to set up the
environment to replicate our results.

## Getting started

Note: in the below steps if you are not on Mac OS X you will likely
have to adjust the makefile.  For running in Linux the options are
shown in the makefile (but they have yet to be tested).

First run

    make get-environment

to get the correct environment downloaded (this is an adjusted version
of the banana collector environment of the Unity ml-agents package).

Then we set up a virtual environment with all needed packages
installed

    make venv

Note: the pytorch install there mirrors what is instructed on
pytorch.org for Mac OSX, in other environments this may need to be
changed (first thing to try: run the above, then go to pytorch.org for
the system specific instructions and run these in the activated
virtual env).

### Testing the environment is correctly set up

Udacity has provided example code for interacting with the
environment, this step downloads this example code, and instructs how
to run it.

Run

    make get-example

then

    jupyter notebook

In the notebook update the second cell to contain the right filename
(on Mac OSX this is Banana.app in place of ...).

Then run the notebook to see the environment works.

Alternately you can run

    make run-interactive
    
or run the script interactive.py in your preferred python environment,
to see the interaction work.  This script is also very suited to
get some feel for the interaction in the environment.  There is a lot
of logging, which you can see what it means in the code of interact.py

## Running the code

Finally you can train the model (storing the episode scores in train_scores.npy
and the model weights in trained_model-fully_trained.pth) using

    make train
    
Note: this also stores the model weights every 100 steps for later
analysis.

## The environment

The agent is in a large square world and will need to collect bananas. Collecting a yellow banana gives a reward of +1,
a blue banana gives a reward of -1.

The agent is not given a camera in the space, all it has for sensing is a collection of rays (pointing generally
forward) that provide information on what can be sensed in the ray direction.  Length of the ray, and the details of
what they are sensing are not provided but we think of sensing up to a limited distance, and then the distance and
color and type (yellow banana, blue banana, zzz wall) of the object that the ray intersects.

From playing with the environment (interact.py) an episode in the environment is 300 steps of the environent.

## Environment summary

### Reward

| event         | reward  |
|---------------|:-------:|
| yellow banana |   +1    |
| blue banana   |   -1    |

### State space

37 dimensional; velocity + rays for sensing.

### Actions

| index   | action        |
|---------|---------------|
| 0       | move forward  |
| 1       | move backward |
| 2       | turn left     |
| 3       | turn right    |

### Solved

Mean score +13 over 100 consequtive episodes, where an episode consists of 300 steps in the environment.

## Files

- agent.py: key code for managing the interaction between the environment
  and the model
- agent_test.py: tests for the agent and support classes, and playing
  with the code from agent.py
- analysis.py: code for getting a basic analysis of the episode scores
  and generating the plot that summarises the learning run
- config.yaml: configuration information for the learning and model
- interact.py: code extracted from the example Navigation.ipynb to test
  that we have the environment set up correctly
- logging.yaml: configuration of the logging system
- Makefile
- model.py: implementation of nn.Model we are using
- model_test.py: tests for the model and playing with its code
- Notes.md: general notes taken while working on the assignment
- README.md: you are reading this now
- Report.md: the assignment report, explains the training and results
  in detail
- requirements.txt: results of pip freeze, needed for setting up the
  environment
- specification.md: the requirements as given for this assignment
- train.py: the training driver
- train_scores.npy: the scores saved from a training run
- trained_model-fully_trained.pth: the model weights saved at the end
  of the training run