# DRLND: Navigation Project

## Introduction

This is a solution to the first project in the deep reinforcement
learning nanadegree from Udacity.  At this point we learned some of
the basics about reinforcement learning, and are using the ideas from
the deepmind paper (Human-level control through deep reinforcement
learning), in particular the experience replay and not updating the
target in every step; both for stability of the Q function we are
learning.  Q here is of the form

    Q : state space -> (list of action values))

So this is the function we are approximating with a neural network.
We are trying to learn the values of states in a modified Banana
Collector environment (modified from the Banana Collector that is part
of the Unity ML-Agents package), that we'll describe more below.

## Environment

The agent is in a large square world and will need to collect bananas.
Collecting a yellow banana gives a reward of +1, a blue banana gives a
reward of -1.

The agent is not given a camera in the space, all it has for sensing
is a collection of rays (pointing generally forward) that provide
information on what can be sensed in the ray direction.  Length of the
ray, and the details of what they are sensing are not provided but we
think of sensing up to a limited distance, and then the distance and
color and type (yellow banana, blue banana, zzz wall) of the object
that the ray intersects.

From playing with the environment (interact.py) an episode in the
environment is 300 steps of the environent.

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

Mean score +13 over 100 consequtive episodes, where an episode consists
of 300 steps in the environment.

## Getting started

Note: in the below steps if you are not on Mac OS X you will likely
have to adjust the makefile.  For running in Linux the options are
shown in the makefile.

First run

    make get-environment

to get the correct environment downloaded.

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

Then run the notebook to see the environment works.  When prompted
("Do you want the application “Python.app” to accept incoming network
connections?") click accept or deny in the network dialog (in my
environment it doesn't make a differenc for if the environment runs.

Alternately you can run

    make run-interactive
    
or run the script interactive.py in your preferred python environment,
to see the interaction work.  This script is also very suited to
get some feel for the interaction in the environment.

## Running the code

We first set up a virtual environment with all needed packages
installed

    make venv

Note: the pytorch install there mirrors what is instructed on
pytorch.org for Mac OSX, in other environments this may need to be
changed (first thing to try: run the above, then go to pytorch.org for
the system specific instructions and run these in the activated
virtual env).

## Notes

We were told that using only the methods used from an earlier section
(describing the human-level control paper) we should be able to solve
the project in fewer than 1800 episodes.