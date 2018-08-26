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

Mean score +13 over 100 consequtive episodes.

## Getting started

TBD

## Running the code

TBD

## Notes

We were told that using only the methods used from an earlier section
(describing the human-level control paper) we should be able to solve
the project in fewer than 1800 episodes.