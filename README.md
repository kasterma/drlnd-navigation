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