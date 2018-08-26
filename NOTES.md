# Notes

Lab notebook to keep track of what was done; to be used as a basis for the report.

- DONE: set up environment in repeatable way (so can easily move to different computer if need be)
- DONE: set up interact.py to check the environment, and learn how to interact with it
- DONE: set up train.py, agent.py, and model.py as a framework for the training
- fill in model.py and do some testing with it
- fill in agent.py and do some testing with it
- fill in train.py and do some testing with it
- train.py version in a notebook to get graphs while learning
- how to test easily how much different a GPU can make?
- how to test our use of the GPU?
- in the lunar lander problem the network that worked well was more complicated than expected; same here?

## Architecture

For starters try similar archtecture as worked for the lunar lander: two hidden layers, relu in between.

Make variable:
- size of the hidden layers
- number of hidden layers

## Training

The paper we are basing this on used experience replay, and a distinct less frequently updated target network.  Try
the same.  There we updated the target network (soft update) after every batch of experiences was run through.

In lunar lander used Adam, try the same here.

- also try basic SGD optim.SGD(net.parameters(), lr=0.001)
- also try SGD with momentum optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
- try different learning rates
- try different sizes of the replay buffer; we expect the initial experience not to be very useful b/c it is too random.
  When does it start becoming useful?