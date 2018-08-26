# Notes

Lab notebook to keep track of what was done; to be used as a basis for the report.

- DONE: set up environment in repeatable way (so can easily move to different computer if need be)
- DONE: set up interact.py to check the environment, and learn how to interact with it
- DONE: set up train.py, agent.py, and model.py as a framework for the training
- DONE: fill in model.py and do some testing with it
- fill in agent.py and do some testing with it
- fill in train.py and do some testing with it
- train.py version in a notebook to get graphs while learning
- how to test easily how much different a GPU can make?
- how to test our use of the GPU?
- in the lunar lander problem the network that worked well was more complicated than expected; same here?
- understand the failing unittest in model.py, and extend the tests

## Architecture

For starters try similar archtecture as worked for the lunar lander: two hidden layers, relu in between.

Make variable:
- size of the hidden layers
- POSTPONED: number of hidden layers

Note: unclear how to make nn.Module extension dynamic in the number of layers.  Do that later.

## Training

The paper we are basing this on used experience replay, and a distinct less frequently updated target network.  Try
the same.  There we updated the target network (soft update) after every batch of experiences was run through.

In lunar lander used Adam, try the same here.

- also try basic SGD optim.SGD(net.parameters(), lr=0.001)
- also try SGD with momentum optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
- try different learning rates
- try different sizes of the replay buffer; we expect the initial experience not to be very useful b/c it is too random.
  When does it start becoming useful?
  
## Torch

### Randomness issues

`torch.manual_seed(seed)` sets the seed for generating random numbers.  Returns a torch._C.Generator object.  The
example code sets this in the model initialization.  The exact execution model for this is not clear to me, seems to me
should be called once per device initialization.  Hence not as part of model initialization.

TODO: find right way of thinking about `torch.<do random stuff>`.

How does the setting of the random seed interact with the `<tensor>.to(device)` action?

TODO: run test to see that with same seed get same results, and different seed get different results.  Then current
random.seed code is at least effective (if not shown to be correct)

Answer: these tests are part of the testing in model.py  This has not been run with multiple devices (e.g. not run
with a GPU)

Idea: would make sense to just generate the initial values on the CPU, and only then ship the weigths off to the GPU
for training.  Then there are no concerns about random generation.

### Random changes in a model affect the output

Test code

    def test_model_sanity1(self):
        """Test all parameters affect the output"""
        for _ in range(2):
            m1 = Model(self.config1)
            wts_before = torch.tensor(m1.wts1.weight)
            input1 = torch.as_tensor(np.random.random(self.config1['input_dim']), dtype=m1.wts1.weight.dtype)
            output1 = m1.forward(input1)
            idx1 = np.random.randint(m1.wts1.weight.shape[0])
            idx2 = np.random.randint(m1.wts1.weight.shape[1])
            m1.wts1.weight[idx1, idx2] = m1.wts1.weight[idx1, idx2] + 1.0   # change a wt
            output2 = m1.forward(input1)
            if torch.all(torch.eq(output1, output2)):
                log.info("idx1 %d idx2 %d", idx1, idx2)
                log.info(wts_before)
                log.info(m1.wts1.weight)
            self.assertFalse(torch.all(torch.eq(output1, output2)))

Output when failing (though it succeeds often)        
            
    2018-08-26 15:57:52,988 - train - INFO - idx1 3 idx2 0
    2018-08-26 15:57:52,989 - train - INFO - tensor([[ 0.3579, -0.0514,  0.0139, -0.0431],
            [ 0.1012,  0.3179,  0.4736,  0.3175],
            [ 0.4747, -0.0362, -0.4492, -0.2370],
            [ 0.3405, -0.0032, -0.2485, -0.3832],
            [-0.4679, -0.4220, -0.1014,  0.2742]], grad_fn=<CopyBackwards>)
    2018-08-26 15:57:52,990 - train - INFO - Parameter containing:
    tensor([[ 0.3579, -0.0514,  0.0139, -0.0431],
            [ 0.1012,  0.3179,  0.4736,  0.3175],
            [ 0.4747, -0.0362, -0.4492, -0.2370],
            [ 1.3405, -0.0032, -0.2485, -0.3832],
            [-0.4679, -0.4220, -0.1014,  0.2742]], grad_fn=<CopySlices>)
            
Question asked on the slack.