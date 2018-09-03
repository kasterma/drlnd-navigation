# Notes

Lab notebook to keep track of what was done; to be used as a basis for the report.

- DONE: set up environment in repeatable way (so can easily move to different computer if need be)
- DONE: set up interact.py to check the environment, and learn how to interact with it
- DONE: set up train.py, agent.py, and model.py as a framework for the training
- DONE: fill in model.py and do some testing with it
- DONE: fill in agent.py and do some testing with it
- DONE: fill in train.py and do some testing with it
- train.py version in a notebook to get graphs while learning
- how to test easily how much difference a GPU can make?
- how to test our use of the GPU?
- in the lunar lander problem the network that worked well was more complicated than expected; same here?
- DONE: understand the failing unittest in model.py
- extend the tests for the neural network (i.e. get a default lists of tests to run against a network to see
  basic implementation is not faulty)

## Architecture

For starters try similar archtecture as worked for the lunar lander: two hidden layers, relu in between.

Make variable:
- size of the hidden layers
- POSTPONED: number of hidden layers

Note: unclear how to make nn.Module extension dynamic in the number of layers.  Do that later.

In the end the model is essentially that from the lunar lander exercise

## Agent

Since one of the future things to try is to try prioritized experience replay we'll wrap the memory for replays in
a class.  Note that in the lunar lander the ReplayBuffer seems to store the experiences as numpy data and then
convert it to torch when remembering.  We have verified witn interact.py that indeed the environment returns its
state as a numpy array.

In the example network for the lunar lander there is a silly trick for getting the same network twice; there we create
a model, set the random seed back to the same value and generate a model again.  This results in twice the same model.
However it is playing with the random seed that hardly is elegant.  Creating a new model of the same shape, and then
copying over the data is more elegant.

Bad design in needing to pass the full config to the memory module.

## Training

The paper we are basing this on used experience replay, and a distinct less frequently updated target network.  Try
the same.  There we updated the target network (soft update) after every batch of experiences was run through.

In lunar lander used Adam, try the same here.

- also try basic SGD optim.SGD(net.parameters(), lr=0.001)
- also try SGD with momentum optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
- also try with RMSProp
- try different learning rates
- try different sizes of the replay buffer; we expect the initial experience not to be very useful b/c it is too random.
  When does it start becoming useful?
  
We are also planning to try different decay strategies for epsilon.  The example implements exponential decay, we also
want to try linear decay up to a constant.

The last type error took a long time to find and fix, though in retrospect it is pretty clear.  In agent.get_action
there are two paths to get an action.  And they returned different types.  The unity environment wasn't able to
handle that different, adding a conversion to int solved the issue.

First training run without type errors:

    $   make train
    $ENVIRON_URL is https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip
    $ENVIRON_FILE is Banana.app.zip
    $ENRIFON_DIR is Banana.app
    virtual env is venv
    (source venv/bin/activate; python train.py; )
    2018-09-02 13:37:13,620 - train - INFO - Starting to train
    2018-09-02 13:37:13,623 - train - INFO - Using config: {'environment': 'Banana.app', 'network_spec': {'input_dim': 37, 'hidden_1_size': 64, 'activation_1': 'relu', 'hidden_2_size': 64, 'activation_2': 'relu', 'output_dim': 4}, 'experience_memory': {'size': 100000}, 'train': {'batch_size': 64, 'update_every': 4, 'tau': 0.001, 'gamma': 0.99, 'learning_rate': 0.0005}, 'scores_filename': 'train_scores'}
    Mono path[0] = '/Users/kasterma/projects/drlnd-navigation/Banana.app/Contents/Resources/Data/Managed'
    Mono config path = '/Users/kasterma/projects/drlnd-navigation/Banana.app/Contents/MonoBleedingEdge/etc'
    2018-09-02 13:38:10,994 - train - INFO - After 50 scores have mean in last 100 of 0.32
    2018-09-02 13:39:09,599 - train - INFO - After 100 scores have mean in last 100 of 0.86
    2018-09-02 13:40:03,182 - train - INFO - After 150 scores have mean in last 100 of 2.45
    2018-09-02 13:40:56,936 - train - INFO - After 200 scores have mean in last 100 of 4.4
    2018-09-02 13:41:52,027 - train - INFO - After 250 scores have mean in last 100 of 6.01
    2018-09-02 13:42:46,937 - train - INFO - After 300 scores have mean in last 100 of 7.26
    2018-09-02 13:43:42,171 - train - INFO - After 350 scores have mean in last 100 of 8.87
    2018-09-02 13:44:37,331 - train - INFO - After 400 scores have mean in last 100 of 10.16
    2018-09-02 13:45:32,838 - train - INFO - After 450 scores have mean in last 100 of 11.05
    2018-09-02 13:46:28,101 - train - INFO - After 500 scores have mean in last 100 of 12.15
    2018-09-02 13:47:23,606 - train - INFO - After 550 scores have mean in last 100 of 12.39
    2018-09-02 13:48:18,941 - train - INFO - After 600 scores have mean in last 100 of 12.37
    2018-09-02 13:49:16,037 - train - INFO - After 650 scores have mean in last 100 of 13.24
    2018-09-02 13:50:12,662 - train - INFO - After 700 scores have mean in last 100 of 14.22
    2018-09-02 13:51:08,768 - train - INFO - After 750 scores have mean in last 100 of 14.81
    2018-09-02 13:52:04,973 - train - INFO - After 800 scores have mean in last 100 of 14.85
    2018-09-02 13:53:01,400 - train - INFO - After 850 scores have mean in last 100 of 15.12
    2018-09-02 13:53:57,599 - train - INFO - After 900 scores have mean in last 100 of 15.63
    2018-09-02 13:54:53,709 - train - INFO - After 950 scores have mean in last 100 of 15.9
    2018-09-02 13:55:50,291 - train - INFO - After 1000 scores have mean in last 100 of 16.14
    2018-09-02 13:56:46,492 - train - INFO - After 1050 scores have mean in last 100 of 16.28
    2018-09-02 13:57:42,484 - train - INFO - After 1100 scores have mean in last 100 of 16.01
    2018-09-02 13:58:38,042 - train - INFO - After 1150 scores have mean in last 100 of 15.2
    2018-09-02 13:59:34,099 - train - INFO - After 1200 scores have mean in last 100 of 15.32
    2018-09-02 14:00:30,804 - train - INFO - After 1250 scores have mean in last 100 of 15.96
    2018-09-02 14:01:27,257 - train - INFO - After 1300 scores have mean in last 100 of 15.99
    2018-09-02 14:02:24,112 - train - INFO - After 1350 scores have mean in last 100 of 15.72
    2018-09-02 14:03:20,788 - train - INFO - After 1400 scores have mean in last 100 of 16.06
    2018-09-02 14:04:19,571 - train - INFO - After 1450 scores have mean in last 100 of 16.05
    2018-09-02 14:05:21,535 - train - INFO - After 1500 scores have mean in last 100 of 15.47
    2018-09-02 14:06:24,013 - train - INFO - After 1550 scores have mean in last 100 of 15.84
    2018-09-02 14:07:27,489 - train - INFO - After 1600 scores have mean in last 100 of 15.76
    2018-09-02 14:08:30,095 - train - INFO - After 1650 scores have mean in last 100 of 15.35
    2018-09-02 14:09:35,864 - train - INFO - After 1700 scores have mean in last 100 of 14.85
    2018-09-02 14:10:37,680 - train - INFO - After 1750 scores have mean in last 100 of 14.29
    2018-09-02 14:11:39,343 - train - INFO - After 1800 scores have mean in last 100 of 15.1
    2018-09-02 14:12:40,440 - train - INFO - After 1850 scores have mean in last 100 of 15.93
    2018-09-02 14:13:43,792 - train - INFO - After 1900 scores have mean in last 100 of 16.06
    2018-09-02 14:14:44,709 - train - INFO - After 1950 scores have mean in last 100 of 15.7
    2018-09-02 14:15:47,432 - train - INFO - After 2000 scores have mean in last 100 of 15.26
    2018-09-02 14:15:47,432 - train - INFO - Done training, saving scores and model
    Traceback (most recent call last):
      File "train.py", line 199, in <module>
        t.train()
      File "train.py", line 89, in train
        self.agent.save()
    AttributeError: 'Agent' object has no attribute 'save'
    make: *** [train] Error 1

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

Problem was that we were using ReLU activation, and the the activations landed in the contant zero part.  By using
tanh activations the problem was resolved.