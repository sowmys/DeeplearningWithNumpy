A reproduction of Joel Grus Live session
========================================
I watched Joel Gru's live session and tried to mimic it from memory. I am not a python expert but his session is very
inspiring and help structure a simple deep learning project into components.

```
                               [Training Inputs], [Expected Output]
                                               +
                                               |
                                               v
                          +----------------------------------------+
                          |                                        |
     [Batched,            |         [Batch Size], [Shuffle]        |
   + Shuffled   <---------+                                        +--------------->  [Batched, Shuffled
   | Inputs]              |         TRAINING DATA ITERATOR         |                  Expected Outputs]+-----+
   |                      +----------------------------------------+                                         |
   |                                                                                                         |
   |  +---------------------+       +---------------------+         +---------------------+                  |
   |  |  +---------------+  |       |  +---------------+  |         |  +---------------+  |                  |
   |  |  |               |  |       |  |               |  |         |  |               |  |    [Predicted    |
   +---->+ Forward Prop  +------------>+ Forward Prop  +-------------->+ Forward Prop  +------> Output]      |
      |  |               |  |       |  |               |  |         |  |               |  |          +       |
      |  +------^--------+  |       |  +------^--------+  |         |  +-------^-------+  |          |       |
      |         |           |       |         |           |         |          |          |      +---v-------v-+
      |         +           |       |         |           |         |          +          |      |             |
      | [Learned Parameters]|       |         |           |         | [Learned Parameters]|      | +---------+ |
      |         ^           |       |         |           |         |          ^          |      | |Calc Loss|-----> [Loss]
      | +-----------------+ |       | +-----------------+ |         | +-----------------+ |      | |for Output |
      | |                 | |       | |                 | |         | |                 | |      | +---------+ |
      | | [Learning Rate] | |       | | [Learning Rate] | |         | | [Learning Rate] | |      | +---------+ |
      | |                 | |       | |                 | |         | |                 | |      | |Calc Loss| |
      | |   OPTIMIZER     | |       | |   OPTIMIZER     | |         | |   OPTIMIZER     | |  +-----+Gradient | |
      | +-------^---------+ |       | +-------^---------+ |         | +--------^--------+ |  |   | +---------+ |
      |         +           |       |         |           |         |          +          |  |   |             |
      | [Computed Gradients]|       |         |           |         | [Computed Gradients]|  |   | LOSS FUNC   |
      |         ^           |       |         |           |         |          ^          |  |   +-------------+
      |         |           |       |         |           |         |          |          |  |
      |  +---------------+  |       |  +---------------+  |         |  +---------------+  |  |
      |  |               |  |       |  |               |  |         |  |               +<----+
      |  | Backward-prop +<------------+ Backward-prop +<--------------+ Backward-prop |  |
      |  |               |  |       |  |               |  |         |  |               |  |
      |  +---------------+  |       |  +---------------+  |         |  +---------------+  |
      |   LINEAR LAYER 1    |       |   ACTIVATION LAYER  |         |   LINEAR LAYER 2    |
      +---------------------+       +---------------------+         +---------------------+






```

+ <b>Neural Net Model</b>: Abstracts a neural network by encapsulating a collection of layers. 
+ <b>Layer</b>: Abstracts a layer which encapsulates weights, bias. It also caches the gradients computed during back prop.
+ <b>Training Data Iterator</b>: Abstracts iteration of the input and the expected output.
+ <b>Loss</b>: Abstracts loss function.
+ <b>Optimizer</b>: Abstracts the component that updates the weights and biases based on gradients computed during back prop.
