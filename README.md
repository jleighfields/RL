# Reinforcement Learning to Control a Robot Arm

This notebook is on based a reinforcement learning assignment in Professor Chuck Anderson's machine learning course (CS545) at Colorado State University and is shared here with his permission. Changes to the original notebook include:

* Neural network implementation in `Pytorch`, including GPU set up
* Augmented state functions to include polar coordinates
* Variable goal states

The Robot class and state functions have been slightly modified to allow for random goal generation, otherwise the `Robot` class and `state` code is left as is from the assignment.



## Overview of problem setup

This notebook walks through a Reinforcement Learning (RL) set up for a simulation of a two-dimensional robot arm with multiple links and joints. A neural network is used to approximate the Q function to predict the reinforcements from each action given a state. The reinforcement is the distance to a goal for the end of the robot's arm, and lower reinforcements are better.

The state of the arm is the angles of each joint. Joint angles are represented with the sine and cosine of the angle, to deal with the discontinuity between 1 and 359 degrees, as the state input to the neural network. The addition of polar coordinates to the state is also tested to see if this helps learning. Valid actions on each step will be $-0.1$, $0$, and $+0.1$ applied to each joint.

The Q function is modeled with neural network built using `Pytorch`, this is refered to as the `Qnet` throughout the notebook. The Pytorch implementation includes flags for running on the `cpu` and `gpu`.  Model architecture is defined in the `MLP` class and the number and size of layers is defined dynamically using inputs to the call to the constructor. Normalization layers are also added by default to help with model training. Traditional data normalization is challenging in RL models since the training data is generated during training and we don't know the means and standard deviations before training. Adding layer normalization seems to have helped in this regard by improving the accuracy and speed of training.

After initial testing of the training functions, a parameter grid search is performed in order to find a good model that can handle a variable goal state. Training results are sorted by the mean reinforcement, R, for the last 20 training trials and the model with the lowest mean reinforcement is selected as the best model. Plots and animations are constructed to visualize training the performance of the combinations of parameters.
