# Object Discovery Thesis Documentation and Usage Guide

The goal of this document is to put into concrete form the knowledge in my head,
so that future researches can understand, replicate, and extend my work.

## Repository Overview

The structure of this repository was founded from Dallan Goldblatt's work on
Hierarchical Object Oriented Markov Decision Processes (https://github.com/rail-cwru/hoomdp). 
Below, I describe the file structure in detail.

* `algorithm`:  Contains `learners` and `models` for each experiment (FO-OO-MDP learning,
  logic-based object transfer, and simplest-explanation object transfer).
  `Learners` run the main "choose action / take step / get reward / update observations" loop by calling the `models`' methods,
  and are all similar to each other. The `models` implement the main bulk of each algorithm
  (except for choosing an action, which is handled by the `policy`).
* `common`: Miscellaneous utilities, most importantly the `data_recorder` which saves data from experiment runs
* `effects`: Code for effect classes (`Increment`, `SetTo`, etc).
* `environment`: All environments follow the base `Environment` class. Each unique environment is implemented here.
* `experiments`: Helper scripts to run experiments on different environments and configurations from one entrypoint.
* `media`: Images and video for the paper and repository
* `policy`: Used by the `models` to choose actions. Could probably be refactored into `algorithm`.
* `runners`: Each individual experiment is a separate runner. A runner is where the model, policy, learner,
  number of experiment trials, steps per episode, episodes, everything is defined.
* `symbolic_stochastic_domains`: My implementation of FO-OO-MDPs. The file names describe what each file contains well.
* `test`: A playground for verifying code works as expected.
* `training`: Experiment data is stored here.
* `visualize_results`: Code to generate graphs/charts from experimental results.
