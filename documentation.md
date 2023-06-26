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

## Replicating Experiments

There are three experiments. The first is FO-OO-MDP learning on taxi, heist, and prison (`experiments/basic_learning.py`).
Next, logic-based object discovery on all three domains, with different numbers of objects known a-priori
(`experiments/logic_based_object_discovery.py`). Finally, simplest-explanation object discovery on all three domains,
where the source and target domain are the same (`experiments/simplest_explanation_object_discovery.py`). For simplest-explanation
for source domain heist and target domain prison, that has to be run manually with `runners/simplest_explanation/prison_from_heist_simplest_explanation_runner.py`.
There is not a prison_from_taxi, because the implementation currently fails in this case. See the areas for future work for more info.

### Visualizing Results with Plots

I make two types of plots: histograms and box plots. Histograms show the detailed distribution of episode finishing times
(for FO-OO-MDP learning), box plots allow many distributions to be compared with each other (for object transfer).

For histograms, open `plot_experiment_data.py`. Update the HOME_FOLDER and TRAIN_FOLDER variables at the top of the script.
By (un)commenting lines 60-61, you can visualize the plots or simply save them directly to disk.
For each plot you want to make, add a `make_plot` and pass the training run name (the name of the folder saved in `training`),
plot title, and maximum steps (controls x axis size). `experiment_type` controls which subfolder in `training`
to look into.

For box plots, use `box_plot_experiment_data.py`. The general process is the same, but list all experiments you want compared on
the same plot in `experiment_names` at the top of the script. Update the labels for these on line 59.

### Creating GIFs from Runs

To save each frame of a run as a png, currently you must manually update the environment's `draw_world` to:
```python
img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
cv2.imshow("Heist World", img)
cv2.waitKey(delay)

cv2.imwrite(f"visualizations/heist_logic_based/{self.steps:03d}.png", img)

self.steps += 1
```

and add `self.steps = 0` to `__init__()`. You can convert these images to a gif with `ffmpeg`
```
ffmpeg -i [xxx].png -vf palettegen=16 palette.png
ffmpeg -i %03d.png -i palette.png -filter_complex "[0:v][1:v] paletteuse" -framerate 10 out.gif
```

Replace `[xxx]` with a image number that contains all the colors of the gif, to be added to the palette.

## Known Problems and Future Work

Current open github issues include:
* [Searching for experiences with new object names](https://github.com/edf42001/masters-thesis/issues/24): When the agent
  is searching for new experiences (that is, new objects to interact with in new ways), it does not try to map unknown objects
  to known objects. For example, if in the previous environment the agent executed PICKUP on PASSENGER, and in the current
  environment the agent knows that OBJECT1 is PASSENGER, it will still think that PICKUP OBJECT1 is a new experience. Most
  likely to resolve this, one will need to generate all permutations based on the obejct map, then take the "expected value" of the experience
  being a new experience. See the functions around line 173 in `simplest_explanation_policy.py`.
* [State doesn't split based on object identity](https://github.com/edf42001/masters-thesis/issues/14): This relates to the
  main issue with my algorithm and of storing only relavent object attributes in the state array. If the agent picks up
  an unknown object that could be a PASSENGER or KEY, it should produce two daughter states, one where it is holding a passenger
  and one a key. But due to the way the environment handles transitions, only the real identity of the object it is on controls the next state.
  See also [weirdness in how state attributes are handled](https://github.com/edf42001/masters-thesis/issues/23). See also the TODO
  `figure this out assert len(transitions) < 2, "Only 1 transition per state` in `object_transfer_policy.py`.
* [Strings instead of variables](https://github.com/edf42001/masters-thesis/issues/13): I originally had all data stored in strings
  and was doing string split operations to fix it. I fixed the majority of this with the `DeicticReference` class, but there
  still may be some string operations that could be replaced.
  
Other major TODOs include:
* See line 106 of `simplest_explanation_model.py`: `TODO: a cheat for testing purposes`. The agent tests if, an unknown
  object is a brand-new object, that would lead to a simpler explanation. I hardcoded doing this for only the relevant objects,
  otherwise the computational complexity would have slowed down the program. The problem is even if it discovers that a new object
  leads to an equally simple explanation, it keeps that possibility in the object map. This leads to more permutations. Perhaps it should
  prune new objects that don't lead to any difference, assuming that they are old objects until proven otherwise.

The main issue, and the reason simplest explanation learning doesn't work on Taxi world, is the way the state is stored.
Currently, each object has different variables. For example, the state will be \[TAXI_X, TAXI_Y, KEY1_STATE, KEY2_STATE, ...\].
In addition, walls are not stored in this state array at all, because in all previous OO-MDP formulations, walls have been static.
So if the agent imagines a world in which it picks up a lock, in the agent's mind (virtual state) it will set the lock's state to held (0).
But for locks, a (0) means unlocked, so the environment is now not at all in sync with what the agent is trying to do.
The way to fix this, is I think by giving every object the exact same state variables, (including walls). So every object
would have a boolean held, unlocked, exists, variable. That way, if the agent *wants* to try unlocking a wall, it can,
and I won't have to use [this](https://github.com/edf42001/masters-thesis/blob/8105ca15c4159cabe8ac41860fe9145766df1c8b/algorithm/symbolic_domains/simplest_explanation_model.py#L239-L244) hack anymore.

This makes the implementation a lot more like the hit video game Baba Is You, and will of course make the state array much bigger
(especially if every wall needs its own variable, even if 90% of the time their variables never change). There might be a way to store
the state hierarchically, or with objects, to prevent this, but those would be harder to hash and compare. 
