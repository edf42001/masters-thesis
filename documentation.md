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
