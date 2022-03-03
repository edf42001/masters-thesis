import numpy as np

data = np.loadtxt("movement_data.csv", delimiter=",")

print("Data + length")
print(data, len(data))

# Data where the action was up
data = data[data[:, 0] == 0]
print(data, len(data))

# Data where the action was up and there was no wall to the left or right or up
data = data[np.all(data[:, [1, 2, 4]] == 0, axis=1)]
# data = data[np.all(data[:, [3]] == 1, axis=1)]
print(data, len(data))

hist = np.histogram(data[:, -1], density=True, bins=3)
print(hist)
