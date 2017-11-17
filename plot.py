import matplotlib.pyplot as plt
import os
import json
import numpy as np

def load_single(filename):
    """
    loads and returns a single experiment stored in filename
    returns None if file does not exist
    """
    if not os.path.exists(filename):
        return None
    with open(filename) as f:
        result = json.load(f)
    return result



style = {
    "linewidth": 2, "alpha": .7, "linestyle": "-", "markersize": 7}

file = load_single('./MNIST_Results/001-results.json')

x = np.arange(len(file['accuracy']))
y_mean = np.asarray(file['accuracy'])
y_sem = np.asarray(file['accuracy_std'])
y_mean *= 100.
y_sem *= 100.
fig = plt.figure()
plt.plot(x, y_mean, **style)

min_ = np.inf
max_ = - np.inf
plt.fill_between(x, y_mean - y_sem, y_mean + y_sem, alpha=.3)
max_ = max(np.max(y_mean + y_sem), max_)
min_ = min(np.min(y_mean - y_sem), min_)
# adjust visible space
y_lim = [min_ - .1 * abs(max_ - min_), max_ + .1 * abs(max_ - min_)]
if min_ != max_:
    plt.ylim(y_lim)

plt.xlabel("Training Epochs", fontsize=16)
plt.ylabel("Accuracy [%]", fontsize=16)
plt.grid(True)

plt.show()

