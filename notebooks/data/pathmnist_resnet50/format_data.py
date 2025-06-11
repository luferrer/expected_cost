
import numpy as np
import sys
from medmnist import PathMNIST
from IPython import embed

pred_file = sys.argv[1]

test_dataset = PathMNIST(split="test", download=True)
targets = test_dataset.labels.squeeze()

# Read a CSV file without header
posts = np.genfromtxt(pred_file, delimiter=',')[:, 1:]
posts[np.where(posts == 0)] = 1e-20
logposts = np.log(posts)

np.save("scores.npy", logposts)
np.save("targets.npy", targets)
