# Analysis of the scores from a training run
import unittest

import numpy as np
import matplotlib.pyplot as plt

filename = "train_scores.npy"

scores = np.load(filename)

moving_average_length = 100


def moving_average(xs, avg_off=moving_average_length):
    """Compute the moving average of the last avg_off scores"""
    cs = np.hstack([0,np.cumsum(xs)])
    diffs = cs[avg_off:] - cs[:-avg_off]   # value change in avg_off steps
    return diffs / avg_off


class TestMovingAverage(unittest.TestCase):
    def test_moving_average(self):
        self.assertTrue(np.array_equal(moving_average([1, 1, 1, 1, 1, 1], 2), np.array([1, 1, 1, 1, 1])))
        self.assertTrue(np.array_equal(moving_average(np.arange(6), 2), np.array([0.5, 1.5, 2.5, 3.5, 4.5])))


moving_average_score = moving_average(scores)
solved = moving_average_score > 13.0

idx_solved = np.where(moving_average_score > 13.0)[0] + 99
idx_not_solved = np.setdiff1d(np.arange(len(scores)-(moving_average_length - 1)), idx_solved)

from matplotlib.pyplot import figure
figure(num=None, figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

plt.scatter(idx_not_solved, scores[idx_not_solved], s=6, c = 'red', label="score before solved")
plt.scatter(idx_solved, scores[idx_solved], s=6, c ='green', label="score after solved")
plt.plot(np.arange(len(moving_average_score))+99, moving_average_score, linewidth=3, color='black')
plt.axvline(x=max(idx_not_solved)+1, linestyle="--")
plt.title("Scores during training")
plt.xlabel("Episode index")
plt.ylabel("Score")
plt.legend()

plt.savefig("scoreplot.png")
plt.show()