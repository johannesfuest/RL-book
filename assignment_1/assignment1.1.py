import itertools
import sys

from rl.markov_process import NonTerminal
from rl.markov_process import FiniteMarkovRewardProcess
from rl.distribution import Categorical
from rl.distribution import Constant
import matplotlib.pyplot as plt
from rl.gen_utils import plot_funcs
import numpy as np
import sys


snakes_and_ladders = {1: 38, 4: 14, 8: 30, 21: 42, 28: 76, 50: 67, 71: 92, 80: 99, 97: 78, 95: 56, 88: 24, 62: 18,
                      48: 26, 36: 6, 32: 10}
absent_states = {1, 4, 8, 21, 28, 50, 71, 80, 88, 97, 95, 62, 48, 36, 32}
states = set(range(0, 100))
states = states - absent_states
transition_map = {}

for i in states:
    probs = {}
    for j in range(101):
        if j in states and j - i in range(1, 7) and j != 100:
            probs[(j, -1.0)] = 1.0 / 6.0
            continue
        if j - i in range(1, 7) and j in snakes_and_ladders:
            probs[(snakes_and_ladders[j], -1.0)] = 1.0 / 6.0
            continue
        if j == 100 and j - i in range(1, 7):
            probs[(j, -1.0)] = (7.0 - 1.0 * (j - i)) / 6.0
            continue
    transition_map[i] = Categorical(probs)

mdp = FiniteMarkovRewardProcess(transition_map)
traces = mdp.traces(Constant(NonTerminal(0)))
num_samples = 100
samples = [list(itertools.islice(i, 100000)) for i in itertools.islice(traces, num_samples)]
samples_plot_1 = samples[:10]

steps = []
for sample in samples_plot_1:
    x = list(range(len(sample)))
    y = []
    for state in sample:
        y.append(state.state)
    plt.plot(x, y)

plt.xlabel('Dice rolls')
plt.ylabel('Field number')
plt.title('Randomly sampled game traces')
plt.show()

for sample in samples:
    steps.append(len(sample))

plt.hist(steps, bins=50)
plt.xlabel('Number of games')
plt.ylabel('Dice rolls required to end game')
plt.title('Distribution of total dice rolls')
plt.show()

mdp_reward = mdp.get_value_function_vec(1.0)
print(mdp_reward[0] * -1.0)
