from rl.markov_process import FiniteMarkovRewardProcess
from rl.distribution import Categorical

states = list((range(0, 11)))
transition_map = {}

for i in states:
    probs = {}
    for j in range(len(states)):
        if j > i:
            probs[(j, -1.0)] = 1.0 / (1.0 * len(states) - i)
    transition_map[i] = Categorical(probs)
mdp = FiniteMarkovRewardProcess(transition_map)
mdp_reward = mdp.get_value_function_vec(1.0)
print(mdp_reward[0] * -1.0)

