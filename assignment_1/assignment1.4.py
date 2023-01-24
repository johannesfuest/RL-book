from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.distribution import Choose
from rl.distribution import Categorical
from rl.distribution import Constant
from rl.policy import FinitePolicy
from rl.gen_utils import plot_funcs
import itertools
import numpy as np


def plot_three_traces(
        trace1: np.ndarray,
        trace2: np.ndarray,
        trace3: np.ndarray
) -> None:
    plot_funcs.plot_list_of_curves(
        [range(len(trace1)), range(len(trace2)), range(len(trace3))],
        [trace1, trace2, trace3],
        ["r-", "b--", "g-."],
        ['n = 3', 'n = 6', 'n = 9'],
        'Lilypad Number',
        'Escape probability',
        ''
    )


def optimal_mdp(n):
    states = list(range(1, n))
    actions = [1, 2]

    transition_map = {}

    for state in states:
        probs = {}
        b_tuples = [(0, 0.0)]
        reward = 0.0
        for iter_state in states:
            b_tuples.append((iter_state, reward))
        b_tuples.append((n, 1.0))
        a = {}
        if state == 1:
            a[(state - 1, 0.0)] = (state * 1.0) / (n * 1.0)
            a[(state + 1, 0.0)] = ((n * 1.0) - (state * 1.0)) / (n * 1.0)
        elif state == n - 1:
            a[(state - 1, 0.0)] = (state * 1.0) / (n * 1.0)
            a[(state + 1, 1.0)] = ((n * 1.0) - (state * 1.0)) / (n * 1.0)
        else:
            a[(state - 1, 0.0)] = (state * 1.0) / (n * 1.0)
            a[(state + 1, 0.0)] = ((n * 1.0) - (state * 1.0)) / (n * 1.0)
        probs[1] = Categorical(a)
        probs[2] = Choose(b_tuples)
        transition_map[state] = probs

    mdp = FiniteMarkovDecisionProcess(transition_map)
    giga_list = list(itertools.product(actions, repeat=n-1))
    policies = []
    mrps = []
    for list_element in giga_list:
        policy_map = {}
        for i in range(n - 1):
            policy_map[i + 1] = Constant(list_element[i])
        current_policy = FinitePolicy(policy_map)
        policies.append(current_policy)
        mrps.append(mdp.apply_finite_policy(current_policy))

    value_vecs = []
    max_prob = 0
    optimal_policy = None
    optimal_mrp = None
    for i in range(len(mrps)):
        value_vecs.append(mrps[i].get_value_function_vec(1.0))
        if np.sum(mrps[i].get_value_function_vec(1.0)) > max_prob:
            max_prob = np.sum(mrps[i].get_value_function_vec(1.0))
            optimal_policy = policies[i]
            optimal_mrp = mrps[i].get_value_function_vec(1.0)
    print(optimal_mrp)
    print(optimal_policy)
    return optimal_mrp, optimal_policy


n_list = [3, 6, 9]
traces = []
for n in n_list:
    mrp_values, policy = optimal_mdp(n)
    mrp_values = np.append(mrp_values, 1.0)
    mrp_values = np.insert(mrp_values, 0, 0.0)
    traces.append(mrp_values)

plot_three_traces(traces[0], traces[1], traces[2])









