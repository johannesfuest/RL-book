import copy
from typing import Iterator, Tuple, TypeVar, Callable, Iterable, Mapping
import random
import rl
from rl.distribution import Distribution, Categorical
from rl.function_approx import FunctionApprox
from rl.iterate import iterate
from rl.markov_decision_process import FiniteMarkovDecisionProcess, MarkovDecisionProcess, NonTerminal, Terminal
from rl.policy import Policy, DeterministicPolicy
from rl.dynamic_programming import policy_iteration
from rl.approximate_dynamic_programming import value_iteration, evaluate_mrp
from rl.function_approx import Tabular, Dynamic
import numpy as np

S = TypeVar('S')
A = TypeVar('A')

ValueFunctionApprox = FunctionApprox[NonTerminal[S]]
QValueFunctionApprox = FunctionApprox[Tuple[NonTerminal[S], A]]
NTStateDistribution = Distribution[NonTerminal[S]]


def approximate_policy_iteration(
        mdp: MarkovDecisionProcess[S, A],
        y: float,
        approx_0: ValueFunctionApprox[S],
        non_terminal_states_distribution: NTStateDistribution[S],
        num_state_samples: int
) -> Iterator[Tuple[ValueFunctionApprox[S], Policy[S, A]]]:
    def update(v: Tuple[ValueFunctionApprox[S], Policy[S, A]]) -> Tuple[
                ValueFunctionApprox[S], DeterministicPolicy[S, A]]:

        # Create mrp from mdp
        mrp = mdp.apply_policy(v[1])
        # Policy evaluation
        it = evaluate_mrp(mrp, y, v[0], non_terminal_states_distribution, num_state_samples)
        V_0 = next(it)
        V_1 = next(it)
        while not V_0.within(V_1, 0.01):
            V_0 = V_1
            V_1 = next(it)

        # Policy improvement
        def greedy_policy_from_vf(
                v_1: ValueFunctionApprox[S],
                actions: Callable[[NonTerminal[S]], Iterable[A]]
        ) -> DeterministicPolicy[S, A]:
            def optimal_action(s: S) -> A:
                opt_action = None
                max_value = -np.inf
                for a in actions(NonTerminal(s)):
                    r = mdp.step(NonTerminal(s), a)
                    # sample transitions from r and compute expectation
                    # Doing expectation here for simplicity (normally sample from r and take average)
                    value = r.expectation(lambda x: x[1] + y * v_1(x[0]))

                    if value > max_value:
                        max_value = value
                        opt_action = a
                return opt_action

            return DeterministicPolicy(optimal_action)

        new_pi = greedy_policy_from_vf(V_1, mdp.actions)
        return V_1, new_pi

    # initialize policy that chooses a random action at each state
    def choose_first(s: S) -> A:
        return list(mdp.actions(NonTerminal(s)))[0]

    pi_0 = DeterministicPolicy(choose_first)
    return iterate(update, (approx_0, pi_0))


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    # Create a mapping for the MDP with 5 states
    mdp_mapping = {}
    # Create a Markov Decision Process with 5 states and 3 actions for each state
    for i in range(5):
        state_mapping = {}
        # For every action
        for j in range(3):
            # For every state including a terminal state generate a probability and reward
            distr_dict = {}
            probs = np.random.dirichlet(np.ones(6), size=1)[0]
            for k in range(6):
                outcome = (k, 10 * random.random())
                distr_dict[outcome] = probs[k]

            # Initialize FiniteDistribution for Action
            distr = Categorical(distr_dict)
            state_mapping[j] = distr
        mdp_mapping[i] = state_mapping

    # Create the MDP
    mdp_test = FiniteMarkovDecisionProcess(mdp_mapping)
    gamma = 0.5
    values_map = {}
    for i in range(5):
        values_map[NonTerminal(i)] = 0.0
    values_map[Terminal(5)] = 0.0
    approx_init = Dynamic(values_map)
    non_terminal_states_dist = Categorical({s: 1 / 5 for s in mdp_test.non_terminal_states})
    num_states = 3

    # Run the policy iteration
    approx_vf_pol = approximate_policy_iteration(mdp_test, gamma, approx_init, non_terminal_states_dist, num_states)

    # Run the value iteration
    approx_vf_val = value_iteration(mdp_test, gamma, approx_init, non_terminal_states_dist, num_states)

    print('starting all algorithms')
    # Check that the value functions are the same
    app_pol_past = next(approx_vf_pol)
    app_val_past = next(approx_vf_val)
    pol_it_current = next(approx_vf_pol)
    val_it_current = next(approx_vf_val)

    print('starting approximate policy iteration')
    # Run approximate policy iteration
    i = 1
    while not pol_it_current[0].within(app_pol_past, 1e-3):
        app_pol_past = copy.deepcopy(pol_it_current[0])
        i += 1
        pol_it_current = next(approx_vf_pol)

    print('approximate policy iteration complete')

    # Run approximate value iteration
    i = 1
    while not val_it_current.within(app_val_past, 1e-3):
        app_val_past = copy.deepcopy(val_it_current)
        val_it_current = next(approx_vf_val)

    # Run policy iteration
    pol_it = policy_iteration(mdp_test, gamma)
    pol_n_past = next(pol_it)
    pol_n_current = next(pol_it)
    i = 1

    print('approximate value iteration complete')

    # Define a function to check for convergence
    def converged(a: Mapping[S, float], b: Mapping[S, float], tol: float) -> bool:
        for s in a.keys():
            if abs(a[s] - b[s]) > tol:
                return False
        return True


    # Run policy iteration
    while not converged(pol_n_current[0], pol_n_past[0], 1e-3):
        pol_n_past = copy.deepcopy(pol_n_current)
        i += 1
        pol_n_current = next(pol_it)

    print('policy iteration complete')

    # Run value iteration
    val_it = rl.dynamic_programming.value_iteration(mdp_test, gamma)
    val_n_past = next(val_it)
    val_n_current = next(val_it)
    i = 1
    while not converged(val_n_current, val_n_past, 1e-3):
        val_n_past = copy.deepcopy(val_n_current)
        i += 1
        val_n_current = next(val_it)

    print('value iteration complete')

    # Print the final value functions for each algorithm
    print('Approximate policy iteration: ' + str(pol_it_current[0]))
    print('Approximate value iteration: ' + str(val_it_current))
    print('Policy iteration: ' + str(pol_n_current[0]))
    print('Value iteration: ' + str(val_n_current))
