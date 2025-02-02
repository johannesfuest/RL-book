from typing import Iterable, Mapping, Sequence, Tuple, TypeVar, Union, Iterator
import itertools
import numpy as np
import rl.markov_process as mp
import rl.markov_decision_process as mdp
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from rl.function_approx import Tabular
from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.distribution import Choose
from rl.returns import returns

# Collaborated with Nabil Ahmed and Spencer Siegel

S = TypeVar('S')


def tabular_mc_prediction(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        γ: float,
        episode_length_tolerance: float = 1e-6,
) -> Iterator[Mapping[S, float]]:
    '''traces is a finite iterable'''
    episodes: Iterator[Iterator[mp.ReturnStep[S]]] = \
        (returns(trace, γ, episode_length_tolerance) for trace in traces)
    count_dict = {}
    mean_dict = {}
    for i, ep in enumerate(episodes):
        for step in ep:
            if step.state not in mean_dict:
                mean_dict[step.state] = 0
                count_dict[step.state] = 0

            if count_dict[step.state] == 0:
                mean_dict[step.state] = step.return_
            else:
                mean_dict[step.state] += (step.return_ - mean_dict[step.state]) / (count_dict[step.state] + 1)
            count_dict[step.state] += 1
        yield mean_dict


def tabular_td_prediction(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        γ: float,
        episode_length_tolerance: float = 1e-6,
        initial_value: float = 0.0
) -> Iterator[Mapping[S, float]]:
    '''traces is a finite iterable'''
    episodes: Iterator[Iterator[mp.ReturnStep[S]]] = \
        (returns(trace, γ, episode_length_tolerance) for trace in traces)
    count_dict = {}
    mean_dict = {}
    for i, ep in enumerate(episodes):
        for step in ep:
            if step.state not in mean_dict:
                mean_dict[step.state] = 0
                count_dict[step.state] = 0

            observed_val = step.reward + γ * mean_dict.get(step.next_state, initial_value)
            if count_dict[step.state] == 0:
                mean_dict[step.state] = observed_val
            else:
                mean_dict[step.state] += (observed_val - mean_dict[step.state]) / (count_dict[step.state] + 1)
            count_dict[step.state] += 1
        yield mean_dict


if __name__ == "__main__":
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mrp = SimpleInventoryMRPFinite(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )
    traces = si_mrp.reward_traces(Choose(si_mrp.non_terminal_states))
    mc_pred = tabular_mc_prediction(traces, γ=user_gamma)
    td_pred = tabular_td_prediction(traces, γ=user_gamma, initial_value=-30)
    print("MC Prediction after 1000 steps:")
    print([i for i in itertools.islice(mc_pred, 1000)][-1])
    print("TD Prediction after 1000 steps:")
    print([i for i in itertools.islice(td_pred, 1000)][-1])
