import copy

import numpy as np
from typing import (DefaultDict, Dict, Iterable, Generic, Mapping,
                    Tuple, Sequence, TypeVar, Set)


def solve_custom_mdp(wages: Iterable[float], probs: Iterable[float], alpha: float, discount: float, threshold: float) \
        -> Tuple[Mapping[str, float], Mapping[str, str]]:
    """
    @param wages: iterable containing wages for unemployment and jobs 1 to N.
    @param probs: iterable of probabilities of receiving job offers for jobs 1 to N. Must sum to one.
    @param alpha: float describing the probability of job loss at the end of the day.
    @param discount: float describing the discount rate on future earnings
    @param threshold: a float describing the convergence threshold of the value iteration algorithm
    @returns: tuple containing an iterable describing the best action at each state, as well as the resulting optimal
              value function in terms of expected discounted sum of wages utility.
    """
    wages = list(wages)
    probs = list(probs)
    # Check that probabilities sum to one
    if not np.isclose(sum(probs), 1.0):
        raise ValueError("Supplied job offer probabilities must sum to one!")

    # Check discount factor is between 0 and one
    if not 0.0 <= discount and 1.0 > discount:
        raise ValueError("Discount factor must be within [0, 1)!")

    # Check that there is at least one job.
    if not len(wages) >= 2:
        raise ValueError("Please supply wages for unemployment and at least one job!")

    # Check that the number of wages is equal to the number of jobs + 1
    if not len(wages) == len(probs) + 1:
        raise ValueError("Please supply exactly one wage per job plus an unemployment wage at position 0!")

    # Check that wages are positive
    for wage in wages:
        if wage < 0.0:
            raise ValueError("All wages must be greater than  or equal to zero!")

    # Check that alpha is between zero and one
    if not alpha >= 0.0 and alpha <= 1.0:
        raise ValueError("Probability of job loss alpha must be within [0, 1]!")

    # Build state space
    S = []
    n_jobs = len(probs)
    for i in range(1, n_jobs + 1):
        S.append("U" + str(i))
    for j in range(1, n_jobs + 1):
        S.append("E" + str(j))

    # Build action space
    A = ["accept", "reject"]

    # Build transition probability function P
    def transition_prob(state_from: str, action: str, state_to: str) -> float:
        """
        @param state_from: string describing the current state
        @param action: the action chosen
        @param state_to: the potential future state
        @returns: a tuple containing the probability of this transition, as well as the reward
        """
        unemployed = state_from.startswith("U")
        job_number = int(state_from[1])
        future_unemployed = state_to.startswith("U")
        future_job_number = int(state_to[1])
        if unemployed:
            if action == "accept":
                if future_unemployed:
                    prob = alpha * probs[future_job_number - 1]
                else:
                    if future_job_number == job_number:
                        prob = 1.0 - alpha
                    else:
                        prob = 0.0
            else:
                if future_unemployed:
                    prob = probs[future_job_number - 1]
                else:
                    prob = 0.0
        else:
            if future_unemployed:
                prob = alpha * probs[future_job_number - 1]
            else:
                if job_number == future_job_number:
                    prob = 1 - alpha
                else:
                    prob = 0.0
        return prob

    # Build reward transition function R
    def reward_transition_func(current_state: str, current_action: str) -> float:
        """
        @param current_state: a string describing the current state
        @param current_action: a string describing the action taken
        @return: a float describing the utility earned from this state action pair
        """
        job_number = int(current_state[1])
        if current_action == "accept":
            return np.log(wages[job_number])
        else:
            if current_state.startswith("E"):
                return np.log(wages[job_number])
            else:
                return np.log(wages[0])

    # Define bellman operator B* for value iteration algorithm
    def bellman_operator(state_values: Mapping[str, float]) -> Mapping[str, float]:
        V_new = {}
        for current_state in state_values:
            options = []
            for current_action in A:
                total = reward_transition_func(current_state, current_action)
                for future_state in S:
                    total += discount * transition_prob(current_state, current_action, future_state) * state_values[future_state]
                options.append(total)
            V_new[current_state] = max(options)
        return V_new

    # Define function for convergence
    def converged(v_0: Mapping[str, float], v_1: Mapping[str, float]) -> bool:
        """
        Function that checks if the value iteration algorithm has converged
        @param v_0: Mapping from state strings to floats, describing their value at iteration t - 1
        @param v_1: Mapping from state strings to floats, describing their value at iteration t
        @return: a boolean of whether the maximum absolute distance between the values for two
        """
        max_dist = 0.0
        for s in S:
            if abs(v_0[s] - v_1[s]) > max_dist:
                max_dist = abs(V_0[s] - V_1[s])
        if max_dist < threshold:
            return True
        else:
            return False

    # Conduct value iteration algorithm
    V_0 = {}
    for s in S:
        V_0[s] = 0.0
    convergence = False
    while not convergence:
        V_1 = bellman_operator(V_0)
        convergence = converged(V_0, V_1)
        V_0 = V_1

    # Extract optimal policy from optimal values
    optimal_policy = {}
    for state in S:
        action_values = []
        for action in A:
            expectation = reward_transition_func(state, action)
            for future_state in S:
                expectation += discount * transition_prob(state, action, future_state) * V_0[future_state]
            action_values.append(expectation)
        if action_values[0] > action_values[1]:
            optimal_policy[state] = "accept"
        else:
            optimal_policy[state] = "reject"

    return V_0, optimal_policy


if __name__ == '__main__':
    p = [0.5, 0.3, 0.1, 0.09, 0.01]
    w = [0.2, 1.6, 2, 3, 4, 5]
    a = 0.000000001
    d = 0.5
    c = 0.0001
    val, opt = solve_custom_mdp(w, p, a, d, c)
    print(val)
    print(opt)

