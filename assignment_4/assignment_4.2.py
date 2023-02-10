from rl.function_approx import LinearFunctionApprox, Dynamic
from typing import Tuple, Iterable
from rl.distribution import Distribution, Constant, Gaussian
from rl.markov_decision_process import MarkovDecisionProcess, S, A, State
from rl.markov_process import NonTerminal, Terminal
from rl.function_approx import FunctionApprox, AdamGradient
from rl.approximate_dynamic_programming import back_opt_vf_and_policy
from rl.iterate import last

ValueFunctionApprox = FunctionApprox[NonTerminal[float]]
NTStateDistribution = Distribution[NonTerminal[S]]


class AmericanMarkovProcess(MarkovDecisionProcess[NonTerminal[float], bool]):
    """
    A simple MDP for an Option stopping problem. The state is the price of the underlying asset, there are two actions,
    to continue or to stop. The reward is for stopping is the payoff of the option at the current state, while there is
    zero reward for continuing. The transition function goes to a terminal state for stopping. For continuing, the
    transition function is a Gaussian distribution with mean equal to the current state times the 1 + the passed in
    return and variance according to the passed in variance.
    """
    def __init__(self, is_call: bool, strike: float, ret: float, var: float):
        if is_call:
            self.payoff = lambda x: max(x - strike, 0)
        else:
            self.payoff = lambda x: max(strike - x, 0)
        self.ret = ret
        self.var = var

    def actions(self, state: State[float]) -> Iterable[bool]:
        return [True, False]

    def step(self, state: NonTerminal[float], action: bool) -> Distribution[Tuple[State[float], float]]:
        if action:
            return Constant((Terminal(0.0), self.payoff(state.state)))
        else:
            return Gaussian(state.state * (1.0 + self.ret), self.var).map(lambda x: (NonTerminal(x), 0.0))


if __name__ == '__main__':

    # Set up basic parameters of the problem
    start_price = 100.0
    strike_price = 100.0
    call = True
    time_step_return = 0.07
    time_step_variance = 5
    t = 10
    discount = 1
    if call:
        pay = lambda x: max(x - strike_price, 0)
    else:
        pay = lambda x: max(strike_price - x, 0)

    # Set up the distribution of the underlying asset at time t
    final_mean = (1.0 + time_step_return)**t * start_price
    final_var = time_step_variance * t
    base_distribution = Gaussian(final_mean, final_var).map(lambda x: (NonTerminal(x)))

    # Set up the function approximator
    approx_0 = LinearFunctionApprox.create(
        [lambda x: 0.0001, lambda x: x.state]
    )

    # Set up the MDP
    mdp = AmericanMarkovProcess(call, strike_price, time_step_return, time_step_variance)
    mdp_f0_mu_triples = [(mdp, approx_0, base_distribution)]

    # Set up and execute backward induction
    current_mean = final_mean
    current_var = final_var
    for i in range(0, t + 1):
        print('Iteration: ' + str(i))
        print('Current mean: ' + str(current_mean) + ', current variance: ' + str(current_var))
        it = back_opt_vf_and_policy(mdp_f0_mu_triples, discount, 100, 0.1)
        vf, policy = next(it)
        current_mean = current_mean / (1.0 + time_step_return)
        current_var = current_var - time_step_variance
        mdp_f0_mu_triples.append((mdp, vf, Gaussian(current_mean, current_var).map(lambda x: (NonTerminal(x)))))
        # Print the value function at time 0
        print('Optimal policy at t - ' + str(i) + ': ')
        print(policy.act(NonTerminal(current_mean)))
        print('Price of the option at t - ' + str(i) + ': ')
        print(vf(NonTerminal(100.0)))
