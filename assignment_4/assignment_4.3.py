import random
from typing import Tuple, Iterable
from functools import partial
from rl.chapter9.order_book import DollarsAndShares, OrderBook, PriceSizePairs
from rl.distribution import Distribution, Constant, Categorical, SampledDistribution
from rl.markov_process import MarkovProcess, NonTerminal, State, S
from rl.markov_process import NonTerminal, State
import numpy as np
from numpy.random import poisson


class OrderbookNormal(MarkovProcess):
    """
    A simple MDP for a limit order book. The state is the order book. The transaction
    function gives equal probability to each of the four possible transactions. The number of shares in each
    transaction is chosen randomly from 1 to 10 for limit orders and from 1 to 2 for market orders.
    The price of limit orders is chosen randomly from a normal distribution with mean 10% above or below the current
    best bid or ask price and standard deviation 10% of the current best bid or ask price.
    """

    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        def next_state(st: OrderBook) -> NonTerminal[OrderBook]:
            try:
                price_buy = (int) (np.random.normal(1.0 * st.bid_price(), 0.1 * st.bid_price()))
                price_sell = (int) (np.random.normal(1.0 * st.ask_price(), 0.1 * st.ask_price()))
            except IndexError:
                price_buy = np.random.normal(1.05 * 8, 0.8)
                price_sell = np.random.normal(0.5 * 12, 1.2)
            m = random.randint(0, 4)
            if m == 0:
                return NonTerminal(st.sell_limit_order(price_sell, random.randint(1, 10))[1])
            elif m == 1:
                return NonTerminal(st.buy_limit_order(price_buy, random.randint(1, 10))[1])
            elif m == 2:
                return NonTerminal(st.sell_market_order(random.randint(1, 2))[1])
            elif m == 3:
                return NonTerminal(st.buy_market_order(random.randint(1, 2))[1])
            else:
                return NonTerminal(st)
        fun = partial(next_state, state.state)
        return SampledDistribution(fun, expectation_samples=10)


class RareBigFish(MarkovProcess):
    """
    This MDP for an orderbook simulates a market where very rarely a large order comes in. The state is the order book.
    The transaction function gives equal probability to each of the four possible transactions, however, the number of
    shares traded will normally be quite small, but occasionally a very large market order will come in. The price of
    limit orders is chosen randomly from a normal distribution with mean 10% blow or above the current best bid or ask
    price and standard deviation 10% of the current best bid or ask price, meaning that small sellers and buyers will
    rarely get their orders filled and "wait" until the next big fish comes in.
    """

    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        def next_state(st: OrderBook) -> NonTerminal[OrderBook]:
            if random.randint(0, 100) >= 99:
                if np.random.standard_normal() >= 0:
                    return NonTerminal(st.buy_market_order(random.randint(100, 150))[1])
                else:
                    return NonTerminal(st.sell_market_order(random.randint(100, 150))[1])
            else:
                try:
                    price_buy = (int) (np.random.normal(1.0 * st.bid_price(), 0.01 * st.bid_price()))
                except IndexError:
                    try:
                        price_buy = (int) (np.random.normal(0.98 * st.ask_price(), 0.01 * st.ask_price()))
                    except IndexError:
                        price_buy = np.random.normal(98, 10)
                try:
                    price_sell = int(np.random.normal(1.0 * st.ask_price(), 0.01 * st.ask_price()))
                except IndexError:
                    try:
                        price_sell = int (np.random.normal(1.02 * st.bid_price(), 0.01 * st.bid_price()))
                    except IndexError:
                        price_sell = np.random.normal(102, 1)
                m = random.randint(0, 2)
                if m == 0:
                    return NonTerminal(st.sell_limit_order(price_sell, random.randint(1, 10))[1])
                else:
                    return NonTerminal(st.buy_limit_order(price_buy, random.randint(1, 10))[1])
        fun = partial(next_state, state.state)
        return SampledDistribution(fun, expectation_samples=10)


if __name__ == '__main__':
    mp_base = OrderbookNormal()
    mp_fish = RareBigFish()
    # Generate a bunch of DollarAndShares objects calles asks and bids to generate a starting state
    bids: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (100 - x) * 10)
    ) for x in range(100, 90, -1)]
    asks: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (x - 105) * 10)
    ) for x in range(105, 115, 1)]

    ob0: OrderBook = OrderBook(descending_bids=bids, ascending_asks=asks)
    starting_state = NonTerminal(ob0)
    traces_base = mp_base.simulate(Constant(starting_state))
    traces_fish = mp_fish.simulate(Constant(starting_state))
    for i in range(1000):
        base_book = next(traces_fish)
        if i % 10 == 0:
            base_book.state.display_order_book()
