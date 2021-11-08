"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Jarod Kennedy (replace with your name)
GT User ID: jKennedy76 (replace with your User ID)
GT ID: 903369277 (replace with your GT ID)
"""

import datetime as dt
import pandas as pd
import util as ut
import random
import QLearner as ql
import numpy as np
import os
import matplotlib.pyplot as plt
from util import get_data, plot_data


def author():
    return 'jKennedy76'


class StrategyLearner(object):
    # constructor
    def __init__(self, verbose=False, impact=0.005):  # SET VERBOSE TO FALSE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.verbose = verbose
        self.impact = impact
        self.holdings = 0
        self.commision = 9.95

    def author(self):
        return 'jKennedy76'

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol="JPM", \
                    sd=dt.datetime(2008, 1, 1), \
                    ed=dt.datetime(2009, 12, 31), \
                    sv=100000):

        # add your code to do learning here
        self.learner = ql.QLearner(num_states=300, \
                                   num_actions=3, \
                                   alpha=0.2, \
                                   gamma=0.9, \
                                   rar=0.98, \
                                   radr=0.999, \
                                   dyna=0, \
                                   verbose=False)  # initialize the learner

        self.symbol = [symbol]
        self.start_date = sd
        self.end_date = ed
        self.start_value = sv
        self.cash = sv

        # calculate prices returns, adjusted close only
        prices = get_data(self.symbol, pd.date_range(self.start_date, self.end_date))
        prices = prices.drop('SPY', axis=1)  # get rid of SPY in dataframe
        daily_returns = pd.DataFrame(0., index=prices.index, columns=[self.symbol])
        self.portval = pd.DataFrame(0., index=prices.index, columns=[self.symbol])
        daily_returns[1:] = (prices[1:] / prices[:-1].values) - 1
        daily_returns.ix[0, :] = 0
        self.daily_returns = daily_returns

        # INDICATORS
        sma = prices.rolling(window=20).mean()
        price_per_sma = prices / sma
        bollinger_low = sma - prices.rolling(window=20, center=False).std() * 2.
        bollinger_high = sma + prices.rolling(window=20, center=False).std() * 2.

        # OBV
        def symbol_to_path(symbol, base_dir=None):
            """Return CSV file path given ticker symbol."""
            if base_dir is None:
                base_dir = os.environ.get("MARKET_DATA_DIR", '../data/')
            return os.path.join(base_dir, "{}.csv".format(str(symbol)))

        volume = pd.read_csv(symbol_to_path(symbol), index_col='Date', parse_dates=True, usecols=['Date', 'Volume'], na_values=['nan'])

        volume = volume['Volume'][self.end_date: self.start_date]
        volume = volume.reindex(index=volume.index[::-1])

        OBV = pd.DataFrame(0., index=prices.index, columns=[symbol])
        dailyret = pd.DataFrame(0., index=prices.index, columns=[symbol])
        dailyret[1:] = (prices[1:] / prices[:-1].values) - 1
        dailyret.ix[0, :] = 0
        dailyret_pos_mask = dailyret > 0
        dailyret_neg_mask = dailyret < 0
        dailyret_pos_mask = dailyret_pos_mask.astype(np.int32)
        dailyret_neg_mask = dailyret_neg_mask.astype(np.int32)

        day_volume = volume * dailyret_pos_mask[symbol] - volume * dailyret_neg_mask[symbol]  # volume being traded for or against obv each day

        # calculate on balance volume from daily exchange / change in price
        for i in range(len(volume)):
            OBV[symbol][i] = day_volume[:i].sum()

        self.trades = pd.DataFrame(0., index=prices.index, columns=[self.symbol])  # initialize trades DF this initialization is used for INDEXING ONLY

        # Descreteize states  LECTURE SAID TO BALANCE STATES BY SAMPLES!!!!!!!!!!!!!!!! REACH BACK IN TIME FOR STARTING DATA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        # price per sma CONSIDER CONDENSING RANGE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for day in range(len(self.trades)):
            if price_per_sma[symbol][day] < 0.5:
                price_per_sma[symbol][day] = 0
            elif price_per_sma[symbol][day] < .8:
                price_per_sma[symbol][day] = 1
            elif price_per_sma[symbol][day] < .9:
                price_per_sma[symbol][day] = 2
            elif price_per_sma[symbol][day] < .95:
                price_per_sma[symbol][day] = 3
            elif price_per_sma[symbol][day] < 1.0:
                price_per_sma[symbol][day] = 4
            elif price_per_sma[symbol][day] < 1.05:
                price_per_sma[symbol][day] = 5
            elif price_per_sma[symbol][day] < 1.1:
                price_per_sma[symbol][day] = 6
            elif price_per_sma[symbol][day] < 1.3:
                price_per_sma[symbol][day] = 7
            elif price_per_sma[symbol][day] < 1.5:
                price_per_sma[symbol][day] = 8
            elif price_per_sma[symbol][day] > 1.5:
                price_per_sma[symbol][day] = 9
            else:
                price_per_sma[symbol][day] = 4  # fillout nan data NOT NEEDED IF PAST DATA IMPLEMENTED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # bolinger
        bollinger_state = pd.DataFrame(0., index=prices.index, columns=[self.symbol])  # initialize bollinger state

        for day in range(len(self.trades)):
            if prices[symbol][day] < bollinger_low[symbol][day]:
                bollinger_state[symbol][day] = 0
            elif prices[symbol][day] > bollinger_high[symbol][day]:
                bollinger_state[symbol][day] = 2
            else:
                bollinger_state[symbol][day] = 1

        # OBV
        OBV_standard = abs(day_volume[:4]).sum()  # todays on balance volume will be judged against first five days CHANGE TO STATISTIC BASED ON OLDER DATA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for day in range(len(self.trades)):
            if OBV[symbol][day] < OBV_standard * -1.:
                OBV[symbol][day] = 0
            elif OBV[symbol][day] < OBV_standard * -0.7:
                OBV[symbol][day] = 1
            elif OBV[symbol][day] < OBV_standard * -0.4:
                OBV[symbol][day] = 2
            elif OBV[symbol][day] < OBV_standard * -0.2:
                OBV[symbol][day] = 3
            elif OBV[symbol][day] < OBV_standard * 0.:
                OBV[symbol][day] = 4
            elif OBV[symbol][day] < OBV_standard * .2:
                OBV[symbol][day] = 5
            elif OBV[symbol][day] < OBV_standard * .4:
                OBV[symbol][day] = 6
            elif OBV[symbol][day] < OBV_standard * .7:
                OBV[symbol][day] = 7
            elif OBV[symbol][day] < OBV_standard * 1.:
                OBV[symbol][day] = 8
            elif OBV[symbol][day] > OBV_standard * 1.:
                OBV[symbol][day] = 9
            else:
                OBV[symbol][day] = 4  # fillout nan data NOT NEEDED IF PAST DATA IMPLEMENTED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # initialize Q learner
        states = bollinger_state * 100 + OBV * 10 + price_per_sma
        states = states.astype(np.int32)

        # run Q learner to learn
        epoch = 0
        last_portval = 0
        while epoch < 10 or last_portval > 1.001 * self.portval[symbol][-2] or last_portval < 0.999 * self.portval[symbol][-2]:  # end learning stage after a minimum of 10 epochs if qlearner has converged based on similarity
            # print last_portval,'last portval'
            # print self.portval[symbol][-2], 'portval'
            epoch += 1
            initial_state = 144  # initial state integer
            # initial_state = states[symbol][self.start_date]
            action = self.learner.querysetstate(initial_state)  # get first action from initial state
            self.buy_days = []  # initialize buy and sell days list for graphing purpose
            self.sell_days = []
            self.trades = pd.DataFrame(0, index=prices.index, columns=[self.symbol])  # initialize trades DF  every epoch
            self.holdings = 0  # initialize holdings every epoch
            last_portval = self.portval[symbol][-2]  # save last epoch's portfolio value for similarity convergence
            self.portval = pd.DataFrame(0., index=prices.index, columns=[self.symbol])
            self.cash = self.start_value

            for day in range(len(self.trades) - 1):  # -1?????????????????????????????????????????????????????????????????????????????????
                # change state based on action
                no_trade = True
                if action == 0:  # go short
                    if self.holdings == 1000:  # holding long sell 2000
                        self.trades[symbol][day] = -2000
                        self.holdings = -1000
                        self.cash += 2000. * prices[symbol][day] * (1 - self.impact) - self.commision
                        self.sell_days.append(day)
                        no_trade = False
                    elif self.holdings == 0:  # not holding sell 1000
                        self.trades[symbol][day] = -1000
                        self.holdings = -1000
                        self.cash += 1000. * prices[symbol][day] * (1 - self.impact) - self.commision
                        self.sell_days.append(day)
                        no_trade = False
                        # holding short dont act

                elif action == 1:  # exit position, hold nothing
                    if self.holdings == 1000:  # holding long sell 1000
                        self.trades[symbol][day] = -1000
                        self.holdings = 0
                        self.cash += 1000. * prices[symbol][day] * (1 - self.impact) - self.commision
                        self.sell_days.append(day)
                        no_trade = False
                    elif self.holdings == -1000:  # holding short buy 1000
                        self.trades[symbol][day] = 1000
                        self.holdings = 0
                        self.cash -= 1000. * prices[symbol][day] * (1 - self.impact) - self.commision
                        self.buy_days.append(day)
                        no_trade = False
                        # holding nothing dont act

                elif action == 2:  # go long
                    if self.holdings == 0:  # holding nothing buy 1000
                        self.trades[symbol][day] = 1000
                        self.holdings = 1000
                        self.cash -= 1000. * prices[symbol][day] * (1 - self.impact) - self.commision
                        self.buy_days.append(day)
                        no_trade = False
                    elif self.holdings == -1000:  # holding short buy 2000
                        self.trades[symbol][day] = 2000
                        self.holdings = 1000
                        self.cash -= 2000. * prices[symbol][day] * (1 - self.impact) - self.commision
                        self.buy_days.append(day)
                        no_trade = False
                        # holding long dont act

                # update state, reward based on daily return day + 1
                s_prime = states[symbol][day + 1]
                r = self.daily_returns[symbol][day + 1] * self.holdings * (1 - self.impact) - self.commision
                if no_trade:
                    r = self.impact * 1000.
                action = self.learner.query(s_prime, r)

                # calculate portval
                self.portval[symbol][day] = self.holdings * prices[symbol][day] + self.cash  # LAST DAY OF PORTVAL BLANK

                # print self.portval[symbol][-2], epoch

        # Print Stats and Build Plots
        # benchmark is buy and hold 1000 of symbol
        benchmark = ((prices[symbol] - prices[symbol][0]) * 1000 + self.start_value)

        self.portval[symbol][-1] = self.portval[symbol][-2]  # get rid of last day zero
        plt.plot((self.portval - self.start_value) / self.start_value + 1., 'r')
        plt.plot((benchmark - self.start_value) / self.start_value + 1., 'g')

        # PLOT vertical lines for long and short positions
        for buyday in self.buy_days:
            plt.axvline(x=self.trades.index[buyday], color='b')  # draw vertical blue line on buy days

        for sellday in self.sell_days:
            plt.axvline(x=self.trades.index[sellday], color='k')
        plt.title('Strategy Learner ' + symbol)
        plt.ylabel('Normalized Portfolio Value')
        plt.xlabel('Date')
        plt.savefig('Experiment_2_in_Sample.png')
        plt.legend(['SL Portfolio', 'Benchmark Portfolio'])
        if self.verbose:
            plt.show()
        plt.clf()

        if self.verbose:
            # print stats NOTHING wrong with cumulative returns, stock being bought on MARGIN
            cumulative_return_MS = (self.portval[symbol][-1] - self.start_value) / self.start_value
            cumulative_return_benchmark = (benchmark[-1] - self.start_value) / self.start_value
            daily_returns_MS = pd.DataFrame(0., index=prices.index, columns=[self.symbol])
            daily_returns_MS[1:] = (self.portval[1:] / self.portval[:-1].values) - 1
            daily_returns_MS.ix[0, :] = 0
            mean_daily_returns_MS = daily_returns_MS.mean()
            mean_daily_returns_benchmark = self.daily_returns.mean()
            std_daily_returns_MS = daily_returns_MS.std()
            std_daily_returns_benchmark = self.daily_returns.std()

            print" "
            print "Impact = " , self.impact
            print np.count_nonzero(self.buy_days), "buy days"
            print np.count_nonzero(self.sell_days), "sell days"
            print cumulative_return_MS, 'cumulative_return_SL'
            print cumulative_return_benchmark, 'cumulative_return_benchmark'
            print mean_daily_returns_MS.values, 'mean_daily_returns_SL.values'
            print mean_daily_returns_benchmark.values, 'mean_daily_returns_benchmark.values'
            print std_daily_returns_MS.values, 'std_daily_returns_SL.values'
            print std_daily_returns_benchmark.values, 'std_daily_returns_benchmark.values'














    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol="JPM", \
                   sd=dt.datetime(2010, 1, 1), \
                   ed=dt.datetime(2011, 12, 31), \
                   sv=100000):

        self.symbol = [symbol]
        self.start_date = sd
        self.end_date = ed
        self.start_value = sv
        self.cash = sv

        # calculate prices returns, adjusted close only
        prices = get_data(self.symbol, pd.date_range(self.start_date, self.end_date))
        prices = prices.drop('SPY', axis=1)  # get rid of SPY in dataframe
        daily_returns = pd.DataFrame(0., index=prices.index, columns=[self.symbol])
        self.portval = pd.DataFrame(0., index=prices.index, columns=[self.symbol])
        daily_returns[1:] = (prices[1:] / prices[:-1].values) - 1
        daily_returns.ix[0, :] = 0
        self.daily_returns = daily_returns

        # INDICATORS
        sma = prices.rolling(window=20).mean()
        price_per_sma = prices / sma
        bollinger_low = sma - prices.rolling(window=20, center=False).std() * 2.
        bollinger_high = sma + prices.rolling(window=20, center=False).std() * 2.

        # OBV
        def symbol_to_path(symbol, base_dir=None):
            """Return CSV file path given ticker symbol."""
            if base_dir is None:
                base_dir = os.environ.get("MARKET_DATA_DIR", '../data/')
            return os.path.join(base_dir, "{}.csv".format(str(symbol)))

        volume = pd.read_csv(symbol_to_path(symbol), index_col='Date', parse_dates=True, usecols=['Date', 'Volume'], na_values=['nan'])

        volume = volume['Volume'][self.end_date: self.start_date]
        volume = volume.reindex(index=volume.index[::-1])

        OBV = pd.DataFrame(0., index=prices.index, columns=[symbol])
        dailyret = pd.DataFrame(0., index=prices.index, columns=[symbol])
        dailyret[1:] = (prices[1:] / prices[:-1].values) - 1
        dailyret.ix[0, :] = 0
        dailyret_pos_mask = dailyret > 0
        dailyret_neg_mask = dailyret < 0
        dailyret_pos_mask = dailyret_pos_mask.astype(np.int32)
        dailyret_neg_mask = dailyret_neg_mask.astype(np.int32)

        day_volume = volume * dailyret_pos_mask[symbol] - volume * dailyret_neg_mask[symbol]  # volume being traded for or against obv each day

        # calculate on balance volume from daily exchange / change in price
        for i in range(len(volume)):
            OBV[symbol][i] = day_volume[:i].sum()

        self.trades = pd.DataFrame(0., index=prices.index, columns=[self.symbol])  # initialize trades DF this initialization is used for INDEXING ONLY

        # Descreteize states  LECTURE SAID TO BALANCE STATES BY SAMPLES!!!!!!!!!!!!!!!! REACH BACK IN TIME FOR STARTING DATA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # price per sma CONSIDER CONDENSING RANGE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for day in range(len(self.trades)):
            if price_per_sma[symbol][day] < 0.5:
                price_per_sma[symbol][day] = 0
            elif price_per_sma[symbol][day] < .8:
                price_per_sma[symbol][day] = 1
            elif price_per_sma[symbol][day] < .9:
                price_per_sma[symbol][day] = 2
            elif price_per_sma[symbol][day] < .95:
                price_per_sma[symbol][day] = 3
            elif price_per_sma[symbol][day] < 1.0:
                price_per_sma[symbol][day] = 4
            elif price_per_sma[symbol][day] < 1.05:
                price_per_sma[symbol][day] = 5
            elif price_per_sma[symbol][day] < 1.1:
                price_per_sma[symbol][day] = 6
            elif price_per_sma[symbol][day] < 1.3:
                price_per_sma[symbol][day] = 7
            elif price_per_sma[symbol][day] < 1.5:
                price_per_sma[symbol][day] = 8
            elif price_per_sma[symbol][day] > 1.5:
                price_per_sma[symbol][day] = 9
            else:
                price_per_sma[symbol][day] = 4  # fillout nan data NOT NEEDED IF PAST DATA IMPLEMENTED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # bolinger
        bollinger_state = pd.DataFrame(0., index=prices.index, columns=[self.symbol])  # initialize bollinger state

        for day in range(len(self.trades)):
            if prices[symbol][day] < bollinger_low[symbol][day]:
                bollinger_state[symbol][day] = 0
            elif prices[symbol][day] > bollinger_high[symbol][day]:
                bollinger_state[symbol][day] = 2
            else:
                bollinger_state[symbol][day] = 1

        # OBV
        OBV_standard = abs(day_volume[:4]).sum() # todays on balance volume will be judged against first five days CHANGE TO STATISTIC BASED ON OLDER DATA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for day in range(len(self.trades)):
            if OBV[symbol][day] < OBV_standard * -1.:
                OBV[symbol][day] = 0
            elif OBV[symbol][day] < OBV_standard * -0.7:
                OBV[symbol][day] = 1
            elif OBV[symbol][day] < OBV_standard * -0.4:
                OBV[symbol][day] = 2
            elif OBV[symbol][day] < OBV_standard * -0.2:
                OBV[symbol][day] = 3
            elif OBV[symbol][day] < OBV_standard * 0.:
                OBV[symbol][day] = 4
            elif OBV[symbol][day] < OBV_standard * .2:
                OBV[symbol][day] = 5
            elif OBV[symbol][day] < OBV_standard * .4:
                OBV[symbol][day] = 6
            elif OBV[symbol][day] < OBV_standard * .7:
                OBV[symbol][day] = 7
            elif OBV[symbol][day] < OBV_standard * 1.:
                OBV[symbol][day] = 8
            elif OBV[symbol][day] > OBV_standard * 1.:
                OBV[symbol][day] = 9
            else:
                OBV[symbol][day] = 4  # fillout nan data NOT NEEDED IF PAST DATA IMPLEMENTED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # initialize Q learner
        states = bollinger_state * 100 + OBV * 10 + price_per_sma
        states = states.astype(np.int32)

        # run Q learner to TEST NO LEARNing

        initial_state = 144  # initial state integer
        # initial_state = states[symbol][self.start_date]
        action = self.learner.querysetstate(initial_state)  # get first action from initial state
        self.buy_days = []  # initialize buy and sell days list for graphing purpose
        self.sell_days = []
        self.trades = pd.DataFrame(0, index=prices.index, columns=[self.symbol])  # initialize trades DF  every epoch
        self.holdings = 0  # initialize holdings every epoch
        last_portval = self.portval[symbol][-2]  # save last epoch's portfolio value for similarity convergence
        self.portval = pd.DataFrame(0., index=prices.index, columns=[self.symbol])
        self.cash = self.start_value

        for day in range(len(self.trades) - 1):  # -1?????????????????????????????????????????????????????????????????????????????????

            # change state based on action
            if action == 0:  # go short
                if self.holdings == 1000:  # holding long sell 2000
                    self.trades[symbol][day] = -2000
                    self.holdings = -1000
                    self.cash += 2000. * prices[symbol][day] * (1 - self.impact) - self.commision
                    self.sell_days.append(day)
                elif self.holdings == 0:  # not holding sell 1000
                    self.trades[symbol][day] = -1000
                    self.holdings = -1000
                    self.cash += 1000. * prices[symbol][day] * (1 - self.impact) - self.commision
                    self.sell_days.append(day)
                    # holding short dont act

            elif action == 1:  # exit position, hold nothing
                if self.holdings == 1000:  # holding long sell 1000
                    self.trades[symbol][day] = -1000
                    self.holdings = 0
                    self.cash += 1000. * prices[symbol][day] * (1 - self.impact) - self.commision
                    self.sell_days.append(day)
                elif self.holdings == -1000:  # holding short buy 1000
                    self.trades[symbol][day] = 1000
                    self.holdings = 0
                    self.cash -= 1000. * prices[symbol][day] * (1 - self.impact) - self.commision
                    self.buy_days.append(day)
                    # holding nothing dont act

            elif action == 2:  # go long
                if self.holdings == 0:  # holding nothing buy 1000
                    self.trades[symbol][day] = 1000
                    self.holdings = 1000
                    self.cash -= 1000. * prices[symbol][day] * (1 - self.impact) - self.commision
                    self.buy_days.append(day)
                elif self.holdings == -1000:  # holding short buy 2000
                    self.trades[symbol][day] = 2000
                    self.holdings = 1000
                    self.cash -= 2000. * prices[symbol][day] * (1 - self.impact) - self.commision
                    self.buy_days.append(day)
                    # holding long dont act

            # update state, reward based on daily return day + 1
            s_prime = states[symbol][day + 1]
            action = self.learner.querysetstate(s_prime)

            # calculate portval
            self.portval[symbol][day] = self.holdings * prices[symbol][day] + self.cash  # LAST DAY OF PORTVAL BLANK

        # Print Stats and Build Plots
        # benchmark is buy and hold 1000 of symbol
        benchmark = ((prices[symbol] - prices[symbol][0]) * 1000 + self.start_value)

        self.portval[symbol][-1] = self.portval[symbol][-2]  # get rid of last day zero
        plt.plot((self.portval - self.start_value) / self.start_value + 1., 'r')
        plt.plot((benchmark - self.start_value) / self.start_value + 1., 'g')

        # PLOT vertical lines for long and short positions
        for buyday in self.buy_days:
            plt.axvline(x=self.trades.index[buyday], color='b')  # draw vertical blue line on buy days

        for sellday in self.sell_days:
            plt.axvline(x=self.trades.index[sellday], color='k')
        plt.title('Strategy Learner ' + symbol)
        plt.ylabel('Normalized Portfolio Value')
        plt.xlabel('Date')
        plt.savefig('Experiment_2_Out_of_Sample.png')
        plt.legend(['SL Portfolio', 'Benchmark Portfolio'])
        if self.verbose:
            plt.show()
        plt.clf()

        if self.verbose:
            # print stats NOTHING wrong with cumulative returns, stock being bought on MARGIN
            cumulative_return_MS = (self.portval[symbol][-1] - self.start_value) / self.start_value
            cumulative_return_benchmark = (benchmark[-1] - self.start_value) / self.start_value
            daily_returns_MS = pd.DataFrame(0., index=prices.index, columns=[self.symbol])
            daily_returns_MS[1:] = (self.portval[1:] / self.portval[:-1].values) - 1
            daily_returns_MS.ix[0, :] = 0
            mean_daily_returns_MS = daily_returns_MS.mean()
            mean_daily_returns_benchmark = self.daily_returns.mean()
            std_daily_returns_MS = daily_returns_MS.std()
            std_daily_returns_benchmark = self.daily_returns.std()

            print" "
            print "Impact = " , self.impact
            print cumulative_return_MS, 'cumulative_return_SL'
            print cumulative_return_benchmark, 'cumulative_return_benchmark'
            print mean_daily_returns_MS.values, 'mean_daily_returns_SL.values'
            print mean_daily_returns_benchmark.values, 'mean_daily_returns_benchmark.values'
            print std_daily_returns_MS.values, 'std_daily_returns_SL.values'
            print std_daily_returns_benchmark.values, 'std_daily_returns_benchmark.values'

        trades = self.trades
        return trades


if __name__ == "__main__":

    verb = True # set to true to see effects of impact

    imp = 0.
    total_trades = []
    cumulative_returns = []

    for k in range(100):
        # print k, imp
        learner = StrategyLearner(verbose=False, impact=imp)  # runs __init___ in qlearner
        learner.addEvidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)  # training phase
        buys = np.count_nonzero(learner.buy_days)
        sells = np.count_nonzero(learner.sell_days)
        total_trades.append(buys+sells)
        cumulative = (learner.portval["JPM"][-1] - learner.start_value) / learner.start_value
        cumulative_returns.append(cumulative)
        imp += 0.001
        # print total_trades

    plt.title('Effects of Impact on # of Trades ')
    plt.ylabel('# of Trades')
    plt.xlabel('Impact')
    plt.savefig('Effects_of_Impact.png')
    plt.plot (total_trades)
    if verb:
        plt.show()
    plt.clf()

    plt.title('Effects of Impact on Cumulative Returns ')
    plt.ylabel('Cumulative Returns')
    plt.xlabel('Impact')
    plt.savefig('Cumulative_Returns.png')
    plt.plot (cumulative_returns)
    if verb:
        plt.show()
    plt.clf()





