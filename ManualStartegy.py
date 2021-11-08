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



import pandas as pd
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
from util import get_data, plot_data

def author():
    return 'jKennedy76'

class ManualStrategy(object):

    def __init__(self, verbose=False, impact=0.005):  # SET VERBOSE TO FALSE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.verbose = verbose
        self.impact = impact
        self.holdings = 0
        self.commision = 9.95


    def author(self):
        return 'jKennedy76'


    def testPolicy(self, symbol, sd, ed, sv):
        self.symbol = [symbol]
        self.start_date = sd
        self.end_date = ed
        self.start_value = sv
        self.cash = sv

        #calculate prices returns, adjusted close only
        prices = get_data(self.symbol, pd.date_range(self.start_date, self.end_date))
        prices = prices.drop('SPY', axis=1)  # get rid of SPY in dataframe
        daily_returns = pd.DataFrame(0., index=prices.index, columns=[self.symbol])
        self.portval = pd.DataFrame(0., index=prices.index, columns=[self.symbol])
        daily_returns[1:] = (prices[1:] / prices[:-1].values) - 1
        daily_returns.ix[0, :] = 0
        self.daily_returns = daily_returns


        # INDICATORS
        # norm_prices = prices / prices.values[0]   not nessecary, prices normalize themselves
        sma = prices.rolling(window=20).mean()
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
        d_ret = pd.DataFrame(0., index=prices.index, columns=[symbol])
        d_ret[1:] = (prices[1:] / prices[:-1].values) - 1
        d_ret.ix[0, :] = 0
        d_ret_pos_mask = d_ret > 0
        d_ret_neg_mask = d_ret < 0
        d_ret_pos_mask = d_ret_pos_mask.astype(np.int32)
        d_ret_neg_mask = d_ret_neg_mask.astype(np.int32)

        day_volume = volume * d_ret_pos_mask[symbol] - volume * d_ret_neg_mask[symbol]  # volume being traded for or against obv each day

        # calculate on balance volume from daily exchange / change in price
        for i in range(len(volume)):
            OBV[symbol][i] = day_volume[:i].sum()


        # create trades
        self.trades = pd.DataFrame(0., index=prices.index, columns=[self.symbol])

        price_per_sma = prices / sma

        self.buy_days = [] # initialize buy and sel days list for graphing purpose
        self.sell_days = []
        obv_count = 0
        for day in range(len(self.trades)-1):

            # MAIN STRATEGY

            # # BAD METRIC OBV positive or negetive
            # if self.holdings == 0:   #only act if you dont already have a position NEVERMIND
            #     # Buys
            #     if price_per_sma[symbol][day] < 0.95: # want price below moving average
            #         if prices[symbol][day] < bollinger_low[symbol][day]: # want price under low bolinger band
            #             if OBV[symbol][day] > 0:  # want a positive On balance Volume SWITCH TO BETTER METRIC
            #                 self.trades[symbol][day] = 1000
            #                 self.cash -= 1000.* prices[symbol][day]
            #                 self.buy_days.append(day)
            #
            #     #Sells
            #     elif price_per_sma[symbol][day] > 1.05:
            #         if prices[symbol][day] > bollinger_high[symbol][day]:
            #             if OBV[symbol][day] < 0: # SWITCH TO BETTER METRIC
            #                 self.trades[symbol][day] = -1000
            #                 self.cash += 1000.* prices[symbol][day]
            #                 self.sell_days.append(day)

            # BETTER METRIC compair obv to 5 days ago
            if day < 6:
                pass # dont trade till day 6 EXPAND METRICS OUT EXTRA DAYS

            elif self.holdings == 0:   #only act if you dont already have a position NEVERMIND
                # Buys
                if price_per_sma[symbol][day] < 0.95: # want price below moving average
                    if prices[symbol][day] < bollinger_low[symbol][day]: # want price under low bolinger band
                        if OBV[symbol][day] > OBV[symbol][day - 5]:  # want a positive On balance Volume
                            self.trades[symbol][day] = 1000
                            self.cash -= 1000.* prices[symbol][day] * (1 + self.impact) + self.commision
                            self.buy_days.append(day)

                #Sells
                elif price_per_sma[symbol][day] > 1.05:
                    if prices[symbol][day] > bollinger_high[symbol][day]:
                        if OBV[symbol][day] < OBV[symbol][day - 5]:
                            self.trades[symbol][day] = -1000
                            self.cash += 1000.* prices[symbol][day] * (1 - self.impact) - self.commision
                            self.sell_days.append(day)


            # # BETTER METRIC running count of OBVs
            # if day < 6:
            #     pass  # dont trade till day 6 EXPAND METRICS OUT EXTRA DAYS
            #
            # elif self.holdings == 0:  # only act if you dont already have a position NEVERMIND
            #     # Buys
            #     if price_per_sma[symbol][day] < 0.95:  # want price below moving average
            #         if prices[symbol][day] < bollinger_low[symbol][day]:  # want price under low bolinger band
            #             if OBV[symbol][day] > OBV[symbol][day -1]:  # want a positive On balance Volume
            #                 obv_count +=1
            #                 if obv_count > 2:
            #                     self.trades[symbol][day] = 1000
            #                     self.cash -= 1000. * prices[symbol][day]
            #                     self.buy_days.append(day)
            #
            #     # Sells
            #     elif price_per_sma[symbol][day] > 1.05:
            #         if prices[symbol][day] > bollinger_high[symbol][day]:
            #             if OBV[symbol][day] < OBV[symbol][day - 1]:
            #                 obv_count -=1
            #                 if obv_count < -2:
            #                     self.trades[symbol][day] = -1000
            #                     self.cash += 1000. * prices[symbol][day]
            #                     self.sell_days.append(day)

            # # EXIT positions in same way you would enter
            # elif self.holdings == 1000:
            #     #Sells
            #     if price_per_sma[symbol][day] > 1.05:
            #         if prices[symbol][day] > bollinger_high[symbol][day]:
            #             if OBV[symbol][day] < 0: # SWITCH TO BETTER METRIC
            #                 self.trades[symbol][day] = -2000
            #                 self.cash += 2000.* prices[symbol][day]
            #                 self.sell_days.append(day)
            #
            # elif self.holdings == -1000:
            #     # Buys
            #     if price_per_sma[symbol][day] < 0.95: # want price below moving average
            #         if prices[symbol][day] < bollinger_low[symbol][day]: # want price under low bolinger band
            #             if OBV[symbol][day] > 0:  # want a positive On balance Volume SWITCH TO BETTER METRIC
            #                 self.trades[symbol][day] = 1000
            #                 self.cash -= 1000.* prices[symbol][day]
            #                 self.buy_days.append(day)

            #exit position when crossing SMA
            elif self.holdings == 1000:
                if price_per_sma[symbol][day] > 1.00:
                    self.trades[symbol][day] = -1000
                    self.cash += 1000. * prices[symbol][day]

            elif self.holdings == -1000:
                if price_per_sma[symbol][day] < 1.00:
                    self.trades[symbol][day] = 1000
                    self.cash -= 1000. * prices[symbol][day]


            self.holdings += self.trades[symbol][day]
            self.portval[symbol][day] = self.holdings * prices[symbol][day] + self.cash


        # benchmark is buy and hold 1000 of symbol
        benchmark = ((prices[symbol] - prices[symbol][0]) *1000 + self.start_value)/ self.start_value

        self.portval[symbol][-1] = self.portval[symbol][-2]  # get rid of last day zero
        plt.plot(self.portval/self.start_value, 'r')
        plt.plot(benchmark, 'g')



        #PLOT vertical lines for long and short positions
        for buyday in self.buy_days:
            plt.axvline(x = self.trades.index[buyday], color = 'b') # draw vertical blue line on buy days

        for sellday in self.sell_days:
            plt.axvline( x = self.trades.index[sellday], color = 'k')
        plt.title('Manual Strategy ' + symbol)
        plt.ylabel('Normalized Portfolio Value')
        plt.xlabel('Date')
        plt.savefig('Experiment_1_MS.png')
        plt.legend(['MS Portfolio', 'Benchmark Portfolio'])
        if self.verbose:
            plt.show()
        plt.clf()

        # print stats
        if self.verbose:
            cumulative_return_MS = self.portval[symbol][-2]/self.start_value - 1.
            cumulative_return_benchmark = (self.start_value - 1000 * prices[symbol][0] + 1000 * prices[symbol][-1] ) / self.start_value - 1.
            daily_returns_MS = pd.DataFrame(0., index=prices.index, columns=[self.symbol])
            daily_returns_MS[1:] = (self.portval[1:] / self.portval[:-1].values) - 1
            daily_returns_MS.ix[0, :] = 0
            mean_daily_returns_MS = daily_returns_MS.mean()
            mean_daily_returns_benchmark = self.daily_returns.mean()
            std_daily_returns_MS = daily_returns_MS.std()
            std_daily_returns_benchmark = self.daily_returns.std()

            print " "
            print "Manual Strategy"
            print cumulative_return_MS , 'cumulative_return_MS'
            print cumulative_return_benchmark , 'cumulative_return_benchmark'
            print mean_daily_returns_MS.values , 'mean_daily_returns_MS.values'
            print mean_daily_returns_benchmark.values, 'mean_daily_returns_benchmark.values'
            print std_daily_returns_MS.values, 'std_daily_returns_MS.values'
            print std_daily_returns_benchmark.values , 'std_daily_returns_benchmark.values'

        return self.trades


if __name__ == '__main__':
    ms = ManualStrategy(verbose=False, impact=0.005)
    # df_trades = ms.testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000) # JPM out of sample as in report part 4
    df_trades = ms.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000) # JPM in sample, as in report part 3

    # df_trades = ms.testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)  # AAPL out of sample test per writeup



