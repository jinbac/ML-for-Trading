"""MC2-P1: Market simulator.

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


def compute_portvals( verbose = False, start_val=100000, commission=9.95, impact=0.005):
    # creating prices dataframe
    symbols = ['JPM']
    # symbols = np.unique(symbols.values).tolist()  # want to call each companies prices only once
    sd = dt.datetime(2008,1,2)
    ed = dt.datetime(2009,12,30)
    dates = pd.date_range(sd , ed )
    start_date = dates[0]
    end_date = dates[-1]
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices = prices.drop('SPY', axis=1)  # get rid of SPY in dataframe



# PLOTTTING FOR PROJECT

    #PRICE per SMA

    prices = prices / prices.values[0]
    sma = prices.rolling(window=20).mean()
    plt.plot (prices)
    plt.plot (sma)
    plt.plot (prices/sma)
    plt.title('Price / SMA Indicator, JPM')
    plt.ylabel('Normalized Price')
    plt.xlabel('Date')
    plt.savefig('Price_SMA_Indicator.png')
    plt.legend(['Price', 'SMA', 'Price / SMA'])
    if verbose:
        plt.show()
    plt.clf()

    # Bolinger Bands
    bollinger_low =  sma - prices.rolling(window=20, center=False).std() *2.
    bollinger_high = sma + prices.rolling(window=20, center=False).std() *2.
    plt.plot (prices)
    plt.plot (bollinger_high)
    plt.plot (bollinger_low)
    plt.title('Bollinger Bands Indicator, JPM')
    plt.ylabel('Normalized Price')
    plt.xlabel('Date')
    plt.savefig('Bollinger_Indicator.png')
    plt.legend(['Price', 'Upper Bollinger Band', 'Lower Bollinger Band'])
    if verbose:
        plt.show()
    plt.clf()




    #OBV
    def symbol_to_path(symbol, base_dir=None):
        """Return CSV file path given ticker symbol."""
        if base_dir is None:
            base_dir = os.environ.get("MARKET_DATA_DIR", '../data/')
        return os.path.join(base_dir, "{}.csv".format(str(symbol)))

    volume = pd.read_csv(symbol_to_path('JPM'), index_col='Date',parse_dates=True, usecols=['Date', 'Volume'], na_values=['nan'])

    volume = volume['Volume'][end_date: start_date]
    volume = volume.reindex(index=volume.index[::-1])


    OBV = pd.DataFrame(0., index=prices.index, columns=['JPM'])
    daily_returns = pd.DataFrame(0., index=prices.index, columns=['JPM'])
    daily_returns[1:] = (prices[1:] / prices[:-1].values) - 1
    daily_returns.ix[0, :] = 0
    daily_returns_pos_mask = daily_returns > 0
    daily_returns_neg_mask = daily_returns < 0
    daily_returns_pos_mask = daily_returns_pos_mask.astype(np.int32)
    daily_returns_neg_mask = daily_returns_neg_mask.astype(np.int32)

    day_volume = volume * daily_returns_pos_mask['JPM'] - volume * daily_returns_neg_mask['JPM'] # volume being traded for or against obv each day


    # calculate on balance volume from daily exchange / change in price
    for i in range(len(volume)):
        OBV['JPM'][i] = day_volume[:i].sum()

    plt.plot (OBV)
    plt.plot (prices * OBV)
    plt.title('On Balance Volume Indicator, JPM')
    plt.ylabel('Normalized Price / Volume')
    plt.xlabel('Date')
    plt.savefig('OBV_Indicator.png')
    plt.legend(['OBV', 'Price'])
    if verbose:
        plt.show()
    plt.clf()



def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    sv = 100000

    # Process orders
    portvals = compute_portvals(verbose=False , start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"


if __name__ == "__main__":
    test_code()
