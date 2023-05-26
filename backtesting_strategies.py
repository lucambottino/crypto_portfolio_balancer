# medias moveis 5, 8, 13, 50, 56, 100, 200, 800

import backtesting
import streamlit as st

from backtesting import Backtest, Strategy
from backtesting.lib import crossover


import urllib.request
import ssl
import json
import time
import pandas as pd
from datetime import datetime
import numpy as np
import plotly
import ffn
import finquant
from pycoingecko import CoinGeckoAPI
cg = CoinGeckoAPI()
from datetime import date

df_ticker = pd.read_csv('tickers.csv')


def get_coin_prices(coin_list):
    frames = []
    for coin_name in coin_list:
        df_price = pd.DataFrame.from_dict(cg.get_coin_market_chart_by_id(id=coin_name, vs_currency='usd', days=364))
        df = pd.DataFrame()
        df[coin_name] = [i[1] for i in df_price["prices"]]
        df.index = [datetime.fromtimestamp(int(i[0])/1000.0) for i in df_price["prices"]]
        frames.append(df)
    df = pd.concat(frames, axis=1)
    df.index.name = "Date"
    return df


def get_coin_name_list(symbol_list):
    l = []
    for c in symbol_list:
        c = c.lower()
        name = df_ticker[df_ticker["symbol"] == c]
    l.append(name)
    return l

coin_list = ["bitcoin"]
df = get_coin_prices(coin_list).dropna()
print(df)

from backtesting.test import SMA, GOOG, EMA

print(GOOG)

media1 = st.number_input("Media movel entrada", value=8)
media2 = st.number_input("Media movel saida", value=15)

class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(EMA, price, media1)
        self.ma2 = self.I(EMA, price, media2)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


bt = Backtest(GOOG, SmaCross, commission=.002,
              exclusive_orders=True)
stats = bt.run()
bt.plot()