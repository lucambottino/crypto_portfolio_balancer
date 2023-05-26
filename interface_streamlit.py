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
import streamlit as st


#df_ticker = pd.DataFrame.from_dict(cg.get_coins_list())
df_ticker = pd.read_csv('tickers.csv')
#df_ticker.to_csv("tickers.csv")

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

coin_list = ["bitcoin", "ethereum", "cardano", "litecoin", "solana", "pancakeswap-token", "ripple", "polkadot", "binancecoin"]
symbol_list = ['ADA', 'ETH', 'AGI', 'LINK', 'SOL', 'LTC', 'COTI', 'ETC' '1inch']
print(get_coin_name_list(symbol_list))

df = get_coin_prices(coin_list).dropna()
print(df)
df_copy = df.copy()

#info = cg.get_coin_market_chart_by_id(id=coin_list[0], vs_currency='usd', days=364)
#print(info)
returns_df = df.to_returns()


returns = ffn.core.to_returns(df)
def calc_stats_from_df(df):
    perf = ffn.core.calc_stats(df)
    stats = perf.stats.T
    stats = stats.sort_values(by="monthly_sharpe", ascending=False)
    return stats


def calc_weights(returns):
    weights_df = pd.DataFrame()
    weights_df["risk_parity_weights"] = ffn.core.calc_erc_weights(returns, initial_weights=None, risk_weights=None, covar_method='standard', risk_parity_method='ccd', maximum_iterations=1000, tolerance=1e-03)
    weights_df["inv_vol_weights"] = ffn.core.calc_inv_vol_weights(returns)
    weights_df["mean_var_weights"] = ffn.calc_mean_var_weights(returns, rf=0.05, covar_method='standard')
    return weights_df
weights_df = calc_weights(returns)
print(weights_df)

def back_testing(df, strategy_name):
    df_rel = df.div(df.iloc[0])
    for col in df.columns:
        df_rel[col] = df_rel[col] * weights_df[strategy_name][col]
    overall = []
    for i in range(len(df_rel.index)):
        overall.append(np.sum(df_rel.iloc[i]))
    return overall

df_rel = df.div(df.iloc[0])
df_rel["risk_parity_weights"] = back_testing(df, "risk_parity_weights")
df_rel["inv_vol_weights"] = back_testing(df, "inv_vol_weights")
df_rel["mean_var_weights"] = back_testing(df, "mean_var_weights")
st.dataframe(df_rel)

df_stats = calc_stats_from_df(df_rel)
st.dataframe(df_stats)

import matplotlib.pyplot as plt
import plotly.graph_objects as go

fig = go.Figure()
for i in df_rel.columns:
    
    if i in ["risk_parity_weights", "inv_vol_weights"]:
        fig.add_trace(go.Scatter(x=df_rel.index, y=df_rel[i],
                    mode='lines',
                    name=i))
    else:
        fig.add_trace(go.Scatter(x=df_rel.index, y=df_rel[i],
                    mode='lines',
                    name=i, opacity=0.2))
        #plt.plot(df_rel.index, df_rel[i], alpha=0.2, linewidth=0.5)
#plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
#fig.update_layout("May 2020 - May 2021 Investiment strategies backtesting")
st.write(fig)

from finquant.portfolio import build_portfolio


pf = build_portfolio(data=df_copy)
# Monte Carlo optimisation
opt_w, opt_res = pf.mc_optimisation(num_trials=5000)
pf.mc_plot_results()
# minimisation to compute efficient frontier and optimal portfolios along it
pf.ef_plot_efrontier()
pf.ef.plot_optimal_portfolios()
# plotting individual stocks
pf.plot_stocks()


