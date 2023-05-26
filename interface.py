import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from datetime import datetime

import pandas as pd

from balance import Binance

from binance.client import Client


client = Client('UwQS6mxHvh88qVFSwPToQL4uPqlcZpv4zGnh5fYSbkW4FAwbvRpCZtHVWWUHpvBy', 'xW4EeUtlKQTP53ytxhAjlYJHQMtnmrDKBBVq0OtIWaPYo0KjCMNtndguHA4wfTD8', testnet=True)


options_df = pd.read_csv('app/usdt_coins.csv')
options_dict = [{'label':i, 'value':i} for i in options_df['baseAsset']]

app = dash.Dash(__name__)

app.layout = html.Div([
    #dcc.Graph(id='graph-with-slider'),
    dcc.Dropdown(
        id='token-selection',
        options=options_dict,
        value='BTC'
    ),
    dcc.DatePickerRange(
        id='my-date-picker-range',
        min_date_allowed=date(1995, 8, 5),
        max_date_allowed=date.today(),
        initial_visible_month=date(2017, 8, 5),
        start_date=date(2017, 8, 5),
        end_date=date.today()
    ),
    dcc.Graph(id='price'),
])


@app.callback(
    Output('price', 'figure'),
    Input('token-selection', 'value'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'))
def update_figure(token_selection, start_date, end_date):

    start_date = start_date.replace('-', '/') + " 12:00:00"
    end_date = end_date.replace('-', '/') + " 12:00:00"

    Binance_conn = Binance()
    df = Binance_conn.market_value(symbol=token_selection, kline_size=Client.KLINE_INTERVAL_1DAY, dateS=start_date, dateF=end_date)
    print(df)

    fig = px.line(df.iloc[0], df.iloc[0])

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
