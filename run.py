#!/usr/bin/env python
import argparse

from finrl import config
from finrl import config_tickers
from finrl.main import check_and_make_directories
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split, DataReader
from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.finrl_meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint
from stable_baselines3 import SAC

import sys
sys.path.append("../FinRL-Library")

import itertools

from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
)


def prepare_data(ticks, start_date, end_date):

    df = DataReader.get_ohlcv(ticks, start_date, end_date, version=1).sort_values(['date','tic'],ignore_index=True)
    print(f"df.shape: {df.shape}")

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list = config.INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature = False)

    processed = fe.preprocess_data(df)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])

    processed_full = processed_full.fillna(0)

    processed_full.sort_values(['date','tic'],ignore_index=True).head(10)

    train = data_split(processed_full, '2009-01-01','2020-07-01')
    trade = data_split(processed_full, '2020-07-01','2021-10-31')
    print(f"len(train): {len(train)}")
    print(f"len(trade): {len(trade)}")
    print(f"train.tail(): {train.tail()}")
    print(f"trade.head(): {trade.head()}")
    
    return processed_full, train, trade




'''
# Part 4: Preprocess Data
Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.
* Add technical indicators. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc. In this article, we demonstrate two trend-following technical indicators: MACD and RSI.
* Add turbulence index. Risk-aversion reflects whether an investor will choose to preserve the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the financial turbulence index that measures extreme asset price fluctuation.
'''


'''
# Part 5. Design Environment
Considering the stochastic and interactive nature of the automated stock trading tasks, a financial task is modeled as a **Markov Decision Process (MDP)** problem. The training process involves observing stock price change, taking an action and reward's calculation to have the agent adjusting its strategy accordingly. By interacting with the environment, the trading agent will derive a trading strategy with the maximized rewards as time proceeds.

Our trading environments, based on OpenAI Gym framework, simulate live stock markets with real market data according to the principle of time-driven simulation.

The action space describes the allowed actions that the agent interacts with the environment. Normally, action a includes three actions: {-1, 0, 1}, where -1, 0, 1 represent selling, holding, and buying one share. Also, an action can be carried upon multiple shares. We use an action space {-k,…,-1, 0, 1, …, k}, where k denotes the number of shares to buy and -k denotes the number of shares to sell. For example, "Buy 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or -10, respectively. The continuous action space needs to be normalized to [-1, 1], since the policy is defined on a Gaussian distribution, which needs to be normalized and symmetric.

# Training data split: 2009-01-01 to 2020-07-01
# Trade data split: 2020-07-01 to 2021-10-31
'''


def train_model(model_name, train):
    print(f"config.INDICATORS: {config.INDICATORS}")

    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(config.INDICATORS)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "make_plots": True
    }

    e_train_gym = StockTradingEnv(df = train, **env_kwargs)

    env_train, _ = e_train_gym.get_sb_env()
    print(f"type(env_train): {type(env_train)}")
    
    PARAMS = {
        'ppo': {
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 128},

        'td3': {
            "batch_size": 100,
            "buffer_size": 1000000,
            "learning_rate": 0.001},

        'sac': {
            "batch_size": 128,
            "buffer_size": 1000000,
            "learning_rate": 0.0001,
            "learning_starts": 100,
            "ent_coef": "auto_0.1"}
    }
    model_args = PARAMS[model_name] if model_name in PARAMS else None
    agent = DRLAgent(env = env_train)
    model = agent.get_model(model_name, model_kwargs=model_args)
    """     trained_model = agent.train_model(
        model=model,
        tb_log_name=model_name,
        total_timesteps=60000) """
    trained_model = SAC.load(os.path.join(config.TRAINED_MODEL_DIR, "sac_20220617-1037.model"))

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    #trained_model.save(os.path.join(config.TRAINED_MODEL_DIR, F"{model_name}_{ts}.model"))

    return trained_model, env_kwargs

'''
## Trading
Assume that we have $1,000,000 initial capital at 2020-07-01. We use the DDPG model to trade Dow jones 30 stocks.
### Set turbulence threshold
Set the turbulence threshold to be greater than the maximum of insample turbulence data, if current turbulence index is greater than the threshold, then we assume that the current market is volatile
'''

def back_test(model_name, ticks, start_date, end_date):
    processed_full, train, trade = prepare_data(ticks, start_date, end_date)

    trained_model, env_kwargs = train_model(model_name, train)

    data_risk_indicator = processed_full[(processed_full.date<'2020-07-01') & (processed_full.date>='2009-01-01')]
    insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=['date'])
    insample_risk_indicator.vix.describe()
    insample_risk_indicator.vix.quantile(0.996)
    insample_risk_indicator.turbulence.describe()
    insample_risk_indicator.turbulence.quantile(0.996)
    e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
    print(f"trade.head(): {trade.head()}")
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_model,
        environment = e_trade_gym)
    
    print(f"df_account_value.shape: {df_account_value.shape}")
    print(f"df_account_value.tail(): {df_account_value.tail()}")
    print(f"df_actions.head(): {df_actions.head()}")


    print("==============Get Backtest Results===========")
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

    perf_stats_all = backtest_stats(account_value=df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')


    print("==============Get Baseline Stats===========")
    #baseline_df = get_baseline(
    #        ticker="^DJI",
    #        start = df_account_value.loc[0,'date'],
    #        end = df_account_value.loc[len(df_account_value)-1,'date'])
    baseline_df = DataReader.get_ohlcv(
        ['^DJI'], df_account_value.loc[0,'date'], 
        df_account_value.loc[len(df_account_value)-1,'date'],
        version=1)
    stats = backtest_stats(baseline_df, value_col_name = 'close')

    df_account_value.loc[0,'date']
    df_account_value.loc[len(df_account_value)-1,'date']


    print("==============Compare to DJIA===========")
    # %matplotlib inline
    # S&P 500: ^GSPC
    # Dow Jones Index: ^DJI
    # NASDAQ 100: ^NDX
    backtest_plot(df_account_value,
                 baseline_ticker = '^DJI',
                 baseline_start = df_account_value.loc[0,'date'],
                 baseline_end = df_account_value.loc[len(df_account_value)-1,'date'])
    
    for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig('figure%d.png' % i)
    #plt.show()
    #plt.savefig("mygraph.png")

'''
### Trade

DRL model needs to update periodically in order to take full advantage of the data, ideally we need to retrain our model yearly, quarterly, or monthly. We also need to tune the parameters along the way, in this notebook I only use the in-sample data from 2009-01 to 2020-07 to tune the parameters once, so there is some alpha decay here as the length of trade date extends.

Numerous hyperparameters – e.g. the learning rate, the total number of samples to train on – influence the learning process and are usually determined by testing some variations.

'''

#trade = data_split(processed_full, '2020-07-01','2021-10-31')
# env_trade, obs_trade = e_trade_gym.get_sb_env()




""" df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_sac,
    environment = e_trade_gym) """
""" from stable_baselines3 import SAC

sac_model = SAC.load(os.path.join(config.TRAINED_MODEL_DIR, "sac_20220616-1720.model"))
df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=sac_model,
    environment = e_trade_gym)  """

""" df_account_value, df_actions = DRLAgent.DRL_prediction_load_from_file(
    model_name="sac",
    cwd=os.path.join(config.TRAINED_MODEL_DIR, "sac_20220616-1720.model"),
    environment = e_trade_gym) """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose",
                        help="set logging level to DEBUG", action='store_true')
    parser.add_argument("-t", "--ticks", type=str,
                        help="a list of tickers separated by comma", default='DOW30')
    parser.add_argument("-a", "--action", type=str,
                        help="action: stat, price", default='')
    parser.add_argument("-s", "--start", type=str,
                        help="start date of daily price", default='2009-01-01')
    parser.add_argument("-e", "--end", type=str,
                        help="end date of daily price", default='2021-10-31')
    parser.add_argument("-m", "--model-name", type=str,
                        help="model name", default='')
    parser.add_argument("-r", "--force-refresh",
                        help="force refreshing all ticks", action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    if args.ticks == 'DOW30':
        ticks = config_tickers.DOW_30_TICKER
    else:
        ticks = args.ticks.split(',')

    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

    back_test(args.model_name, ticks, args.start, args.end)