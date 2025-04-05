#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")

import datetime
import itertools
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from pprint import pprint

from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent,DRLEnsembleAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

sys.path.append("../FinRL-Library")

from data_reader import DataReader
from finrl.main import check_and_make_directories
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.config_tickers import DOW_30_TICKER

def prepare_data(ticks, indicators, start_date, end_date):
    df = DataReader('prediction_dailyprice').get_data_for(ticks, start_date, end_date)

    df.rename(columns={
        'Tick': 'tic',
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'AdjVolume': 'adj_volume'}, inplace=True)
    print(df.head())

    fe = FeatureEngineer(use_technical_indicator=True,
                        tech_indicator_list = indicators,
                        use_turbulence=True,
                        user_defined_feature = False)

    processed = fe.preprocess_data(df)
    processed = processed.copy()
    processed = processed.fillna(0)
    processed = processed.replace(np.inf,0)

    return processed


with open('ensemble.config') as f:
    config = json.load(f)

check_and_make_directories(
    [config["DATA_SAVE_DIR"],
     config["TRAINED_MODEL_DIR"],
     config["TENSORBOARD_LOG_DIR"],
     config["RESULTS_DIR"]])

processed = prepare_data(
    DOW_30_TICKER,
    indicators=config["INDICATORS"],
    start_date=config['TRAIN_START_DATE'],
    end_date=config['TEST_END_DATE'])

stock_dimension = len(processed.tic.unique())
state_space = 1 + 2*stock_dimension + len(config['INDICATORS'])*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config["INDICATORS"],
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "print_verbosity":5
}

rebalance_window = 63 # rebalance_window is the number of days to retrain the model
validation_window = 63 # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)
ensemble_agent = DRLEnsembleAgent(
    df=processed,
    train_period=(config["TRAIN_START_DATE"], config["TRAIN_END_DATE"]),
    val_test_period=(config["TEST_START_DATE"], config["TEST_END_DATE"]),
    rebalance_window=rebalance_window,
    validation_window=validation_window,
    **env_kwargs)


from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3
df_summary = ensemble_agent.run_ensemble_strategy(
    config['A2C_model_kwargs'],
    config['PPO_model_kwargs'],
    config['DDPG_model_kwargs'],
    config['SAC_model_kwargs'],
    config['TD3_model_kwargs'],
    config['timesteps_dict'],
    MODELS={"a2c": A2C, "ppo": PPO, "ddpg": DDPG, "sac": SAC, "td3": TD3},)

unique_trade_date = processed[(processed.date > config["TEST_START_DATE"])&(processed.date <= config["TEST_END_DATE"])].date.unique()

df_trade_date = pd.DataFrame({'datadate':unique_trade_date})

df_account_value=pd.DataFrame()
for i in range(rebalance_window+validation_window, len(unique_trade_date)+1,rebalance_window):
    csv_file = 'results/account_value_trade_{}_{}.csv'.format('ensemble',i)
    temp = pd.read_csv(csv_file)
    print(f"append temp: {temp.shape} from {csv_file}, pd shape: {df_account_value.shape}")
    df_account_value = pd.concat([df_account_value, temp], ignore_index=True)

    #df_account_value = df_account_value.append(temp,ignore_index=True)
sharpe=(252**0.5)*df_account_value.account_value.pct_change(1).mean()/df_account_value.account_value.pct_change(1).std()
print('Sharpe Ratio: ',sharpe)
df_account_value=df_account_value.join(df_trade_date[validation_window:].reset_index(drop=True))


df_account_value.head()

df_account_value.account_value.plot()


# <a id='6.1'></a>
# ## 7.1 BackTestStats
# pass in df_account_value, this information is stored in env class
# 

# In[ ]:


print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)


# In[ ]:


#baseline stats
print("==============Get Baseline Stats===========")
#df_dji_ = get_baseline(
#        ticker="^DJI",
#        start = df_account_value.loc[0,'date'],
#        end = df_account_value.loc[len(df_account_value)-1,'date'])

df_dji_ = DataReader('prediction_dailyprice').get_data_for(
    ['AAPL'], start_date = df_account_value.loc[0,'date'],
        end_date = df_account_value.loc[len(df_account_value)-1,'date'])
df_dji_ = df_dji_.rename(columns={
    'Tick': 'tic',
    'Date': 'date',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume',
    'AdjVolume': 'adj_volume'})
stats = backtest_stats(df_dji_, value_col_name = 'close')


# In[ ]:


df_dji = pd.DataFrame()
df_dji['date'] = df_account_value['date']
df_dji['dji'] = (df_dji_['close'] / df_dji_['close'].iloc[0] * env_kwargs["initial_amount"]).to_list()
print("df_dji: ", df_dji)
df_dji.to_csv("df_dji.csv")
df_dji = df_dji.set_index(df_dji.columns[0])
print("df_dji: ", df_dji)
df_dji.to_csv("df_dji+.csv")

df_account_value.to_csv('df_account_value.csv')


# <a id='6.2'></a>
# ## 7.2 BackTestPlot

# In[ ]:


# print("==============Compare to DJIA===========")
# %matplotlib inline
# # S&P 500: ^GSPC
# # Dow Jones Index: ^DJI
# # NASDAQ 100: ^NDX
# backtest_plot(df_account_value,
#               baseline_ticker = '^DJI',
#               baseline_start = df_account_value.loc[0,'date'],
#               baseline_end = df_account_value.loc[len(df_account_value)-1,'date'])
df.to_csv("df.csv")
df_result_ensemble = pd.DataFrame({'date': df_account_value['date'], 'ensemble':df_account_value['account_value']})
df_result_ensemble = df_result_ensemble.set_index('date')

print("df_result_ensemble.columns: ", df_result_ensemble.columns)

print("df_trade_date: ", df_trade_date)
# df_result_ensemble['date'] = df_trade_date['datadate']
# df_result_ensemble['account_value'] = df_account_value['account_value']
df_result_ensemble.to_csv("df_result_ensemble.csv")
print("df_result_ensemble: ", df_result_ensemble)
print("==============Compare to DJIA===========")
result = pd.DataFrame()
# result = pd.merge(result, df_result_ensemble, left_index=True, right_index=True)

result = pd.merge(df_result_ensemble, df_dji, left_index=True, right_index=True)
print("result: ", result)
result.to_csv("result.csv")
result.columns = ['ensemble', 'dji']

plt.rcParams["figure.figsize"] = (15, 5)
plt.figure()
result.plot()
plt.show()


# 
