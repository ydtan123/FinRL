#!/usr/bin/env python
from cProfile import label
import warnings
warnings.filterwarnings("ignore")

import argparse
from datetime import datetime, timedelta
from django.db.utils import OperationalError

from glob import glob
import itertools
# matplotlib.use('Agg')

from finrl import config
from finrl import config_tickers
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, DataReader, data_split
from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent,DRLEnsembleAgent, MODELS
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from pprint import pprint
import pytz
from stable_baselines3.common.vec_env import DummyVecEnv
from trade.td_trader import Trader
import sys
import talib
from talib import abstract
from talib import stream
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.trend import CCIIndicator
from ta.trend import ADXIndicator

import schedule
from schedule import every, repeat, run_pending, CancelJob, idle_seconds
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s :: %(levelname)s :: %(message)s')

#sys.path.append("../FinRL-Library")
tech_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30']

def dir_setup():
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    results_dir = os.path.join(config.RESULTS_DIR, F"{timestamp}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    config.TRAINED_MODEL_DIR = os.path.join(config.TRAINED_MODEL_DIR, F"{timestamp}")
    Path(config.TRAINED_MODEL_DIR).mkdir(parents=True, exist_ok=True)

    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)

    return results_dir

def prepare_data(ticks, start_date='2009-04-01', end_date='2022-06-01'):

    df = DataReader.get_ohlcv(
        ticks, start_date, end_date, version=1).sort_values(['date','tic'],ignore_index=True)
    print(F"Total ticks: {len(df.tic.unique())}, total prices: {len(df.index.unique())}")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list = tech_indicators,
        use_turbulence=True,
        user_defined_feature = False)

    processed = fe.preprocess_data(df)
    processed = processed.copy()
    processed = processed.fillna(0)
    processed = processed.replace(np.inf,0)

    processed.sample(5)
    return processed

def get_params(data):
    stock_dimension = len(data.tic.unique())
    state_space = 1 + 2*stock_dimension + len(tech_indicators)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 1000000, 
        "buy_cost_pct": 0.001, 
        "sell_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": tech_indicators,
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4,
        "print_verbosity":1
    }
    return env_kwargs

def train_ensemble(data, start, end, val_split, results_dir):
    rebalance_window = 63 # rebalance_window is the number of days to retrain the model
    validation_window = 63 # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)
    #train_start = '2009-04-01'
    #train_end = '2020-04-01'
    #val_test_start = '2020-04-01'
    #val_test_end = '2022-06-01'
    train_start = start
    train_end = val_split
    val_start = val_split
    val_end = end

    env_kwargs, _, _ = get_params(data)
    ensemble_agent = DRLEnsembleAgent(df=data,
                     train_period=(train_start,train_end),
                     val_test_period=(val_start,val_end),
                     rebalance_window=rebalance_window, 
                     validation_window=validation_window,
                     results_dir=results_dir,
                     **env_kwargs)

    A2C_model_kwargs = {
                        'n_steps': 5,
                        'ent_coef': 0.01,
                        'learning_rate': 0.0005
                        }

    PPO_model_kwargs = {
                        "ent_coef":0.01,
                        "n_steps": 2048,
                        "learning_rate": 0.00025,
                        "batch_size": 64
                        }

    DDPG_model_kwargs = {
                          #"action_noise":"ornstein_uhlenbeck",
                          "buffer_size": 10_000,
                          "learning_rate": 0.0005,
                          "batch_size": 64
                        }

    timesteps_dict = {'a2c' : 10_000, 
                     'ppo' : 10_000, 
                     'ddpg' : 10_000
                     }

    df_summary = ensemble_agent.run_ensemble_strategy(
        A2C_model_kwargs, 
        PPO_model_kwargs,
        DDPG_model_kwargs,
        timesteps_dict)

    df_summary

    unique_trade_date = data[(data.date > val_start)&(data.date <= val_end)].date.unique()
    df_trade_date = pd.DataFrame({'datadate':unique_trade_date})

    df_account_value=pd.DataFrame()
    for i in range(rebalance_window+validation_window, len(unique_trade_date)+1,rebalance_window):
        temp = pd.read_csv(os.path.join(config.RESULTS_DIR, F'account_value_trade_ensemble_{i}.csv'))
        df_account_value = df_account_value.append(temp,ignore_index=True)
    sharpe=(252**0.5)*df_account_value.account_value.pct_change(1).mean()/df_account_value.account_value.pct_change(1).std()
    print('Sharpe Ratio: ',sharpe)
    df_account_value=df_account_value.join(df_trade_date[validation_window:].reset_index(drop=True))

    df_account_value.head()

    df_account_value.account_value.plot()

    print("==============Get Backtest Results===========")
    now = datetime.now().strftime('%Y%m%d-%Hh%M')

    perf_stats_all = backtest_stats(account_value=df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)

    print("==============Get Baseline Stats===========")
    baseline_df = DataReader.get_ohlcv(
        ['^DJI'], df_account_value.loc[0,'date'], 
        df_account_value.loc[len(df_account_value)-1,'date'],
        version=1)

    stats = backtest_stats(baseline_df, value_col_name = 'close')

    print("==============Compare to DJIA===========")
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


def trade(model_dir, model_name, df, trade_start_date, results_dir):
    last_date = df.iloc[-1].date

    for model_file in get_lastest_model_file(model_dir):
        mname = os.path.basename(model_file).split('_')[0].lower()
        if mname != model_name:
            continue
        trade_one_model(model_name, model_file, df, trade_start_date, last_date, results_dir)
    
    print("==============Get Baseline Stats===========")
    print(F"Trading dates: {trade_start_date} - {last_date}")
    baseline_df = DataReader.get_ohlcv(['^DJI'], trade_start_date, last_date, version=1)
    stats = backtest_stats(baseline_df, value_col_name = 'close')


def trade_one_model(model_name, model_file, df, start_date, end_date, results_dir):
    trained_model = MODELS[model_name].load(model_file)

    last_state = []

    #insample_turbulence = df[(df.date < end_date) & (df.date >= start_date)]
    turbulence= np.quantile(df.turbulence.values, 0.90)
    logging.info(F"Find turbulence from {start_date} to {end_date}: {turbulence}")

    ## trading env
    trade_data = data_split(
        df,
        start=start_date,
        end=end_date,
    )

    env_kwargs = get_params(trade_data)

    #results_dir = os.path.join(config.RESULTS_DIR, datetime.now().strftime('%Y%m%d%H%M'))
    initial = True
    ticks = trade_data.tic.unique()

    dates = trade_data.date.unique()
    days = np.where(dates>=start_date)
    assert len(days) != 0
    logging.info(F"Start trade from {start_date}, day {days[0][0]} to {dates[-1]}, day {len(trade_data.index.unique())}")
    for i in range(days[0][0], len(trade_data.index.unique())):
        last_state, action = trade_one_day(
            model_name, 
            trained_model,
            trade_data.loc[:i+1, :], 
            env_kwargs,
            last_state, 
            turbulence,
            results_dir, 
            initial,
            day=i)
        #actions = pd.concat([actions, pd.DataFrame(columns=ticks, data=action)])
        initial = False

    df_last_state = pd.DataFrame({"last_state": last_state})
    df_last_state.to_csv(
        os.path.join(results_dir, f"last_state_{model_name}_{i}.csv"), 
        index=False)
    #actions.to_csv('actions.csv')
    return last_state

def trade_one_day(
    model_name, 
    trained_model, 
    trade_data, 
    env_kwargs, 
    last_state, 
    turbulence, 
    results_dir, 
    initial,
    day):
    trade_env = DummyVecEnv(
        [
            lambda: StockTradingEnv(
                df=trade_data,
                stock_dim=env_kwargs['stock_dim'],
                hmax=env_kwargs["hmax"],
                initial_amount=env_kwargs["initial_amount"],
                num_stock_shares=[0]*env_kwargs['stock_dim'],
                buy_cost_pct=[env_kwargs["buy_cost_pct"]]*env_kwargs['stock_dim'],
                sell_cost_pct=[env_kwargs["sell_cost_pct"]]*env_kwargs['stock_dim'],
                reward_scaling=env_kwargs["reward_scaling"],
                state_space=env_kwargs["state_space"],
                action_space=env_kwargs["action_space"],
                tech_indicator_list=env_kwargs["tech_indicator_list"],
                turbulence_threshold=turbulence,
                initial=initial,
                previous_state=last_state,
                model_name=model_name,
                mode="trade",
                iteration=0,
                print_verbosity=env_kwargs["print_verbosity"],
                results_dir=results_dir,
                day=day
            )
        ]
    )

    trade_obs = trade_env.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = trained_model.predict(trade_obs)
        trade_obs, rewards, dones, info = trade_env.step(action)
        #if i == (len(trade_data.index.unique()) - 2):
            # print(env_test.render())
        last_state = trade_env.render()
        #pprint(action)
    return last_state, action


def get_lastest_model_file(model_dir):
    files = sorted([os.path.basename(f) for f in glob(os.path.join(model_dir, '*.zip'))])
    for _, f in itertools.groupby(files, lambda x: x.split('_')[0]):
        yield os.path.join(model_dir, sorted(list(f), key=lambda x: x.split('_'))[-1])


def compare_stat(data, func_name, ss_col, needhl=False, **kwargs):
    data = data.sort_values(by=['tic', 'date'])
    #func = getattr(talib, func_name) 
    func = abstract.Function(func_name)
    sfunc = getattr(stream, func_name)
    rsi_data = pd.DataFrame(columns=['tic', 'stockstat', 'ta'])
    for tick in ['AAPL']: #data.tic.unique():
        tick_data = data.loc[data['tic']==tick]
        num_samples = tick_data.shape[0]
        tmp_data = tick_data.iloc[:-100]
        if needhl:
            ta_data = func(tmp_data['high'], tmp_data['low'], tmp_data['close'], **kwargs)
        else:
            ta_data = func(tmp_data['close'], **kwargs)
        print(F"{tick}, {ss_col}, {len(ta_data)}")
        if func_name == 'MACD':
            ta_data = ta_data[0]

        stream_results = []
        for i in range(num_samples-100, num_samples):
            sdata = tick_data.iloc[:i]
            if needhl:
                val_arr = func(sdata['high'], sdata['low'], sdata['close'], **kwargs)
                val = sfunc(sdata['high'], sdata['low'], sdata['close'], **kwargs)
            else:
                import ipdb; ipdb.set_trace()
                val_arr = func(sdata['close'], **kwargs)
                val = sfunc(sdata['close'], **kwargs)
                #print(F"Calling RSI: ta: {val_arr[-1]}, stream: {val}")
                
            if func_name == 'MACD':
                val_arr = val_arr[0]
                val = val[0]
            #assert val_arr[-1]==val, F'{tick}/{ss_col}: ta={val_arr[-1]}, stream={val}'

            stream_results.append(val)

        ta_data = np.append(ta_data, stream_results)
        print(F'length of ta data = {len(ta_data)}')
        diff = tick_data.iloc[-100:]['close'] - pd.Series(stream_results)
        print(F"{tick}/{ss_col}: min={diff.min()}, max={diff.max()}")
        rsi_data = pd.concat([rsi_data, pd.DataFrame(
            {'tic': tick_data['tic'], 'stockstat': tick_data[ss_col], 'ta': ta_data})],
            ignore_index=False)
        if tick == 'AAPL':
            plt.plot(tick_data['date'], tick_data[ss_col], label=F'ss_{ss_col}')
            plt.plot(tick_data['date'], ta_data, label=F'ta_{ss_col}')
            plt.legend()
            plt.show()
    rsi_data.to_csv(F'{ss_col}.csv')

def trade_job(ticks, model_dir, model_name):
    yesterday = datetime.today() - timedelta(days=2)
    ystr = yesterday.strftime('%Y-%m-%d')
    first_day = datetime.today() - timedelta(days=500)  #must greater than one year, turbulence needs a window of 252 in front
    fstr = first_day.strftime('%Y-%m-%d')
    logging.info(F"Sequence from {fstr} to {ystr}")
    data = prepare_data(ticks, fstr, ystr)

    results_dir = os.path.join(config.RESULTS_DIR, F"trade_{datetime.now().strftime('%Y%m%d%H%M')}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    trade_start = datetime.today() - timedelta(days=3)
    trade(model_dir, model_name, data, trade_start.strftime('%Y-%m-%d'), results_dir)

def to_nyc_time(hour, minute):
    nyc_tz = pytz.timezone('US/Eastern')
    today = datetime.today()
    hm = nyc_tz.localize(datetime(today.year, today.month, today.day, hour, minute)).astimezone()
    return hm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose",
                        help="set logging level to DEBUG", action='store_true')
    parser.add_argument("-t", "--ticks", type=str,
                        help="a list of tickers separated by comma", default='DOW30')
    parser.add_argument("-s", "--start", type=str,
                        help="start date of daily price", default='2009-01-01')
    parser.add_argument("-e", "--end", type=str,
                        help="end date of daily price", default='2022-06-10')
    parser.add_argument("--split-date", type=str,
                        help="split date of daily price", default='2020-04-01')
    parser.add_argument("--schedule", type=str,
                        help="split date of daily price", default='')
    parser.add_argument("-m", "--model-name", type=str, help="model name", default='')
    parser.add_argument("--model-dir", type=str, help="directories to find model files", default='')
    parser.add_argument("--train", help="train a model first", action='store_true')
    parser.add_argument("--trade", help="trade", action='store_true')
    parser.add_argument("--run-job", help="run the trading job online", action='store_true')
    parser.add_argument("--run-now", help="run the trading job now", action='store_true')
    parser.add_argument("--compare", help="trade", action='store_true')
    args = parser.parse_args()

    if args.ticks == 'DOW30' or args.ticks == '':
        ticks = config_tickers.DOW_30_TICKER
    else:
        ticks = args.ticks.split(',')


    """state of environment:
        self.initial_amount]
        self.data.close.values.tolist()
        self.num_stock_shares
        sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
    """
    if args.train:
        data = prepare_data(ticks, args.start, args.end)
        results_dir = dir_setup()
        train_ensemble(data, args.start, args.end, args.split_date, results_dir)
    elif args.trade:
        data = prepare_data(ticks, args.start, args.end)
        results_dir = os.path.join(config.RESULTS_DIR, F"trade_{datetime.now().strftime('%Y%m%d%H%M')}")
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        trade(args.model_dir, data, args.split_date, args.end, results_dir)
    elif args.run_now:
        trade_job(ticks=ticks, model_dir=args.model_dir, model_name='ppo')
    elif args.run_job:
        if args.schedule != '':
            hm = datetime.strptime(args.schedule, "%H:%M")
        else:
            # by default, run tasks at 16:25 NYC time, i.e. 5 minutes before market close
            hm = to_nyc_time(15, 55)
        hmstr = hm.strftime('%H:%M')
        logging.info(F"Schedule running update_price task at {hmstr}")
        every().day.at(hmstr).do(Trader.update_price, ticks=ticks)

        hm = hm + timedelta(minutes=1)
        hmstr = hm.strftime('%H:%M')
        logging.info(F"Schedule running trade task at {hmstr}")

        every().day.at(hmstr).do(
            trade_job, 
            ticks=ticks, 
            model_dir=args.model_dir, 
            model_name='ppo')

        while True:
            n = idle_seconds()
            if n is None:
                logging.info("No more tasks in the queue. Quit!")
                break
            if n > 0:
                logging.info(F"Sleep {n} seconds before running the next task")
                # sleep exactly the right amount of time
                time.sleep(n)
            try:
                run_pending()
            except Exception as e:
                schedule.clear('temp-trade-job')  # in case nested exception, clear all previous temp jobs
                logging.info(F"Reschedule task because of django DB error: {e}")
                every(1).minutes.do(
                    trade_job, 
                ticks=ticks, 
                model_dir=args.model_dir, 
                model_name='ppo',
                start_date=args.split_date,
                end_date=args.end).tag('temp-trade-job')
                    
    elif args.compare:
        """compare tech indicator between stockstat and ta"""
        #data.to_csv("all.csv")
        data = prepare_data(ticks, args.start, args.end)
        data = data.sort_values(by=['tic', 'date'])

        compare_stat(data,  'RSI', 'rsi_30', timeperiod=30)
        #compare_stat(data, 'MACD', 'macd')
        #compare_stat(data, 'CCI', 'cci_30', needhl=True)
        #compare_stat(data, 'DX', 'dx_30', needhl=True)