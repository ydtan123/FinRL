#!/usr/bin/env python
# coding: utf-8
import argparse
from django import db
from django.core.exceptions import ValidationError
from django.db.utils import OperationalError
import prediction.django_conf
from prediction.models import StreamPrice

from finrl.train import train
from finrl.test import test
from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS
from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.finrl_meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading
from finrl.finrl_meta.data_processor import DataProcessor
from finrl.finrl_meta.data_processors.processor_alpaca import AlpacaProcessor
from finrl.finrl_meta.preprocessor.preprocessors import DataReader
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
import logging
import math
import numpy as np
import pandas as pd
from pathlib import Path

from sqlalchemy import create_engine

from datetime import datetime
import threading
import alpaca_trade_api as tradeapi
import time
import torch
import gym

import exchange_calendars as tc
import pytz
from pprint import pprint
import yfinance as yf
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from datetime import datetime as dt
import matplotlib.pyplot as plt
import os
import pytz
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


API_KEY = "AKFH2IC3PO7ZSVPCVDA5"
API_SECRET = "SBy3J98it6uciS5inYR9Kjpo4WWet6FPc51JoQkQ"

PAPER_API_KEY = "PKHVT9PL25YHLWQZPX85"
PAPER_API_SECRET = "vihuXTdDum0Pm60VhK8uJOQH2r0n5Zv4g9Pkj9so"

API_BASE_URL = 'https://paper-api.alpaca.markets'
data_url = 'wss://data.alpaca.markets'

ticker_list = DOW_30_TICKER
action_dim = len(DOW_30_TICKER)
candle_time_interval = '15Min'

class AlpacaPaperSimulator(object):

    def fetch_and_save_alpaca_data(self, start, end, fromdb=False):
        if not fromdb:
            DP = DataProcessor(
                data_source = 'alpaca',
                API_KEY = API_KEY, 
                API_SECRET = API_SECRET, 
                API_BASE_URL = API_BASE_URL)
            data = DP.download_data(
                start_date = start, 
                end_date = end,
                ticker_list = ticker_list, 
                time_interval= candle_time_interval)
            #data.to_csv("raw.csv")
            data = pd.read_csv('raw.csv')
            print(F"Downloaded data: {data.shape}")
            
            data = DP.clean_data(data)
            data.to_csv("cleaned.csv")
            print(F"Cleaned data: {data.shape}")

            data = DP.add_technical_indicator(data, INDICATORS)
            data.to_csv("ti.csv")
            print(F"Added ti: {data.shape}")

            data = DP.add_vix(data)
            print(F"Added vix: {data.shape}")
            data.to_csv('vix.csv')
            return
        else:
            data = pd.read_csv("alpaca_with_features.csv")
            data = data.drop("Unnamed: 0", axis=1)
        
        eastern = pytz.timezone('America/New_York')
        for i  in range(data.shape[0]):
            rec = data.iloc[i].to_dict()
            if isinstance(rec['timestamp'], str):
                t = datetime.fromisoformat(rec['timestamp'])
            else:
                t = pd.to_datetime(rec['timestamp'])
            if t.tzinfo is None:
                t = t.replace(tzinfo=pytz.utc)
            rec['timestamp'] = t.astimezone(eastern).strftime('%Y-%m-%d %H:%M')

            try:
                StreamPrice(**rec).save()
            except (ValidationError, OperationalError) as e:
                #pprint(rec)
                for k, v in rec.items():
                    if not isinstance(v, str) and (math.isnan(v) or math.isinf(v)):
                        rec[k] = 0
                StreamPrice(**rec).save()
            except Exception as e:
                print(F"{rec['tic']}, {rec['timestamp']} orig: {data.iloc[i]['timestamp']}")
                pprint(e)
        return data

    def fetch_alpaca_data_from_db(self, ticks, start, end):
        DP = DataProcessor(
                data_source = 'alpaca',
                technical_indicator_list=INDICATORS,
                API_KEY = API_KEY, 
                API_SECRET = API_SECRET, 
                API_BASE_URL = API_BASE_URL)
        data = DataReader.get_ohlcv(ticks, start, end, version=2)
        data = DP.add_technical_indicator(data, INDICATORS)
        data = self.add_vix(data, start, end)
        data.to_csv('daily.csv')
        return data

    def add_vix(self, data, start, end):
        vix_df = DataReader.get_ohlcv(['^VIX'], start, end, version=2)
        vix = vix_df[['timestamp', 'close']].rename(columns={"close": "VIXY"})

        df = data.copy()
        df = df.merge(vix, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def split_data(self, data, split_date):
        train_data = data.loc[data['timestamp']<split_date]
        test_data  = data.loc[data['timestamp']>=split_date]
        return train_data, test_data

    def train(self, data):
        ERL_PARAMS = {
            "learning_rate": 3e-6,
            "batch_size": 2048,
            "gamma":  0.985,
            "seed":312,
            "net_dimension":512, 
            "target_step":5000, 
            "eval_gap":30,
            "eval_times":1}
        start_date = data.iloc[0]['timestamp']
        end_date = data.iloc[-1]['timestamp']
        timestamp = F"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{datetime.now().strftime('%Y%m%d%H%M')}"
        model_dir = os.path.join("models", F"{timestamp}")
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        print(F"Train data from {start_date} to {end_date}")
        train(start_date = start_date, 
              end_date = end_date,
              ticker_list = ticker_list, 
              data_source = 'alpaca',
              time_interval= candle_time_interval, 
              technical_indicator_list= INDICATORS,
              drl_lib='elegantrl', 
              env=env,
              model_name='ppo', 
              data=data,
              API_KEY = API_KEY, 
              API_SECRET = API_SECRET, 
              API_BASE_URL = API_BASE_URL,
              erl_params=ERL_PARAMS,
              cwd=model_dir, #'./papertrading_erl', #current_working_dir
              break_step=4e6)
        print(F"Finish training, model saved in {model_dir}!")
        return timestamp

    def test(self, data, model_dir):
        model_dir = os.path.join("models", model_dir)
        print(F"Loading model from {model_dir}")
        account_value_erl=test(
            data,
            start_date = '2021-10-18', 
            end_date = '2021-10-19',
            ticker_list = ticker_list, 
            data_source = 'alpaca',
            time_interval= candle_time_interval, 
            technical_indicator_list= INDICATORS,
            drl_lib='elegantrl', 
            env=env, 
            model_name='ppo', 
            API_KEY = API_KEY, 
            API_SECRET = API_SECRET, 
            API_BASE_URL = API_BASE_URL,
            cwd=model_dir, #'./papertrading_erl',
            net_dimension = 512)
        return account_value_erl

# amount + (turbulence, turbulence_bool) + (price, shares, cd (holding time)) * stock_dim + tech_dim
state_dim = 1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim
env = StockTradingEnv


class AlpacaPaperTrading():

    def __init__(self,ticker_list, time_interval, drl_lib, agent, cwd, net_dim, 
                 state_dim, action_dim, API_KEY, API_SECRET, 
                 API_BASE_URL, tech_indicator_list, turbulence_thresh=30, 
                 max_stock=1e2, latency = None):
        #load agent
        self.drl_lib = drl_lib
        if agent =='ppo':
            if drl_lib == 'elegantrl':              
                from elegantrl.agents import AgentPPO
                from elegantrl.train.run import init_agent
                from elegantrl.train.config import Arguments
                #load agent
                config = {'state_dim':state_dim,
                            'action_dim':action_dim,}
                args = Arguments(agent=AgentPPO, env=StockEnvEmpty(config))
                args.cwd = cwd
                args.net_dim = net_dim
                # load agent
                try:
                    agent = init_agent(args, gpu_id = 0)
                    self.act = agent.act
                    self.device = agent.device
                except BaseException:
                    raise ValueError("Fail to load agent!")
                        
            elif drl_lib == 'rllib':
                from ray.rllib.agents import ppo
                from ray.rllib.agents.ppo.ppo import PPOTrainer
                
                config = ppo.DEFAULT_CONFIG.copy()
                config['env'] = StockEnvEmpty
                config["log_level"] = "WARN"
                config['env_config'] = {'state_dim':state_dim,
                            'action_dim':action_dim,}
                trainer = PPOTrainer(env=StockEnvEmpty, config=config)
                trainer.restore(cwd)
                try:
                    trainer.restore(cwd)
                    self.agent = trainer
                    print("Restoring from checkpoint path", cwd)
                except:
                    raise ValueError('Fail to load agent!')
                    
            elif drl_lib == 'stable_baselines3':
                from stable_baselines3 import PPO
                
                try:
                    #load agent
                    self.model = PPO.load(cwd)
                    print("Successfully load model", cwd)
                except:
                    raise ValueError('Fail to load agent!')
                    
            else:
                raise ValueError('The DRL library input is NOT supported yet. Please check your input.')
               
        else:
            raise ValueError('Agent input is NOT supported yet.')
            
            
            
        #connect to Alpaca trading API
        try:
            self.alpaca = tradeapi.REST(API_KEY,API_SECRET,API_BASE_URL, 'v2')
        except:
            raise ValueError('Fail to connect Alpaca. Please check account info and internet connection.')
        
        #read trading time interval
        if time_interval == '1s':
            self.time_interval = 1
        elif time_interval == '5s':
            self.time_interval = 5
        elif time_interval == candle_time_interval:
            self.time_interval = 60
        elif time_interval == '5Min':
            self.time_interval = 60 * 5
        elif time_interval == '15Min':
            self.time_interval = 60 * 15
        else:
            raise ValueError('Time interval input is NOT supported yet.')
        
        #read trading settings
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_thresh = turbulence_thresh
        self.max_stock = max_stock 
        
        #initialize account
        self.stocks = np.asarray([0] * len(ticker_list)) #stocks holding
        self.stocks_cd = np.zeros_like(self.stocks) 
        self.cash = None #cash record 
        self.stocks_df = pd.DataFrame(self.stocks, columns=['stocks'], index = ticker_list)
        self.asset_list = []
        self.price = np.asarray([0] * len(ticker_list))
        self.stockUniverse = ticker_list
        self.turbulence_bool = 0
        self.equities = []
        
    def test_latency(self, test_times = 10): 
        total_time = 0
        for i in range(0, test_times):
            time0 = time.time()
            self.get_state()
            time1 = time.time()
            temp_time = time1 - time0
            total_time += temp_time
        latency = total_time/test_times
        print('latency for data processing: ', latency)
        return latency
        
    def run(self):
        orders = self.alpaca.list_orders(status="open")
        for order in orders:
          self.alpaca.cancel_order(order.id)
    
        # Wait for market to open.
        print("Waiting for market to open...")
        tAMO = threading.Thread(target=self.awaitMarketOpen)
        tAMO.start()
        tAMO.join()
        print("Market opened.")
        while True:

          # Figure out when the market will close so we can prepare to sell beforehand.
          clock = self.alpaca.get_clock()
          closingTime = clock.next_close.replace(tzinfo=datetime.timezone.utc).timestamp()
          currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
          self.timeToClose = closingTime - currTime
    
          if(self.timeToClose < (60)):
            # Close all positions when 1 minutes til market close.
            print("Market closing soon. Stop trading.")
            break
            
            '''# Close all positions when 1 minutes til market close.
            print("Market closing soon.  Closing positions.")
    
            positions = self.alpaca.list_positions()
            for position in positions:
              if(position.side == 'long'):
                orderSide = 'sell'
              else:
                orderSide = 'buy'
              qty = abs(int(float(position.qty)))
              respSO = []
              tSubmitOrder = threading.Thread(target=self.submitOrder(qty, position.symbol, orderSide, respSO))
              tSubmitOrder.start()
              tSubmitOrder.join()
    
            # Run script again after market close for next trading day.
            print("Sleeping until market close (15 minutes).")
            time.sleep(60 * 15)'''
            
          else:
            trade = threading.Thread(target=self.trade)
            trade.start()
            trade.join()
            last_equity = float(self.alpaca.get_account().last_equity)
            cur_time = time.time()
            self.equities.append([cur_time,last_equity])
            time.sleep(self.time_interval)
            
    def awaitMarketOpen(self):
        isOpen = self.alpaca.get_clock().is_open
        while(not isOpen):
          clock = self.alpaca.get_clock()
          openingTime = clock.next_open.replace(tzinfo=datetime.timezone.utc).timestamp()
          currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
          timeToOpen = int((openingTime - currTime) / 60)
          print(str(timeToOpen) + " minutes til market open.")
          time.sleep(60)
          isOpen = self.alpaca.get_clock().is_open
    
    def trade(self):
        state = self.get_state()
        
        if self.drl_lib == 'elegantrl':
            with torch.no_grad():
                s_tensor = torch.as_tensor((state,), device=self.device)
                a_tensor = self.act(s_tensor)  
                action = a_tensor.detach().cpu().numpy()[0]  
                
            action = (action * self.max_stock).astype(int)
            
        elif self.drl_lib == 'rllib':
            action = self.agent.compute_single_action(state)
        
        elif self.drl_lib == 'stable_baselines3':
            action = self.model.predict(state)[0]
            
        else:
            raise ValueError('The DRL library input is NOT supported yet. Please check your input.')
        
        self.stocks_cd += 1
        if self.turbulence_bool == 0:
            min_action = 10  # stock_cd
            for index in np.where(action < -min_action)[0]:  # sell_index:
                sell_num_shares = min(self.stocks[index], -action[index])
                qty =  abs(int(sell_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(target=self.submitOrder(qty, self.stockUniverse[index], 'sell', respSO))
                tSubmitOrder.start()
                tSubmitOrder.join()
                self.cash = float(self.alpaca.get_account().cash)
                self.stocks_cd[index] = 0

            for index in np.where(action > min_action)[0]:  # buy_index:
                if self.cash < 0:
                    tmp_cash = 0
                else:
                    tmp_cash = self.cash
                buy_num_shares = min(tmp_cash // self.price[index], abs(int(action[index])))
                qty = abs(int(buy_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(target=self.submitOrder(qty, self.stockUniverse[index], 'buy', respSO))
                tSubmitOrder.start()
                tSubmitOrder.join()
                self.cash = float(self.alpaca.get_account().cash)
                self.stocks_cd[index] = 0
                
        else:  # sell all when turbulence
            positions = self.alpaca.list_positions()
            for position in positions:
                if(position.side == 'long'):
                    orderSide = 'sell'
                else:
                    orderSide = 'buy'
                qty = abs(int(float(position.qty)))
                respSO = []
                tSubmitOrder = threading.Thread(target=self.submitOrder(qty, position.symbol, orderSide, respSO))
                tSubmitOrder.start()
                tSubmitOrder.join()
            
            self.stocks_cd[:] = 0
            
    
    def get_state(self):
        alpaca = AlpacaProcessor(api=self.alpaca)
        price, tech, turbulence = alpaca.fetch_latest_data(ticker_list = self.stockUniverse, time_interval=candle_time_interval,
                                                     tech_indicator_list=self.tech_indicator_list)
        turbulence_bool = 1 if turbulence >= self.turbulence_thresh else 0
        
        turbulence = (self.sigmoid_sign(turbulence, self.turbulence_thresh) * 2 ** -5).astype(np.float32)
        
        tech = tech * 2 ** -7
        positions = self.alpaca.list_positions()
        stocks = [0] * len(self.stockUniverse)
        for position in positions:
            ind = self.stockUniverse.index(position.symbol)
            stocks[ind] = ( abs(int(float(position.qty))))
        
        stocks = np.asarray(stocks, dtype = float)
        cash = float(self.alpaca.get_account().cash)
        self.cash = cash
        self.stocks = stocks
        self.turbulence_bool = turbulence_bool 
        self.price = price

        amount = np.array(self.cash * (2 ** -12), dtype=np.float32)
        scale = np.array(2 ** -6, dtype=np.float32)
        state = np.hstack((amount,
                    turbulence,
                    self.turbulence_bool,
                    price * scale,
                    self.stocks * scale,
                    self.stocks_cd,
                    tech,
                    )).astype(np.float32)
        print(len(self.stockUniverse))
        return state
        
    def submitOrder(self, qty, stock, side, resp):
        if(qty > 0):
          try:
            self.alpaca.submit_order(stock, qty, side, "market", "day")
            print("Market order of | " + str(qty) + " " + stock + " " + side + " | completed.")
            resp.append(True)
          except:
            print("Order of | " + str(qty) + " " + stock + " " + side + " | did not go through.")
            resp.append(False)
        else:
          print("Quantity is 0, order of | " + str(qty) + " " + stock + " " + side + " | not completed.")
          resp.append(True)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh
    
class StockEnvEmpty(gym.Env):
    #Empty Env used for loading rllib agent
    def __init__(self,config):
      state_dim = config['state_dim']
      action_dim = config['action_dim']
      self.env_num = 1
      self.max_step = 10000
      self.env_name = 'StockEnvEmpty'
      self.state_dim = state_dim  
      self.action_dim = action_dim
      self.if_discrete = False  
      self.target_return = 9999
      self.observation_space = gym.spaces.Box(low=-3000, high=3000, shape=(state_dim,), dtype=np.float32)
      self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        
    def reset(self):
        return 

    def step(self, actions):
        return


def run_paper_trading():
    paper_trading_erl = AlpacaPaperTrading(ticker_list = DOW_30_TICKER, 
                                       time_interval = candle_time_interval, 
                                       drl_lib = 'elegantrl', 
                                       agent = 'ppo', 
                                       cwd = './papertrading_erl_retrain', 
                                       net_dim = 512, 
                                       state_dim = state_dim, 
                                       action_dim= action_dim, 
                                       API_KEY = PAPER_API_KEY, 
                                       API_SECRET = PAPER_API_SECRET, 
                                       API_BASE_URL = API_BASE_URL, 
                                       tech_indicator_list = INDICATORS, 
                                       turbulence_thresh=30, 
                                       max_stock=1e2)
    paper_trading_erl.run()


def get_trading_days(start, end):
    nyse = tc.get_calendar('NYSE')
    df = nyse.sessions_in_range(pd.Timestamp(start,tz=pytz.UTC),
                                pd.Timestamp(end,tz=pytz.UTC))
    trading_days = []
    for day in df:
        trading_days.append(str(day)[:10])

    return trading_days

def alpaca_history(key, secret, url, start, end):
    api = tradeapi.REST(key, secret, url, 'v2')
    trading_days = get_trading_days(start, end)
    df = pd.DataFrame()
    for day in trading_days:
        df = df.append(api.get_portfolio_history(date_start = day,timeframe='5Min').df.iloc[:78])
    equities = df.equity.values
    cumu_returns = equities/equities[0]
    cumu_returns = cumu_returns[~np.isnan(cumu_returns)]
    
    return df, cumu_returns

def DIA_history(start):
    data_df = yf.download(['^DJI'],start=start, interval="5m")
    data_df = data_df.iloc[48:]
    baseline_returns = data_df['Adj Close'].values/data_df['Adj Close'].values[0]
    return data_df, baseline_returns


# ## Get cumulative return
def get_cumulative_return():

    history_start_date='2022-04-15'
    history_end_date='2022-05-10'

    df_erl, cumu_erl = alpaca_history(key=API_KEY, 
                                      secret=API_SECRET, 
                                      url=API_BASE_URL, 
                                      start=history_start_date, #must be within 1 month
                                      end='2021-10-22') #change the date if error occurs


    df_djia, cumu_djia = DIA_history(start=history_start_date)

    print("====DF ERL====")
    print(df_erl)
    print("====DF DJI====")
    print(df_djia)
    returns_erl = cumu_erl -1 
    returns_dia = cumu_djia - 1
    returns_dia = returns_dia[:returns_erl.shape[0]]
    print('len of erl return: ', returns_erl.shape[0])
    print('len of dia return: ', returns_dia.shape[0])

    plt.figure(dpi=1000)
    plt.grid()
    plt.grid(which='minor', axis='y')
    plt.title('Stock Trading (Paper trading)', fontsize=20)
    plt.plot(returns_erl, label = 'ElegantRL Agent', color = 'red')
    #plt.plot(returns_sb3, label = 'Stable-Baselines3 Agent', color = 'blue' )
    #plt.plot(returns_rllib, label = 'RLlib Agent', color = 'green')
    plt.plot(returns_dia, label = 'DJIA', color = 'grey')
    plt.ylabel('Return', fontsize=16)
    plt.xlabel('Year 2021', fontsize=16)
    plt.xticks(size = 14)
    plt.yticks(size = 14)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(78))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(6))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(['','10-19','','10-20',
                                                        '','10-21','','10-22']))
    plt.legend(fontsize=10.5)
    plt.savefig('papertrading_stock.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose",
                        help="set logging level to DEBUG", action='store_true')
    parser.add_argument("-t", "--ticks", type=str,
                        help="a list of tickers separated by comma", default='')
    parser.add_argument("-a", "--action", type=str,
                        help="action: stat, price", default='')
    parser.add_argument("-s", "--start", type=str,
                        help="start date of daily price", default='2021-10-11')
    parser.add_argument("-e", "--end", type=str,
                        help="end date of daily price", default='2021-10-19')
    parser.add_argument("-m", "--model-name", type=str,
                        help="model name", default='')
    parser.add_argument("--split-date", type=str,
                        help="split date from testing data", default='')
    parser.add_argument("--train",
                        help="train", action='store_true')
    parser.add_argument("--test",
                        help="test", action='store_true')
    parser.add_argument("-r", "--force-refresh",
                        help="force refreshing all ticks", action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    sim = AlpacaPaperSimulator()
    data = sim.fetch_alpaca_data_from_db(DOW_30_TICKER, args.start, args.end)

    if args.train:
        train_data, test_data = sim.split_data(data, args.split_date)
        print(F"Train a model using data from {train_data.iloc[0]['timestamp']} to {train_data.iloc[-1]['timestamp']}")

        model_dir = sim.train(train_data)

        test_start = test_data.iloc[0]['timestamp']
        test_end = test_data.iloc[-1]['timestamp']
        print(F"Test a model using data from {test_start} to {test_end}")
        results = sim.test(test_data, model_dir)

        trading_days = get_trading_days(test_start, test_end)
        results = pd.DataFrame({"date": trading_days, "return": results})
        results.to_csv("results.csv")
        results.plot(x='date', y='return')
        plt.savefig('return.jpg')

    elif args.test:
        train_data, test_data = sim.split_data(data, args.split_date)

        test_start = test_data.iloc[0]['timestamp']
        test_end = test_data.iloc[-1]['timestamp']
        print(F"Load model from {args.model_name}")
        print(F"Test a model using data from {test_start} to {test_end}")

        results = sim.test(test_data, args.model_name)

        trading_days = get_trading_days(test_start, test_end)
        results = pd.DataFrame({"date": trading_days, "return": results})
        results.to_csv("results_test.csv")
        results.plot(x='date', y='return')
        plt.savefig('return_test.jpg')
