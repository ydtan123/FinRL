#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")

import argparse
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
from stable_baselines3.common.logger import configure
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

sys.path.append("../FinRL-Library")

from common.data_reader import DataReader
from finrl.main import check_and_make_directories
from finrl.config_tickers import DOW_30_TICKER

models_map = {
    "a2c": A2C,
    "ppo": PPO,
    "ddpg": DDPG,
    "td3": TD3,
    "sac": SAC
}

model_params = {
    "a2c": {},
    "ppo": {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128},
    "ddpg": {},
    "td3": {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001},
    "sac": {"batch_size": 129, "buffer_size": 100000, "learning_rate": 0.0001, "learning_starts": 100, "ent_coef": "auto_0.1"}
}

def prepare_data(config):
    processed = DataReader('prediction_dailyprice').prepare_data(
        DOW_30_TICKER + ['^VIX'],
        indicators=config["INDICATORS"],
        start_date=config['TRAIN_START_DATE'],
        end_date=config['TRADE_END_DATE'])

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])

    processed_full = processed_full.fillna(0)
    print(processed_full.head())
    processed_full.to_csv(config["DATA_SAVE_DIR"] + '/processed_full.csv', index=False)
    return processed_full


def train(data, config, model_name):

    print("Training the model...")
    stock_dimension = len(data.tic.unique())
    state_space = 1 + 2*stock_dimension + len(config['INDICATORS'])*stock_dimension
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
        "tech_indicator_list": config["INDICATORS"],
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    e_train_gym = StockTradingEnv(df = data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))

    agent = DRLAgent(env = env_train)

    model = agent.get_model(model_name, model_kwargs=model_params[model_name])

    tmp_path = os.path.join(config["RESULTS_DIR"], model_name)
    new_logger= configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    trained = agent.train_model(
        model=model, tb_log_name='a2c', total_timesteps=50000)
    trained.save(os.path.join(config["TRAINED_MODEL_DIR"], model_name))
    print(f"Model {model_name} trained and saved to {config['TRAINED_MODEL_DIR']}/{model_name}")
    print("Training completed.")
    print("===============================================")
    

def trade(data, config, model_name):
    print("================== Backtest =====================")
    stock_dimension = len(data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(config["INDICATORS"]) * stock_dimension
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
        "tech_indicator_list": config["INDICATORS"],
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    e_trade_gym = StockTradingEnv(
        df = data, 
        make_plots=True,
        turbulence_threshold = 70, risk_indicator_col='vix', **env_kwargs)
    model = models_map[model_name].load(
        os.path.join(config["TRAINED_MODEL_DIR"], model_name))
    print(f"Model {model_name} loaded from {config['TRAINED_MODEL_DIR']}/{model_name}")
    df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
        model=model, 
        environment = e_trade_gym)
    df_account_value_a2c.to_csv(
        os.path.join(config["RESULTS_DIR"], f"account_value_{model_name}.csv"))
    df_actions_a2c.to_csv(
        os.path.join(config["RESULTS_DIR"], f"actions_{model_name}.csv"))
    print("Backtesting completed.")
    print("===============================================")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Stock Trading with DRL')
    parser.add_argument('--config', type=str, default='drl.config', help='Path to the config file')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--backtest', action='store_true', help='Backtest the model')
    parser.add_argument('--model', type=str, default='a2c', help='Model to use (a2c, ppo, ddpg, td3, sac)')

    args = parser.parse_args()

    try:
        with open(args.config) as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Config file {args.config} not found. Please provide a valid config file.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from config file {args.config}. Please check the file format.")
        sys.exit(1)
    except Exception as e: 
        print(f"An error occurred while loading the config file {args.config}: {e}.")
        sys.exit(1)
    
    processed_full = prepare_data(config)

    stock_dimension = len(processed_full.tic.unique())
    state_space = 1 + 2*stock_dimension + len(config['INDICATORS'])*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    train_data = data_split(processed_full, config["TRAIN_START_DATE"], config["TRAIN_END_DATE"])
    trade_data = data_split(processed_full, config["TRADE_START_DATE"], config["TRADE_END_DATE"])


    if args.train:
        train(train_data, config, args.model)
    elif args.test:
        print("Testing the model...")
    elif args.backtest:
        print("Backtesting the model...")
        trade(trade_data, config, args.model)
    else:
        print("Please specify --train, --test, or --backtest.")

