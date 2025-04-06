#!/usr/bin/env python
import argparse
from datetime import datetime
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

class DataReader(object):
    def __init__(self, table_name):
        self.data_table = self.read_from_db(table_name)

    def read_from_db(self, table_name):
        engine = create_engine('postgresql+psycopg://django:django@localhost/stock')
        data = pd.read_sql_table(table_name, engine) #'prediction_dailyprice'
        return data

    def get_data_for(self, ticks, start_date=None, end_date=None):

        start_date = datetime(1999, 1, 1) if start_date is None else datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime(2024, 12, 31) if end_date is None else datetime.strptime(end_date, '%Y-%m-%d')
        print(f'Getting data for {ticks} from {start_date} to {end_date}')

        df = self.data_table.loc[
            (self.data_table['Tick'].isin(ticks)) & (self.data_table['Date'] >= start_date) & (self.data_table['Date'] <= end_date),
            ['Tick', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjVolume']]
        df.dropna(inplace=True)
        
        print('--------------------------------------------------')
        print(df.head())
        print('--------------------------------------------------')
        return df
    

    def prepare_data(self, ticks, indicators, start_date, end_date):
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
        print(f'Indicators: {indicators}')
        fe = FeatureEngineer(use_technical_indicator=True,
                            tech_indicator_list = indicators,
                            use_turbulence=True,
                            user_defined_feature = False)

        processed = fe.preprocess_data(df)
        processed = processed.copy()
        processed = processed.fillna(0)
        processed = processed.replace(np.inf,0)

        return processed
    

    def dry_run(self, table_name, tick, start_date=None, end_date=None):
        '''
        Pass data through a cerebro engine and do nothing,
        Then plot the original data.
        '''

        data_table = self.read_from_db(table_name)
        data = self.get_data_for(data_table, tick, start_date, end_date)
        print(data.head())
       

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process stock data.')
    parser.add_argument('--tick', type=str, default='MSFT', help='The tick name')
    parser.add_argument('--table', type=str, default='prediction_dailyprice', help='The table name')
    args = parser.parse_args()

    data_reader = DataReader()
    data_reader.dry_run(args.table, args.tick, start_date='2020-01-01', end_date='2021-01-01') 
    
