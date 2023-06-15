import os
import requests
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import time
import re
from tqdm import tqdm


class BitcoinData:
    def __init__(self, db_path='sqlite:///btc_data.db'):
        self.db_path = db_path
        self.engine = create_engine(db_path)
        self.url = "https://api.binance.com/api/v3/klines"

    def fetch_and_store_data(self, start_date=datetime(2021, 1, 1), end_date=datetime.now(), delta_days=1000):
        data = []
        delta = timedelta(days=delta_days)
        while start_date < end_date:
            temp_end_date = min(start_date + delta, end_date)
            params = {
                "symbol": "BTCUSDT",
                "interval": "1d",
                "startTime": int(start_date.timestamp() * 1000),
                "endTime": int(temp_end_date.timestamp() * 1000),
            }
            response = requests.get(self.url, params=params)
            if response.status_code == 200:  # successful response
                response_data = response.json()
                if response_data:  # non-empty response
                    data.extend(response_data)
                    # get the latest date from the response data
                    latest_data_date = max([int(item[0]) for item in response_data])
                    # convert the timestamp to datetime and add one day
                    start_date = datetime.fromtimestamp(latest_data_date / 1000) + timedelta(days=1)
                    time.sleep(1)

        if data:
            df = pd.DataFrame(data, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                             'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                             'Taker buy quote asset volume', 'Ignore'])

            # Converting to datetime and dropping duplicates
            df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
            df = df.drop_duplicates(subset=['Open time'])

            df.to_sql('btc_data', self.engine, if_exists='replace', index=False)


class TweetData:
    def __init__(self, file_paths, sample_frac=0.09, clean_file_path='./datasets/Bitcoin_tweets_clean.csv'):
        self.file_paths = file_paths
        self.sample_frac = sample_frac
        self.clean_file_path = clean_file_path
        self.df = None

    def load_data(self):
        # Check if the cleaned file already exists
        if os.path.isfile(self.clean_file_path):
            self.df = pd.read_csv(self.clean_file_path)
            return

        # If the cleaned file doesn't exist, proceed with loading and cleaning
        chunks_list = []
        for file_path in self.file_paths:
            for chunk in pd.read_csv(file_path, chunksize=100000, engine='python'):
                chunks_list.append(chunk)
        self.df = pd.concat(chunks_list).drop_duplicates().dropna()

    def clean_data(self):
        # If the cleaned file exists, skip cleaning
        if os.path.isfile(self.clean_file_path):
            return

        # If the cleaned file doesn't exist, proceed with cleaning
        self.df = self.df.sort_values(by='date')
        self.df = self.df.sample(frac=self.sample_frac, replace=False, random_state=1)
        self.df.reset_index(inplace=True)

        for i, s in enumerate(tqdm(self.df['text'], position=0, leave=True)):
            text = str(self.df.loc[i, 'text'])
            text = text.replace("#", "")
            text = re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', text, flags=re.MULTILINE)
            text = re.sub('@\\w+ *', '', text, flags=re.MULTILINE)
            self.df.loc[i, 'text'] = text

        self.df.to_csv(self.clean_file_path, header=True, encoding='utf-8', index=False)



