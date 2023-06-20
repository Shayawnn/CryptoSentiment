import math
import nltk
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from keras_tuner.tuners import RandomSearch
from keras.callbacks import EarlyStopping

from database import *
from visualize import DataVisualizer

nltk.download('vader_lexicon')


class SentimentAnalysis:
    def __init__(self):
        self.test_rmse = None
        self.train_rmse = None
        self.y_test_pred = None
        self.model = None
        self.X_test = None
        self.y_train_pred = None
        self.X_train = None
        self.X = None
        self.scaler_sentiment = None
        self.scaler_close = None
        self.tweet_data = TweetData(['./datasets/Bitcoin_tweets.csv', './datasets/Bitcoin_tweets_dataset_2.csv'])
        self.df = None
        self.bitcoin_data = BitcoinData()
        self.bitcoin_df = None
        self.data = None
        self.sid = SentimentIntensityAnalyzer()
        self.correlation = None
        self.lagged_correlation = None
        self.reg = None
        self.poly_reg = None

    # @staticmethod
    # def analyze_sentiment(row):
    #     sentiment_score = TextBlob(row['text']).sentiment.polarity
    #     is_retweet = int(row['is_retweet'])
    #     user_followers = np.log1p(int(float(row['user_followers'])))  # Apply logarithm transformation
    #     user_favourites = np.log1p(int(float(row['user_favourites'])))  # Apply logarithm transformation
    #
    #     # Calculate a score based on followers, favorites, and retweet
    #     user_score = ((user_followers + 1) * (user_favourites + 1)) / ((user_followers + 1) * (is_retweet + 1))
    #
    #     weighted_sentiment_score = sentiment_score * user_score
    #
    #     return weighted_sentiment_score

    def analyze_sentiment(self, row):
        sentiment_score = self.sid.polarity_scores(row['text'])['compound']
        is_retweet = int(row['is_retweet'])
        user_followers = np.log1p(int(float(row['user_followers'])))  # Apply logarithm transformation
        user_favourites = np.log1p(int(float(row['user_favourites'])))  # Apply logarithm transformation

        # Calculate a score based on followers, favorites, and retweet
        user_score = ((user_followers + 1) * (user_favourites + 1)) / ((user_followers + 1) * (is_retweet + 1))

        weighted_sentiment_score = sentiment_score * user_score

        return weighted_sentiment_score

    def prepare_tweet_data(self):
        # Convert TRUE/FALSE to 1/0
        self.tweet_data.df['is_retweet'] = self.tweet_data.df['is_retweet'].map({True: 1, False: 0})

        self.tweet_data.df['sentiment'] = self.tweet_data.df.apply(self.analyze_sentiment, axis=1)

        # Convert to datetime.date
        self.tweet_data.df['date'] = pd.to_datetime(self.tweet_data.df['date']).dt.date

    def load_bitcoin_data(self):
        self.bitcoin_df = pd.read_sql_table('btc_data', self.bitcoin_data.engine)

        # Convert to datetime.date
        self.bitcoin_df['Open time'] = pd.to_datetime(self.bitcoin_df['Open time']).dt.date

    def trim_data(self):
        # Group by date and calculate mean sentiment
        tweet_sentiment = self.tweet_data.df.groupby('date')['sentiment'].mean()

        # Find the latest start date and the earliest end date
        start_date = max(min(self.tweet_data.df['date']), min(self.bitcoin_df['Open time']))
        end_date = min(max(self.tweet_data.df['date']), max(self.bitcoin_df['Open time']))

        # Trim both dataframes to these dates
        tweet_sentiment = tweet_sentiment[(tweet_sentiment.index >= start_date) & (tweet_sentiment.index <= end_date)]
        self.bitcoin_df = self.bitcoin_df[
            (self.bitcoin_df['Open time'] >= start_date) & (self.bitcoin_df['Open time'] <= end_date)]

        # Join with tweet sentiment
        self.data = self.bitcoin_df.set_index('Open time').join(tweet_sentiment)

        # Drop rows with NaN values
        self.data = self.data.dropna()

        # Convert the 'Close' column to float
        self.data['Close'] = self.data['Close'].astype(float)

    def calculate_correlation(self):
        self.correlation = np.corrcoef(self.data['Close'], self.data['sentiment'])[0, 1]

    def calculate_lagged_correlation(self):
        self.lagged_correlation = self.data['Close'].corr(self.data['sentiment'].shift(-1))

    def fit_linear_regression(self):
        X = self.data[['sentiment']]
        y = self.data['Close']
        reg = LinearRegression().fit(X, y)
        self.reg = reg.score(X, y)

    def fit_polynomial_regression(self):
        X = self.data[['sentiment']]
        y = self.data['Close']
        poly_reg = make_pipeline(PolynomialFeatures(2), LinearRegression())
        poly_reg.fit(X, y)
        self.poly_reg = poly_reg.score(X, y)

    def build_model(self, hp):
        model = Sequential()
        model.add(LSTM(units=hp.Int('units_input', min_value=32, max_value=512, step=32),
                       return_sequences=True,
                       input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=False))
        model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(Dense(1))

        model.compile(optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
            loss='mean_squared_error')

        return model

    def prepare_lstm_model(self):
        self.scaler_close = MinMaxScaler(feature_range=(0, 1))
        self.scaler_sentiment = MinMaxScaler(feature_range=(0, 1))

        # Normalize the 'Close' column and 'sentiment' column
        self.data['Close_normalized'] = self.scaler_close.fit_transform(self.data['Close'].values.reshape(-1, 1))
        self.data['sentiment_normalized'] = self.scaler_sentiment.fit_transform(
            self.data['sentiment'].values.reshape(-1, 1))

        # Prepare input data for LSTM
        self.X = self.data[['Close_normalized', 'sentiment_normalized']].values

        # Split data into training and testing sets (80% training, 20% testing)
        train_len = int(0.8 * len(self.X))
        self.X_train, self.X_test = self.X[:train_len], self.X[train_len:]

        # Reshape data to 3D for LSTM input
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.X_test.shape[1])

    def train_lstm_model(self):
        tuner = RandomSearch(
            self.build_model,
            objective='loss',
            max_trials=6,
            executions_per_trial=3
        )

        early_stopping_tuner = EarlyStopping(monitor='val_loss', patience=6)  # adjust 'patience' as needed
        tuner.search(self.X_train, self.X_train[:, :, 0], epochs=12, validation_split=0.1, callbacks=[early_stopping_tuner])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        self.model = tuner.hypermodel.build(best_hps)

        early_stopping_fit = EarlyStopping(monitor='val_loss', patience=21)  # adjust 'patience' as needed
        self.model.fit(self.X_train, self.X_train[:, :, 0], epochs=100, batch_size=12, validation_split=0.1, callbacks=[early_stopping_fit])

    def evaluate_lstm_model(self):
        # Predict with the model
        self.y_train_pred = self.model.predict(self.X_train)
        self.y_test_pred = self.model.predict(self.X_test)

        # Inverse transform the predicted data
        self.y_train_pred = self.scaler_close.inverse_transform(self.y_train_pred)
        self.y_test_pred = self.scaler_close.inverse_transform(self.y_test_pred)

        return self.y_train_pred, self.y_test_pred

    def calculate_rmse(self):
        # Inverse transform the target Close price from the train and test dataset
        y_train_true = self.scaler_close.inverse_transform(self.X_train[:, :, 0])
        y_test_true = self.scaler_close.inverse_transform(self.X_test[:, :, 0])

        # Calculate RMSE for training data
        self.train_rmse = math.sqrt(mean_squared_error(y_train_true, self.y_train_pred))

        # Calculate RMSE for testing data
        self.test_rmse = math.sqrt(mean_squared_error(y_test_true, self.y_test_pred))

        return self.train_rmse, self.test_rmse

    # Getters
    def get_data(self):
        return self.data

    def get_correlation(self):
        return self.correlation

    def get_lagged_correlation(self):
        return self.lagged_correlation

    def get_linear_regression(self):
        return self.reg

    def get_polynomial_regression(self):
        return self.poly_reg


if __name__ == '__main__':
    # Instantiate the SentimentAnalysis class
    sa = SentimentAnalysis()

    # Fetch and store bitcoin data
    sa.bitcoin_data.fetch_and_store_data()

    # Load and Clean tweet data
    sa.tweet_data.load_data()
    sa.tweet_data.clean_data()

    # Prepare and load data
    sa.prepare_tweet_data()
    sa.load_bitcoin_data()

    # Trim data, calculate correlation and fit regression models
    sa.trim_data()
    sa.calculate_correlation()
    sa.calculate_lagged_correlation()
    sa.fit_linear_regression()
    sa.fit_polynomial_regression()

    data = sa.get_data()
    correlation = sa.get_correlation()
    correlation_lagged = sa.get_lagged_correlation()
    reg = sa.get_linear_regression()
    poly_reg = sa.get_polynomial_regression()

    # Train and evaluate the LSTM model
    sa.prepare_lstm_model()
    sa.train_lstm_model()
    y_train_pred, y_test_pred = sa.evaluate_lstm_model()

    # Calculate correlation
    print(f"Correlation between bitcoin price and sentiment: {correlation}")

    # Calculate lagged correlation
    print(f'Correlation between Bitcoin price and lagged Twitter sentiment: {correlation_lagged}')

    # Fit a linear regression model
    print(f"Regression score: {reg}")

    # Descriptive statistics
    print("\nDescriptive statistics for sentiment scores:")
    print(data['sentiment'].describe())

    print("\nDescriptive statistics for Bitcoin prices:")
    print(data['Close'].describe())

    # Polynomial regression
    print(f"Polynomial regression score: {poly_reg}")

    # Calculate RMSE "Root Mean Squared Error"
    train_rmse, test_rmse = sa.calculate_rmse()
    print(f'Training RMSE: {train_rmse}, Testing RMSE: {test_rmse}')

    # visualizing the line plot
    DataVisualizer.plot_line(data.reset_index(), 'Open time', 'Close', 'sentiment',
                             'Bitcoin Price and Sentiment over Time', show=True)

    # visualizing the correlation matrix
    DataVisualizer.plot_correlation(data, show=True)

    # visualizing the scatter plot
    DataVisualizer.plot_scatter(data, 'sentiment', 'Close', 'Scatter Plot of Sentiment vs Bitcoin Price', show=True)

    # visualizing the distribution plot
    DataVisualizer.plot_dist(data, 'sentiment', 'Distribution of Sentiment Scores', show=True)

    # visualizing the count plot
    DataVisualizer.plot_sentiment_bins(data, 'sentiment', show=True)

    # visualizing the line plot
    DataVisualizer.plot_line_overtime(data.reset_index(), 'Open time', 'sentiment', show=True)

    # visualizing the LSTM model results
    DataVisualizer.plot_lstm_results(data.reset_index(), 'Open time', 'Close', y_train_pred, y_test_pred,
                                     'Bitcoin Price and Sentiment over Time with LSTM Prediction', show=True)

