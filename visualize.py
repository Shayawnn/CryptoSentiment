import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
plt.style.use('fivethirtyeight')


class DataVisualizer:
    @staticmethod
    def plot_line(df, x, y1, y2, title, show=False):
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df[x], df[y1], label=y1, color='blue')
        ax.set_xlabel('Date')
        ax.set_ylabel('Bitcoin Price', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        ax2 = ax.twinx()
        ax2.plot(df[x], df[y2], label=y2, color='green')
        ax2.set_ylabel('Tweet Sentiment', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    @staticmethod
    def plot_correlation(df, show=False):
        corr = df[['sentiment', 'Number of trades', 'Close', 'Volume']].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        ax.set_title("Correlation Matrix")
        if show:
            plt.show()
        return fig

    @staticmethod
    def plot_scatter(df, x, y, title, show=False):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df[x], df[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        if show:
            plt.show()
        return fig

    @staticmethod
    def plot_dist(df, column, title, show=False):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.distplot(df[column])
        ax.set_xlabel(column)
        ax.set_ylabel("Density")
        ax.set_title(title)
        if show:
            plt.show()
        return fig

    @staticmethod
    def plot_sentiment_bins(df, column, show=False):
        # Define bins based on the provided descriptive statistics
        bins = [-np.inf, 0.9, 1.2, np.inf]
        labels = ["Negative: Less than 25th %ile", "Neutral", "Positive: Greater than 50th %ile"]

        # Create a new column for sentiment category
        df['sentiment_category'] = pd.cut(df[column], bins=bins, labels=labels)

        # Calculate the proportions
        proportions = df['sentiment_category'].value_counts(normalize=True)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='sentiment_category', data=df, order=labels, ax=ax)
        ax.set_title('Sentiment Category Counts')
        ax.set_xlabel('Sentiment Category')
        ax.set_ylabel('Count')

        # Add proportions to the bars
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2.,
                    height + 0.01,
                    '{:1.2f}'.format(proportions[labels[i]] * 100) + "%",
                    ha="center")
        if show:
            plt.show()
        return fig

    @staticmethod
    def plot_line_overtime(df, x, y, show=False):
        # Make subplots with 2 rows
        fig = make_subplots(rows=2, shared_xaxes=True, vertical_spacing=0.03)

        # Candlestick chart for bitcoin prices
        fig.add_trace(go.Candlestick(x=df[x],
                                     open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'],
                                     name='Bitcoin prices'),
                      row=1, col=1)

        # Scatter plot for sentiment scores
        fig.add_trace(go.Bar(x=df[x], y=df[y],
                             name='Sentiment'),
                      row=2, col=1)

        fig.update_layout(height=600, width=800,
                          title_text="Interactive candlestick chart of Bitcoin prices and sentiment scores")

        if show:
            fig.show()
        return fig

    @staticmethod
    def plot_lstm_results(df, x, y, y_train_pred, y_test_pred, title, show=False):
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(df[x], df[y], label='Actual Price', color='blue')
        ax.plot(df[x][:len(y_train_pred)], y_train_pred, label='LSTM Train', color='orange')
        ax.plot(df[x][len(y_train_pred):len(y_train_pred) + len(y_test_pred)], y_test_pred, label='LSTM Test', color='green')

        ax.set_xlabel('Date')
        ax.set_ylabel('Bitcoin Price', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        ax.set_title(title)
        ax.legend()

        fig.tight_layout()

        if show:
            plt.show()

        return fig

