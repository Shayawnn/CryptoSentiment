import streamlit as st
from main import *

# Instantiate the SentimentAnalysis class
sa = SentimentAnalysis()

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

# Now you can get data and models to use in your Streamlit app
data = sa.get_data()
correlation = sa.get_correlation()
correlation_lagged = sa.get_lagged_correlation()
reg = sa.get_linear_regression()
poly_reg = sa.get_polynomial_regression()

# Train and evaluate the LSTM model
sa.prepare_lstm_model()
sa.train_lstm_model()
y_train_pred, y_test_pred = sa.evaluate_lstm_model()

# Calculate RMSE "Root Mean Squared Error"
train_rmse, test_rmse = sa.calculate_rmse()

# Page setup
st.set_page_config(layout="wide")

# Introduction
st.title('Bitcoin Sentiment Analysis')
st.write("")

col1, col2, col3 = st.columns([1, 8, 1])

with col1:
    st.write("")

with col2:
    st.image("./Bitcoin Sentiment Analysis.jpg")

with col3:
    st.write("")

# Introduction
st.header('Introduction')
st.write("Bitcoin, the first and largest cryptocurrency, has gained tremendous popularity and attention from investors, enthusiasts, and researchers over the past decade. Its volatile price movements have intrigued people from various fields, fueling attempts to predict its future trends. Among the several factors that may influence Bitcoin's price, the sentiment in the cryptocurrency community, particularly as expressed on social media platforms, has been of particular interest. This study explores the relationship between Bitcoin prices and sentiment scores derived from Twitter data, utilizing a range of statistical techniques and machine learning models, including regression analysis and Long Short-Term Memory (LSTM) networks.")
st.write("The objective of this research is to gain insights into the potential predictive power of sentiment scores on Bitcoin price movements. We examine the descriptive statistics of the sentiment scores and Bitcoin prices, as well as their correlation. Subsequently, we investigate the applicability and efficacy of LSTM, a type of Recurrent Neural Network (RNN) known for its ability to handle sequential data, in forecasting Bitcoin prices based on sentiment scores. Our findings offer valuable perspectives on the interplay between social sentiment and Bitcoin prices, and the utility of machine learning in cryptocurrency price prediction.")

# Load and display data
st.header('Data Overview')
st.dataframe(data)

# Descriptive statistics
st.header('Descriptive Statistics')
st.write("The descriptive statistics provide some insights into the distribution of both sentiment scores and Bitcoin prices.")

# Descriptive statistics
st.header('Descriptive Statistics for Sentiment Scores')
st.write("This is a summary of the sentiment scores that we've obtained from Twitter data.")
st.write(data['sentiment'].describe())
st.write("The average (mean) sentiment score is 1.42 (approximately), with a standard deviation of 0.45. The minimum score is 0.40 and the maximum score is 3.68. The 25th, 50th (median), and 75th percentiles are also provided.")

st.header('Descriptive Statistics for Bitcoin Prices')
st.write("This is a similar summary, but for Bitcoin prices.")
st.write(data['Close'].describe())
st.write("We have 231 observations, with an average price of approximately 37,019.38. The standard deviation is large at 13,753.93, indicating substantial variability in Bitcoin prices. The minimum price is 15,781.29 and the maximum is 66,001.41.")

# Distribution plot
st.header('Distribution of Sentiment Scores')
st.write(
    "It might be useful to visualize the distribution of the sentiment scores.")
fig5 = DataVisualizer.plot_dist(data, 'sentiment', 'Distribution of Sentiment Scores')
st.pyplot(fig5)
st.write(
    "For instance, you might find that extreme sentiment scores (either very positive or very negative).")

# Count plot
st.header('Count Plot of Sentiment Scores')
st.write(
    "We define three bins for negative (less than -0.9), neutral (between -0.9 and 1.2), and positive (greater than 1.2) sentiment scores. Which categorizes the sentiment scores according to these bins.")
fig6 = DataVisualizer.plot_sentiment_bins(data, 'sentiment')
st.pyplot(fig6)
st.write(
    "Here in the count plot, x-axis represents the sentiment category and y-axis represents the count of tweets in each category.")

# Line plot
st.header('Bitcoin Price and Sentiment over Time')
st.write(
    "Let's start with a simple line plot of Bitcoin prices over time. On a secondary y-axis, we'll plot the tweet sentiment:")
fig1 = DataVisualizer.plot_line(data.reset_index(), 'Open time', 'Close', 'sentiment',
                                'Bitcoin Price and Sentiment over Time')
st.pyplot(fig1)
st.write(
    "This will create a plot with Bitcoin prices on the left y-axis and tweet sentiment on the right y-axis. You can visually examine if the two lines tend to move together.")

# Interactive Line plot
st.header('Bitcoin Price and Sentiment over Time')
st.write(
    "We can plot the sentiment and price over time to see if there are specific periods of positive or negative sentiment that correspond with price increases or decreases.")
fig7 = DataVisualizer.plot_line_overtime(data.reset_index(), 'Open time', 'sentiment')
st.plotly_chart(fig7, use_container_width=True)
st.write(
    "This interactive subplot enables you to zoom in, zoom out, pan, and hover for detailed information over the charts.")

# Scatter plot
st.header('Scatter Plot of Sentiment vs Bitcoin Price')
st.write(
    "Let's create a scatterplot of tweet sentiment (x-axis) and Bitcoin price (y-axis) to see if a pattern exists:")
fig3 = DataVisualizer.plot_scatter(data, 'sentiment', 'Close', 'Scatter Plot of Sentiment vs Bitcoin Price')
st.pyplot(fig3)
st.write("If there's a strong correlation, you'll see a clear upward or downward trend in the scatterplot.")

# Distribution results
st.header('Distribution Results')
st.subheader("What is the distribution of sentiment scores in the Twitter data?")
st.write("The sentiment scores in the Twitter data have a mean of approximately 1.42 and a standard deviation of 0.45. The scores range from a minimum of 0.40 to a maximum of 3.68.")
st.subheader("What is the distribution of Bitcoin prices in the data?")
st.write("Bitcoin prices in the data have a mean of approximately 37,019.38 and a standard deviation of 13,753.93. The prices range from a minimum of 15,781.29 to a maximum of 66,001.41.")

# Display the correlation
st.header('Correlation')
st.write("This will output a number between -1 and 1. A positive number indicates a positive correlation (when sentiment is high, Bitcoin price tends to be higher), while a negative number indicates a negative correlation. A number close to 0 indicates no correlation.")
st.success(f"Correlation between bitcoin price and sentiment: {correlation}")
st.write("The correlation coefficient of approximately 0.39 indicates a moderate positive relationship between Bitcoin prices and sentiment scores. This suggests that higher sentiment scores are somewhat associated with higher Bitcoin prices.")

# Correlation matrix
st.header('Correlation Matrix')
st.write(
    "This will create a correlation heatmap where each cell in the grid represents the correlation between two variables. The color of the cell represents the strength and direction of the correlation:")
fig2 = DataVisualizer.plot_correlation(data)
st.pyplot(fig2)
st.write(
    "Light colors represent strong positive correlations, dark colors represent strong negative correlations, and colors close to white represent weak or no correlation.")

# Correlation results
st.header('Correlation Results')
st.subheader("Is there a relationship between Twitter sentiment and Bitcoin prices?")
st.write("There is a moderate positive correlation (0.39) between Twitter sentiment and Bitcoin prices, suggesting that more positive sentiment is associated with higher Bitcoin prices.")

# Display the lagged correlation
st.header('Lagged Correlation')
st.write("Next, let's create a lagged version of the sentiment column, to see if sentiment on a given day is correlated with the Bitcoin price on the next day.")
st.write("Again, a positive number would suggest that when sentiment is high, the Bitcoin price tends to be higher on the next day.")
st.success(f"Correlation between Bitcoin price and lagged Twitter sentiment: {correlation_lagged}")
st.write("The correlation between Bitcoin price and lagged sentiment (sentiment from the previous time step) is 0.35, which is also a moderate positive relationship.")

# Lagged Correlation results
st.header('Lagged Correlation Results')
st.subheader("Is there a relationship between Bitcoin prices and the previous time step's Twitter sentiment?")
st.write("There is a moderate positive correlation (0.35) between Bitcoin prices and the previous time step's Twitter sentiment.")

# Polynomial regression score
st.header('Polynomial Regression Score')
st.success(f"Polynomial regression score: {poly_reg}")
st.write("This score is only marginally better than the simple linear regression score, suggesting that even a more complex polynomial regression model does not do a great job of predicting Bitcoin prices based on sentiment scores alone.")

# Regression score
st.header('Regression Score')
st.write("This number is the R-squared value from a linear regression model that predicts the Bitcoin price based on the sentiment score. The R-squared value is a measure of how well the regression model fits the data; a higher R-squared means a better fit.")
st.success(f"Regression score: {reg}")
st.write("The regression scores for the polynomial regression and the regular regression are both approximately 0.15. These scores, indicate that around 15% of the variability in Bitcoin prices can be explained by the sentiment scores.")

# Regression results
st.header('Regression Score Results')
st.subheader("How much of the variation in Bitcoin prices can be explained by sentiment scores?")
st.write("About 15% of the variation in Bitcoin prices can be explained by sentiment scores, according to both polynomial and regular regression models.")

# Line plot LSTM
st.header('Line plot LSTM')
st.write(
    "Visualizing the predicted and actual values over time can give you a good sense of how well the model is performing.")
fig8 = DataVisualizer.plot_lstm_results(data.reset_index(), 'Open time', 'Close', y_train_pred, y_test_pred,
                                        'Bitcoin Price and Sentiment over Time with LSTM Prediction')
st.pyplot(fig8)
st.write("This will give you a plot where you can visually compare the actual and predicted Bitcoin prices.")

# LSTM model score
st.header('Root Mean Squared Error (RMSE)')
st.write("The RMSE measures the differences between the predicted and actual values. Specifically, it's the square root of the average of squared differences between prediction and actual observation. The smaller the RMSE value, the better the model is performing.")
st.success(f'Training RMSE: {train_rmse}, Testing RMSE: {test_rmse}')
st.write("In another words, RMSE is a measure of the average deviation of the predictions from the observed values. The RMSE for the LSTM model on the training data is approx. 90.00 ≤ and on the testing data is approx. 270.00 ≤.")

# LSTM results
st.header('LSTM Results')
st.subheader("How well does the LSTM model predict Bitcoin prices based on sentiment scores?")
st.write("The LSTM model has an RMSE of 90.00 ≤ on the training data and an RMSE of 270.00 ≤ on the testing data. This indicates a higher discrepancy between the model's predictions and the actual values on the testing data compared to the training data.")

# Conclusion
st.header('Conclusion')
st.write("From these results, it's clear that while there's some positive correlation between Twitter sentiment and Bitcoin prices, the sentiment scores we've computed are not a strong predictor of Bitcoin prices.")
st.write(
    "It's important to remember that correlation does not imply causation. Even if we found a significant correlation between Bitcoin prices and Twitter sentiment, this doesn't necessarily mean that Twitter sentiment is driving Bitcoin prices (or vice versa).")
st.write(
    "In our LSTM model analysis, we found that the model was able to predict Bitcoin prices based on sentiment scores with a varying degree of accuracy, as measured by the Root Mean Squared Error (RMSE). In repeated runs of the model, the RMSE varied due to inherent randomness in the model's training process and weight initialization. Despite this variability in the exact RMSE values, the LSTM model consistently demonstrated its capability in predicting the Bitcoin price trends based on sentiment scores, highlighting the potential value of sentiment analysis in forecasting cryptocurrency price movements.")
st.write(
    "Furthermore, the data we are working with is relatively limited. Cryptocurrency prices can be influenced by many factors, including macroeconomic indicators, regulations, technological advancements, market manipulation, and more.")
st.write(
    "It's always a good idea to explore the data further and consider integrating additional datasets to enrich the analysis, such as news data, Reddit sentiment data, transaction volume data, etc.")
st.write(
    "With these additional data points, you could potentially create a more comprehensive model to predict Bitcoin prices based on several different factors, not just Twitter sentiment.")
st.write(
    "In conclusion, this analysis has shown that there is a moderate correlation between Twitter sentiment and Bitcoin prices, with more positive sentiment associated with higher prices. However, the regression models suggest that sentiment scores can only explain about 15% of the variation in Bitcoin prices. Furthermore, the LSTM model showed a higher discrepancy between predictions and actual values in the testing data compared to the training data, indicating potential overfitting. Therefore, while Twitter sentiment does have some relationship with Bitcoin prices, it does not appear to be a strong predictor. Future research may consider incorporating additional variables or using different modeling techniques to improve predictive accuracy.")
st.write('')
st.write('')
st.markdown('**[Shayawn.com](https://shayawn.com/)**')
