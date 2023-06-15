# CryptoSentiment

*An analysis of the relationship between Bitcoin prices and Twitter sentiment.*

![Architecture](Bitcoin%20Sentiment%20Analysis.jpg)

## Table of Contents

1. [Introduction](#introduction)
2. [Setup Instructions](#setup-instructions)
3. [Project Structure](#project-structure)
4. [Dataset](#dataset)
5. [Results](#results)
6. [License](#license)

## Introduction <a name="introduction"></a>

Bitcoin, the first and largest cryptocurrency, has gained tremendous popularity and attention from investors, enthusiasts, and researchers over the past decade. Its volatile price movements have intrigued people from various fields, fueling attempts to predict its future trends. Among the several factors that may influence Bitcoin's price, the sentiment in the cryptocurrency community, particularly as expressed on social media platforms, has been of particular interest. This study explores the relationship between Bitcoin prices and sentiment scores derived from Twitter data, utilizing a range of statistical techniques and machine learning models, including regression analysis and Long Short-Term Memory (LSTM) networks.

The objective of this research is to gain insights into the potential predictive power of sentiment scores on Bitcoin price movements. We examine the descriptive statistics of the sentiment scores and Bitcoin prices, as well as their correlation. Subsequently, we investigate the applicability and efficacy of LSTM, a type of Recurrent Neural Network (RNN) known for its ability to handle sequential data, in forecasting Bitcoin prices based on sentiment scores. Our findings offer valuable perspectives on the interplay between social sentiment and Bitcoin prices, and the utility of machine learning in cryptocurrency price prediction.

## Setup Instructions <a name="setup-instructions"></a>

This project requires Python and pip for installing dependencies. It is recommended to use a virtual environment to keep the dependencies required by different projects in separate places.

### Prerequisites

- Python 3.8 or later. You can download it from [here](https://www.python.org/downloads/).
- pip. It is already installed if you have Python 2 >=2.7.9 or Python 3 >=3.4 downloaded from python.org. If not, you can download it from [here](https://pip.pypa.io/en/stable/installation/).

### Installing

1. Clone this repository to your local machine.
    ```bash
    git clone https://github.com/Shayawnn/CryptoSentiment.git
    ```
2. Change into the project directory.
    ```bash
    cd CryptoSentiment
    ```
3. (Recommended) Setup a virtual environment. If you're using python3, it comes with the built-in `venv` module.
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```
4. Install the required dependencies from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
5. If you want to run the main analysis script, execute the `main.py` file.
    ```bash
    python main.py
    ```
    If you want to run the Streamlit app, execute the `app.py` file with Streamlit.
    ```bash
    streamlit run app.py
    ```

You should now be up and running with the CryptoSentiment project!

## Project Structure <a name="project-structure"></a>

- `app.py`: This is the Streamlit app for presenting the results of the analysis and plots.
- `main.py`: This is the main script used for data merging, reshaping, running the analysis, and training the LSTM model.
- `visualize.py`: This script contains all the plotting and data visualization functions.
- `database.py`: This script is used for loading and cleaning the Tweet dataset and fetching and storing Bitcoin data from the Binance API.

## Dataset <a name="dataset"></a>

This project utilizes two key data sources:

1. **Bitcoin daily price data**: This data is fetched directly from the Binance API and stored in the project's database. The data includes the daily open, high, low, and closing prices.

2. **Bitcoin-related tweets**: This data is sourced from a [Kaggle dataset](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets). The original dataset has been cleaned and processed to better suit the needs of this project. The cleaned dataset is included in the `./datasets` directory of this project.

Before running the scripts, make sure the cleaned tweet dataset is in the `./datasets` directory. The Binance data is fetched directly when running the scripts, so no prior setup is needed for that.

## Results <a name="results"></a>

In conclusion, this analysis has shown that there is a moderate correlation between Twitter sentiment and Bitcoin prices, with more positive sentiment associated with higher prices. However, the regression models suggest that sentiment scores can only explain about 15% of the variation in Bitcoin prices. Furthermore, the LSTM model showed a higher discrepancy between predictions and actual values in the testing data compared to the training data, indicating potential overfitting. Therefore, while Twitter sentiment does have some relationship with Bitcoin prices, it does not appear to be a strong predictor. Future research may consider incorporating additional variables or using different modeling techniques to improve predictive accuracy.

## License <a name="license"></a>

This project is licensed under the terms of the [GNU General Public License v3.0](LICENSE).

