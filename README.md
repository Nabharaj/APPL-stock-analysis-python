#Apple Stock Analysis with Python
This project aims to analyze Apple's stock performance using Python libraries such as yfinance, pandas, Autoviz, and Tableau for visualization. The analysis focuses on exploring historical stock price data, identifying trends, and visualizing insights that could be useful for traders or investors.

Project Workflow
1. Data Collection
We started by fetching historical stock data for Apple (AAPL) from Yahoo Finance using the yfinance library. This provided us with essential data like closing prices, volume, and stock splits.

2. Exploratory Data Analysis (EDA)
To understand the data better, we performed Exploratory Data Analysis (EDA) using Autoviz to generate visual insights. While we primarily used Autoviz in this project, we are also familiar with other alternative tools such as Lux and Yellowbrick, which provide additional visual recommendations and model performance insights.

3. Initial Stock Price Prediction (LSTM Model)
In the initial phase, we built an LSTM model (Long Short-Term Memory) using only historical closing prices to predict Apple’s stock prices. This provided us with baseline predictions but highlighted that stock price movements are influenced by more than just historical prices, resulting in moderate accuracy.

4. Incorporating Sentiment Analysis
To improve prediction accuracy, we integrated sentiment analysis using VADER and the News API. We fetched news articles related to Apple, analyzed the sentiment of each article, and added this as a feature in the LSTM model. The results showed a significant improvement in prediction accuracy, highlighting the importance of considering external news events on stock price movements.

5. Future Improvements
Although incorporating sentiment analysis enhanced our model's performance, there’s more to explore. In future iterations, we plan to incorporate Bloomberg Terminal data, which includes comprehensive financial documentation, analyst ratings, and real-time data to further refine our predictions and make them more reliable for trading and investment purposes.


We chose LSTM because it excels at capturing long-term dependencies and temporal patterns in sequential data, making it ideal for stock price prediction. Its ability to handle complex non-linear relationships and integrate multiple input features offers an advantage over traditional models.
