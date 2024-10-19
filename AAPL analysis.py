#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# <div style="text -align:center;">
#     <font size="13" color="blue"> Apple Stock Analysis</font>
#     </div>

# <div style="text -align:center;">
#     <font size="3.25" color="red"> 1. Fetch Apple Stock data using yfinance</font>
#     </div>

# In[1]:


import yfinance as yf

#Fetch apple stock data for last 5 years.
apple_data=yf.download("AAPL",start="2019-01-01", end="2024-01-01")

#Checking first few rows from data
print(apple_data.head())


# In[ ]:


#save the data to csv for future use.
#apple_data.to_csv("apple_stock_data.csv")


# <div style="text -align:center;">
#     <font size="3.25" color="red"> 2.Perform EDA with AutoViz</font>
#     </div>
# 

# In[2]:


import pandas as pd

from autoviz.AutoViz_Class import AutoViz_Class



# Load  dataset (Apple stock data)
apple_data = pd.read_csv("apple_stock_data.csv")

# Convert 'Date' column to datetime if necessary
if 'Date' in apple_data.columns:
    apple_data['Date'] = pd.to_datetime(apple_data['Date'])

# Ensure the DataFrame is a pure Pandas DataFrame (to avoid any Lux interference)
apple_data = pd.DataFrame(apple_data)

# Initialize AutoViz
AV = AutoViz_Class()

# Run AutoViz on the DataFrame
df_autoviz = AV.AutoViz(filename="", dfte=apple_data)




# In[5]:


# This line ensures that plots are displayed inline in Jupyter Notebook
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import mplfinance as mpf

# Load the dataset (Apple stock data)
apple_data = pd.read_csv("apple_stock_data.csv")

# Convert the 'Date' column to datetime
apple_data['Date'] = pd.to_datetime(apple_data['Date'])

# Set 'Date' as the index for the candlestick chart
apple_data.set_index('Date', inplace=True)

# Create the candlestick chart
mpf.plot(apple_data, type='candle', volume=True, style='charles', title='Apple Stock Price',
         ylabel='Price (USD)', ylabel_lower='Volume')

# Explicitly show the plot (optional)
mpf.show()


# <div style="text -align:center;">
#     <font size="3.25" color="red"> What does the data tells us?
#     </font>
#     </div>
# 
# 

# The Apple stock prices (Open, High, Low, Close, and Adjusted Close) are highly correlated with each other, meaning they generally move in the same direction. 
# 
# This is expected in stock market data because prices during the day tend to move together.
# 
# Volume has outliers, which could indicate high trading activity on specific days—perhaps due to important news or events that impacted Apple.
# 
# Date should be transformed if you plan to build machine learning models to predict future prices, as models may benefit from understanding time-based patterns.
# 

# In[ ]:


import pandas as pd
from ydata_profiling import ProfileReport

# Load  dataset (Apple stock data)
apple_data = pd.read_csv("apple_stock_data.csv")

# Generate a profile report
profile = ProfileReport(apple_data, title="Apple Stock Data Profiling Report", explorative=True)

# Save the report to an HTML file
profile.to_file("apple_stock_profile_report.html")

# Display the report in a Jupyter notebook 
profile.to_notebook_iframe()



# In[6]:


import pandas as pd

# Load  dataset (Apple stock data)
apple_data = pd.read_csv("apple_stock_data.csv")

# Convert the 'Date' column to datetime
apple_data['Date'] = pd.to_datetime(apple_data['Date'])

# Calculate the percentage change in 'Close' price between consecutive days
apple_data['Pct_Change'] = apple_data['Close'].pct_change() * 100

# View the first few rows to confirm
print(apple_data[['Date', 'Close', 'Pct_Change']].head())


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt

# Load dataset (Apple stock data)
apple_data = pd.read_csv("apple_stock_data.csv")

# Convert the 'Date' column to datetime
apple_data['Date'] = pd.to_datetime(apple_data['Date'])

# Calculate the percentage change in 'Close' price between consecutive days
apple_data['Pct_Change'] = apple_data['Close'].pct_change() * 100

# Plot percentage changes to observe sharp movements
plt.figure(figsize=(12, 6))
plt.plot(apple_data['Date'], apple_data['Pct_Change'], label='Daily % Change', color='blue')
plt.title('Percentage Change in Apple Stock Price (Daily)')
plt.xlabel('Date')
plt.ylabel('% Change')
plt.grid(True)
plt.show()

# Calculate the mode for all percentage changes (most frequent value)
mode_pct_change = apple_data['Pct_Change'].mode()

# Identify the maximum and minimum percentage change (sharpest changes)
max_pct_change = apple_data['Pct_Change'].max()
min_pct_change = apple_data['Pct_Change'].min()

# Display results
print(f"Mode of percentage changes: {mode_pct_change}")
print(f"Maximum (sharpest increase) percentage change: {max_pct_change}%")
print(f"Minimum (sharpest decrease) percentage change: {min_pct_change}%")


# In[8]:


import pandas as pd

# Load your dataset (Apple stock data)
apple_data = pd.read_csv("apple_stock_data.csv")

# Convert the 'Date' column to datetime
apple_data['Date'] = pd.to_datetime(apple_data['Date'])

# Calculate the percentage change in 'Close' price between consecutive days
apple_data['Pct_Change'] = apple_data['Close'].pct_change() * 100

# Set the thresholds for sharp increase and sharp decrease
sharp_increase_threshold = 5  # 5% increase
sharp_decrease_threshold = -5  # -5% decrease

# Identify dates with sharp increases (greater than 5%)
sharp_increase = apple_data[apple_data['Pct_Change'] > sharp_increase_threshold]

# Identify dates with sharp decreases (less than -5%)
sharp_decrease = apple_data[apple_data['Pct_Change'] < sharp_decrease_threshold]

# Display the dates and stock prices for sharp increases
print("Dates with sharp increases (greater than 5%):")
print(sharp_increase[['Date', 'Close', 'Pct_Change']])

# Display the dates and stock prices for sharp decreases
print("\nDates with sharp decreases (less than -5%):")
print(sharp_decrease[['Date', 'Close', 'Pct_Change']])


# In[13]:


import yfinance as yf
import pandas as pd
from newsapi import NewsApiClient
import matplotlib.pyplot as plt

# Initialize the News API client (use  API key from https://newsapi.org/)
newsapi = NewsApiClient(api_key='be43932358024211ad050e5053e4b34e')

# Fetch Apple stock data for the last 30 days
apple_data = yf.download("AAPL", period="1mo")

# Calculate the percentage change in 'Close' price between consecutive days
apple_data['Pct_Change'] = apple_data['Close'].pct_change() * 100

# Define threshold for significant price movements 
increase_threshold = 2  # 2% increase
decrease_threshold = -2  # 2% decrease

# Identify dates with significant increases and decreases
significant_changes = apple_data[(apple_data['Pct_Change'] >= increase_threshold) | 
                                 (apple_data['Pct_Change'] <= decrease_threshold)]

# Plot percentage changes
plt.figure(figsize=(10, 8))
plt.plot(apple_data.index, apple_data['Pct_Change'], label='Daily % Change', color='blue')
plt.axhline(y=increase_threshold, color='green', linestyle='--', label='Increase Threshold (2%)')
plt.axhline(y=decrease_threshold, color='red', linestyle='--', label='Decrease Threshold (-2%)')
plt.title('Apple Stock: Percentage Change Over the Last 30 Days')
plt.xlabel('Date')
plt.ylabel('% Change')
plt.legend()
plt.grid(True)
plt.show()

# Function to fetch news for a given date
def fetch_news_for_date(date):
    # Convert date to string in YYYY-MM-DD format
    date_str = date.strftime('%Y-%m-%d')
    
    # Fetch news articles related to Apple on the given date
    articles = newsapi.get_everything(q='Apple',
                                      from_param=date_str,
                                      to=date_str,
                                      language='en',
                                      sort_by='relevancy')
    return articles['articles']

# Loop through each significant date and fetch related news
for index, row in significant_changes.iterrows():
    print(f"\nDate: {index.date()}, Percentage Change: {row['Pct_Change']:.2f}%")
    
    # Fetch news articles for the date
    news_articles = fetch_news_for_date(index)
    
    # Display the news headlines and URLs
    if news_articles:
        for article in news_articles:
            print(f"Title: {article['title']}")
            print(f"URL: {article['url']}\n")
    else:
        print("No news articles found for this date.")


# In[ ]:




