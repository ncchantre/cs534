import pandas as pd

# cleaning data with bad dates from scored csv

df_sentiment = pd.read_csv('sentiment_scores_reddit.csv')

df_sentiment['Date'] = pd.to_datetime(df_sentiment['Date'], errors='coerce')

df_sentiment.dropna(subset=['Date'], inplace=True)

df_sentiment['Date'] = df_sentiment['Date'].dt.strftime('%m/%d/%Y')

# Beginning of merge logic

df_amazon = pd.read_csv('amazon_stock_prices.csv')

df_amazon['formatted_date'] = pd.to_datetime(df_amazon['formatted_date']).dt.strftime('%m/%d/%Y')

df_merged = pd.merge(df_sentiment, df_amazon, left_on='Date', right_on='formatted_date', how='left')

df_merged.to_csv('merged_sentiment_and_price.csv', index=False)
