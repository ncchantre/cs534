import pandas as pd

df = pd.read_csv('RS_2022_Amazon_with_subreddits_no_stopwords.csv')

# Format that the date is currently in from reddit data
df['Timestamp'] = pd.to_datetime(df['Timestamp'], infer_datetime_format=True, errors='coerce')

# Extract the day and the month from the timestamp column
df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month

# Filter out rows of year 2021
df = df[df['Timestamp'].dt.year == 2022]

# New date column with properly formatted date
df['Date'] = df['Timestamp'].dt.strftime('%m/%d')

# Drop the original Timestamp column
df.drop(columns=['Timestamp'], inplace=True)

# Save the df to a new CSV to be used in pipeline
df.to_csv('date_formatted_reddit_posts.csv', index=False)

