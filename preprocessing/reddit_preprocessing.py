import pandas as pd
import re

# Stop words to filter out
stop_words = ['a', 'an', 'and', 'about', 'are', 'as', 'at', 'be', 'but', 'by',
              'for', 'if', 'from', 'Hi', 'hi', 'Hey', 'hey', 'has', 'I', 'Im', 'i', 'im'
                                                                                    'in', 'into', 'is', 'it', 'no',
              'not', 'of', 'on', 'or', 'such',
              'that', 'the', 'their', 'then', 'there', 'these', 'they', 'this', 'to',
              'was', 'will', 'with', 'you', 'youve', 'youre', 'your']


def extract_subreddit(link):
    return re.search('r//*[A-Za-z\_0-9]+', link).group(0)


def filter_stop_words(sentence):
    words = sentence.split()
    filtered = [word for word in words if word not in stop_words and len(word) <= 20]
    return " ".join(filtered)


# Read in 2022 Amazon data
df = pd.read_csv('RS_2022_amazon.csv',
                 names=['ID', 'User', 'Title', 'Score', 'Timestamp', 'Comments Link', 'Body', 'Comments Link 2'])
# Drop redundant column
df.drop('Comments Link 2', 'columns', inplace=True)

# Filter for score, extract subreddit, drop unnecessary column
df_filtered = df[df['Score'] >= 6.0]
df_filtered['Subreddit'] = df_filtered['Comments Link'].apply(lambda x: extract_subreddit(x))
df_filtered.drop('Comments Link', 'columns', inplace=True)

# Filter out numbers and other non-alphabetic characters
df_filtered['Body'] = df_filtered['Body'].str.replace('[^a-zA-Z\s]+', '')
df_filtered['Body'] = df_filtered['Body'].str.replace('\s+', ' ')

# Filter out stopwords
df_filtered['Body'] = df_filtered['Body'].apply(filter_stop_words)

df_filtered.to_csv('RS_2022_Amazon_with_subreddits_no_stopwords.csv')
