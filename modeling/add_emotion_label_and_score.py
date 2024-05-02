# This script cleans up the Reddit post body (e.g., removing URLs, digits, stopwords)
# Then uses the j-hartmann/emotion-english-distilroberta-base model (https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
# to create an emotion score and label. Then saves the result to CSV
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')
from transformers import pipeline
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import plotly.express as px
import random
import re
import torch
pd.options.display.width = 0

def preprocessing(df):
    # Determine subreddit from URL
    df[['idx', 'subreddit', 'ignore']] = df['url'].str.split('https://www.reddit.com/r/(.*)/comments/', expand=True)
    df_body = df[['body']]

    # Convert to lowercase
    df_body = df_body.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    # Regex pattern for URLs
    url_pattern = re.compile(r'https?://\S+')

    # Function to remove URLs from body
    def remove_urls(text):
        return url_pattern.sub('', text)

    # Apply the function to body
    df_body['body'] = df_body['body'].apply(remove_urls)

    # Remove nonwords
    df_body = df_body.replace(to_replace=r'[^\w\s]', value='', regex=True)

    # Remove digits
    df_body = df_body.replace(to_replace=r'\d', value='', regex=True)

    # Remove stopwords
    stop = stopwords.words('english')
    df_body['body'] = df_body['body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return df_body

def assign_sentiment(df, df_body, destination_file):
    # Download pre-trained emotion classification model, assign sentiment, and append to CSV
    model_path = 'j-hartmann/emotion-english-distilroberta-base'
    model = pipeline('text-classification',
                     model='j-hartmann/emotion-english-distilroberta-base',
                     tokenizer=model_path,
                     max_length=512,
                     truncation=True)

    # Determine the emotion of each Reddit post using the model and append to the CSV
    all_texts = df_body['body'].values.tolist()
    all_emotions = model(all_texts)
    df_body['emotion_label'] = [d['label'] for d in all_emotions]
    df_body['emotion_score'] = [d['score'] for d in all_emotions]
    print([df_body[['body', 'emotion_label', 'emotion_score']]])
    df = pd.concat([df[['id', 'subreddit', 'date', 'score']], df_body[['emotion_label', 'emotion_score']]], axis=1)
    df.to_csv(destination_file, mode='a', header=False)
    return True

def main():
    # Create empty CSV with headers
    header_df = pd.read_csv('D:/reddit/demo.csv', nrows=1)
    header_df.iloc[:0].to_csv('D:/reddit/demo_emotion.csv', index=False)

    source_file = 'D:/reddit/demo.csv'
    destination_file = 'D:/reddit/demo_emotion.csv'
    chunksize = 100
    for chunk in pd.read_csv(source_file,
                             names=['id', 'user', 'title', 'score', 'date', 'url', 'body', 'link'],
                             chunksize=chunksize):
        df = chunk
        df_body = preprocessing(df)
        assign_sentiment(df, df_body, destination_file)

if __name__ == '__main__':
    main()