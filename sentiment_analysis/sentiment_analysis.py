import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn.functional import softmax

# Load the pre-trained RoBERTA tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base")
model.eval()

# Function to get sentiment score
def assign_sentiment(text):

    if isinstance(text, str):
        # Tokenize the input text and truncate if necessary
        inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=512)

        # Forward pass through model
        with torch.no_grad():
            outputs = model(**inputs)

        # Get logits and apply softmax to get probs
        logits = outputs.logits
        probabilities = softmax(logits, dim=1)

        # Extract prob for positive (label 1)
        sentiment_score = probabilities[:, 1].item()

        return sentiment_score


# Main script
df = pd.read_csv('date_formatted_reddit_posts.csv')

df_subset = df.head(10000)

df_subset['Sentiment Score'] = df_subset['Body'].apply(assign_sentiment)

min_score = df_subset['Sentiment Score'].min()
max_score = df_subset['Sentiment Score'].max()

df_subset['Normalized Sentiment'] = (df_subset['Sentiment Score'] - min_score) / (max_score - min_score)

df_subset.to_csv('sentiment_scores_reddit.csv', index=False)
