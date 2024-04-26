import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Load data
df = pd.read_csv('merged_sentiment_and_price.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Normalize the data
scaler = MinMaxScaler()
df[['Sentiment Score', 'close']] = scaler.fit_transform(df[['Sentiment Score', 'close']])


# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        curr_target = data[i + seq_length]

        sequences.append(seq)
        targets.append(curr_target)

    # convert to numpy arrays
    sequences = np.array(sequences)
    targets = np.array(targets)

    return sequences, targets

# main script

# convert df to numpy array
data = df[['Sentiment Score', 'close']].values

seq_length = 2
test_size = 0.2

# Create a sequence and split data
x, y = create_sequences(data, seq_length)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False)

# Tranform data to tensors
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)

# Define LSTM
input_size = 2
hidden_size = 50
output_size = 1

lstm = nn.LSTM(input_size, hidden_size)
fc = nn.Linear(hidden_size, output_size)

# funcs for eval
criterion = nn.MSELoss()
optimizer = optim.Adam(list(lstm.parameters()) + list(fc.parameters()), lr=0.001)

# Training
epochs = 50
for epoch in range(epochs):
    lstm.train()
    optimizer.zero_grad()
    lstm_out, _ = lstm(x_train.view(len(x_train), 1, -1))
    output = fc(lstm_out.view(len(x_train), -1))
    loss = criterion(output.view(-1), y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluation
lstm.eval()
with torch.no_grad():
    lstm_out, _ = lstm(x_test.view(len(x_test), 1, -1))
    y_pred = fc(lstm_out.view(len(x_test), -1))
    test_loss = criterion(y_pred.view(-1), y_test)

print(f'Test Loss: {test_loss.item():.4f}')


