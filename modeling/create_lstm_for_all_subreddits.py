# This script is adapted from this tutorial:
# Create LSTM model with reddit post count and sum of scores
from math import sqrt
from numpy import concatenate, expand_dims
from pandas import DataFrame, concat, read_csv
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM


#  Function to convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# load dataset
dataset = read_csv('D:/reddit/RS_2022_amazon_emotion_pivoted.csv', header=0, index_col=0)
dataset = dataset[['post_count_anger', 'post_count_disgust', 'post_count_fear',\
                    'post_count_joy', 'post_count_neutral', 'post_count_sadness',\
                    'post_count_surprise', 'score_anger', 'score_disgust',\
                    'score_fear', 'score_joy', 'score_neutral', 'score_sadness',\
                    'score_surprise', 'close']]
values = dataset.values
# Convert to float data type
values = values.astype('float32')

# split into train and test sets
n_train_hours = 189
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# normalize features for training set
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(train)
reframed = series_to_supervised(scaled, 1, 1)
train = reframed.values

# normalize features for testing set
scaled = scaler.fit_transform(test)
reframed = series_to_supervised(scaled, 1, 1)
test = reframed.values
print(test)

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=300, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -14:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -14:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
results = DataFrame({'actual': inv_y, 'prediction': inv_yhat})
results.to_csv('D:/reddit/z.csv')
print('Test RMSE: %.3f' % rmse)