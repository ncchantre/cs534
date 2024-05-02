# This script is adapted from this tutorial: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# Create LSTM model with reddit post count and sum of scores
import warnings
warnings.filterwarnings('ignore')
from math import sqrt
import matplotlib.pyplot as plt
from numpy import concatenate, expand_dims
from pandas import DataFrame, concat, read_csv
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

def main():
    source_file = 'D:/reddit/RS_2022_amazon_emotion_pivoted.csv'
    destination_file = 'D:/reddit/lstm_all_subreddits.csv'

    # Function to convert series to supervised learning
    # This function allows for lag periods, which we opted not to use
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        agg = concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # load dataset
    dataset = read_csv(source_file, header=0, index_col=0)
    dataset = dataset[['post_count_anger', 'post_count_disgust', 'post_count_fear',\
                        'post_count_joy', 'post_count_neutral', 'post_count_sadness',\
                        'post_count_surprise', 'score_anger', 'score_disgust',\
                        'score_fear', 'score_joy', 'score_neutral', 'score_sadness',\
                        'score_surprise', 'close']]
    values = dataset.values
    values = values.astype('float32')

    # Split into train and test sets
    n_train_hours = 189
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    # Normalize features for training set
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(train)
    reframed = series_to_supervised(scaled, 1, 1)
    reframed.drop(reframed.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    train = reframed.values

    # Normalize features for training set
    scaled = scaler.fit_transform(test)
    reframed = series_to_supervised(scaled, 1, 1)
    reframed.drop(reframed.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    test = reframed.values

    # Split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # Reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # Design LSTM network
    model = Sequential()
    model.add(LSTM(5, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # Fit LSTM
    history = model.fit(train_X, train_y, epochs=200, batch_size=10, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    # Use the network to predict stock price
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # Invert scaling for prediction
    inv_yhat = concatenate((yhat, test_X[:, -14:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # Invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -14:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    # Calculate metrics
    mae = mean_absolute_error(inv_y, inv_yhat)
    mse = mean_squared_error(inv_y, inv_yhat)
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    avg = sum(inv_y)/len(inv_y)
    nmae = mae/avg
    nmse = mse/avg
    nrmse = rmse/avg
    results = DataFrame({'actual': inv_y, 'prediction': inv_yhat})
    results.to_csv(destination_file)
    print('MAE: %.3f' % mae)
    print('MSE: %.3f' % mse)
    print('RMSE: %.3f' % rmse)
    print('Normalized MAE: %.3f' % nmae)
    print('Normalized MSE: %.3f' % nmse)
    print('Normalized RMSE: %.3f' % nrmse)
    # plot lines
    x_axis = list(range(0, 61))
    plt.plot(x_axis, inv_y, label = 'Actual', linestyle="-")
    plt.plot(x_axis, inv_yhat, label = 'Predicted', linestyle="-")
    plt.yticks([])
    plt.title('Actual vs. Predicted Stock Price for Amazon between 10/6/2022 - 12/30/2022')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()