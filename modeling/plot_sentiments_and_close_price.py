# This script plots the close price and the # of reddit posts and/or scores
import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

df = read_csv('D:/reddit/RS_2022_amazon_emotion_pivoted.csv', header=0)
values = df.values

# Metrics to plot (e.g., #  of posts and score)
groups = [9, 10, 11, 12, 13, 14, 15, 16] # 2, 3, 4, 5, 6, 7, 8,
i = 1

for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(df.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()