import pandas as pd
import numpy as np
import pmdarima as arima
import arch

amazon_prices = pd.read_csv("amazon_stock_prices.csv")
sentiment_scores_df = pd.read_csv("sentiment_scores.csv")

# Merge the stock prices and sentiment data
prices_series = pd.Series(data=amazon_prices['adjclose'])
prices_series.index = amazon_prices['formatted_date']
merged_df = sentiment_scores_df.merge(amazon_prices, on='formatted_date', how='outer')
merged_df = merged_df.fillna(0)
merged_df = merged_df[merged_df['adjclose'] != 0.]

reshaped_sentiment_scores = np.reshape(merged_df['0'], (-1,1))

# Train/test split
y_train = prices_series[:231]
X_train = reshaped_sentiment_scores[:231]
y_test = prices_series[231:]
X_test = reshaped_sentiment_scores[231:]

# Fit the ARIMA model
arima_model_fitted = arima.auto_arima(y=y_train, exogenous=X_train)
arima_residuals = arima_model_fitted.resid()

# Fit the GARCH model
garch = arch.arch_model(arima_residuals, p=1, q=1).fit()
print(garch.summary())

predicted_mu = arima_model_fitted.predict(n_periods=3, exogenous=X_test)
garch_forecast = garch.forecast(horizon=3, start=1)
predicted_et = garch_forecast.mean['h.1'].iloc[-1]
prediction = predicted_mu + predicted_et

print(garch_forecast.mean)
print(prediction)
