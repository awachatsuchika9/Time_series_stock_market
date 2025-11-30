import itertools
import numpy as np
import statsmodels.api as sm
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==============================
# ARIMA (with grid search)
# ==============================
def fit_arima(train_series, p_range=(0,3), d_range=(0,2), q_range=(0,3)):
    p = range(*p_range)
    d = range(*d_range)
    q = range(*q_range)
    pdq = list(itertools.product(p, d, q))

    best_aic = float("inf")
    best_order = None
    best_res = None

    for order in pdq:
        try:
            model = sm.tsa.ARIMA(train_series, order=order)
            res = model.fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_order = order
                best_res = res
        except:
            continue

    return best_order, best_res

def forecast_arima(fitted_model, steps):
    pred = fitted_model.get_forecast(steps=steps)
    pred_mean = pred.predicted_mean
    pred_ci = pred.conf_int()
    return pred_mean, pred_ci

# ==============================
# Prophet
# ==============================
def fit_prophet(train_df):
    m = Prophet(daily_seasonality=True, yearly_seasonality=True)
    m.fit(train_df)
    return m

def forecast_prophet(model, periods, freq="B"):
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

# ==============================
# LSTM
# ==============================
def create_sequences(X, lookback=60):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i, 0])
        ys.append(X[i, 0])
    return np.array(Xs), np.array(ys)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_lstm_data(series, lookback=60, split_ratio=0.8):
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1,1))

    Xs, ys = create_sequences(series_scaled, lookback)
    train_size = int(len(Xs)*split_ratio)
    X_train, y_train = Xs[:train_size], ys[:train_size]
    X_test, y_test   = Xs[train_size:], ys[train_size:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test, scaler
