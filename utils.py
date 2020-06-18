import matplotlib.pyplot as plt  # plots
import numpy as np  # vectors and matrices
import pandas as pd  # tables and data manipulations
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler



import statsmodels.tsa.api as smt
import statsmodels.api as sm


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def sensor_data_mean(clm, data_pivot_date=[], fillna='', DROP_CLM=[]):
    data_pivot = data_pivot_date.groupby(clm).mean()
    data_pivot = data_pivot.drop(DROP_CLM, axis=1)
    data_temp = dict()
    for i in data_pivot.columns:
        data_temp[i] = pd.Series(data_pivot[i].dropna().to_list())
    if fillna:
        corr_data_frame = pd.DataFrame(data_temp).fillna(method=fillna)
        data_pivot = data_pivot.fillna(method=fillna)
    return corr_data_frame, data_pivot


def timeseries_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """

    # get the index after which test set starts
    test_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:test_index - 1]
    y_train = y.iloc[:test_index - 1]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]

    return X_train, X_test, y_train, y_test


def plotModelResults(model, y_train, y_test, tscv, X_train=None, X_test=None, plot_intervals=False,
                     plot_anomalies=False, title=None):
    """
        Plots modelled vs fact values, prediction intervals and anomalies

    """

    prediction = model.predict(X_test)

    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)

    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train,
                             cv=tscv,
                             scoring="neg_mean_absolute_error")
        mae = cv.mean() * (-1)
        deviation = cv.std()

        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)

        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)

        if plot_anomalies:
            anomalies = np.array([np.NaN] * len(y_test))
            anomalies[y_test < lower] = y_test[y_test < lower]
            anomalies[y_test > upper] = y_test[y_test > upper]
            plt.plot(anomalies, "o", markersize=10, label="Anomalies")

    error = mean_absolute_percentage_error(prediction, y_test)
    # print("Mean absolute percentage error {} {}".format(round(error, 2), title))
    plt.title("Mean absolute percentage error {} {}".format(round(error, 2), title))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);


def plotCoefficients(model, X_train):
    """
        Plots sorted coefficient values of the model
    """

    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');


def get_predict_all(prepared_data, device_name_dict, tscv, scale=True, plot=True):
    lst_models = []
    lr = LinearRegression()
    scaler = StandardScaler()
    scaler_1 = StandardScaler()
    for data_p in prepared_data:
        data_for_model = data_p
        y_clmn = data_for_model.columns[0]
        name_clm = device_name_dict[y_clmn]
        y = data_for_model[y_clmn]
        X = data_for_model.drop([y_clmn], axis=1)
        # reserve 30% of data for testing
        X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)
        print(X_train.columns)
        colmn = X_test.columns
        X_train_interval = X_train

        if scale:
            scaler_1.fit(X_train)
            scaler_1.transform(X_train)
            X_train = scaler.fit_transform(X_train)

            X_test = scaler.transform(X_test)
        # machine learning in two lines
        lr.fit(X_train, y_train)
        prediction = lr.predict(X_test)

        error = mean_absolute_percentage_error(prediction, y_test)
        print("Mean absolute percentage error {} {}".format(round(error, 2), y_clmn))
        lst_models.append(
            {'model': lr, 'error': round(error, 2), 'name': y_clmn, 'full_name': name_clm, 'X_test': X_test,
             'columns': colmn, 'scaler': scaler_1})
        # %%
    if plot:
        plotModelResults(lr, y_train, y_test, tscv, X_train=X_train, X_test=X_test, plot_intervals=True,
                         title=name_clm)
        plotCoefficients(lr, X_train_interval)
    return lst_models


def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test

        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        print(sm.tsa.stattools.adfuller(y)[4])
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
