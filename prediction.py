import logging
import pickle
from operator import itemgetter

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from adtk.data import validate_series
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config import COLORS_SENS, COORS_ANOMAL, COLOR_BG
from utils import mean_absolute_percentage_error

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(filename="sample.log", level=logging.INFO)

tscv = TimeSeriesSplit(n_splits=5)
scaler = StandardScaler()
DROP_CLM = ['SRN1', 'TEMPSENS2', 'TEMPSENS11', 'TEMPSENS12', 'HUMSENS10', 'HUMSENS11', 'MOTSENS10', 'MOTSENS9']


# TODO проверить пути


def create_data_for_pred(data, corr_data, weekday=True, pred_d=1, lag=7, another=True, CORR_COEFF=None, time=False):
    data_frames = dict()
    # print(data.index)
    # data.set_index('id_sp', inplace=True, drop=True)
    for d in data.columns:
        clm = [d]
        if another and d in corr_data.columns:
            clm = clm + list(corr_data[d].dropna().to_dict().keys())
            data_feature = data[clm]
        else:
            data_feature = data[clm]

        if d == 'GAZDTCT1':
            data_feature = data_feature[:484]  # print(data_feature)
        for l in range(pred_d, lag + 1):
            data_feature['lag_{}'.format(l)] = data_feature[d].shift(l)
        if weekday:
            data_feature.index = pd.to_datetime(data_feature.index)
            seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
            month_to_season = dict(zip(range(1, 13), seasons))
            data_feature['season'] = data_feature.index.month.map(month_to_season)
            data_feature['weekday'] = data_feature.index.weekday
            data_feature['is_weekend'] = data_feature.weekday.isin([5, 6]) * 1
        if CORR_COEFF:
            corr = data_feature.corr()
            corr = pd.melt(corr.reset_index(),
                           id_vars='index')  # Unpivot the dataframe, so we can get pair of arrays for x and y
            corr.columns = ['x', 'y', 'value']
            corr['value'] = corr['value'].abs()
            # фильтруем корреляцию
            corr = corr[(corr['value'] >= CORR_COEFF) & (corr['value'] != 1.0)]
            corr_data_frame = corr.pivot('y', 'x')['value']
        #             print(corr)
        if time:
            data_feature['time'] = data_feature.index
        data_feature = data_feature.dropna().reset_index(drop=True)

        data_frames[d] = data_feature
    return data_frames


def find_diff(value_first, value_second):
    return 'red' if (value_first - value_second) < 0 else 'green'


def filter_data_frame(data_frame, time):
    data_frame = data_frame[data_frame['time'] == time]
    data_frame = data_frame.drop('time', axis=1)
    return data_frame


# time format is 2018-01-14 lag format 'lag_{n}' n pred to days in forward
def prepared_block_lag_data(data_sensor: dict, lag: str, block: list, time: str, models_path=None) -> object:
    # Загружаются данные
    # Беруться данные lag
    # Берутся данные block
    # фигачится фильтрация по time
    # удаляется колонка time
    data_sensor_lag = data_sensor[lag]
    data_sensor_block = itemgetter(*block)(data_sensor_lag)
    data_sensor_filter = [filter_data_frame(data, time) for data in data_sensor_block]
    models = []
    for m in models_path:
        with open(m, 'rb') as f:
            models.append(pickle.load(f).get(lag))
    return data_sensor_filter, models


def get_prediction(data_sensor_filter, models, error=True):
    data_prediction_sensor = []
    у_true = []
    error_mean = None
    for data_sensor, model_file in zip(data_sensor_filter, models):
        y_clmn = data_sensor.columns[0]
        y = data_sensor[y_clmn]
        у_true.append(y.values[0])
        X = data_sensor.drop([y_clmn], axis=1)
        predict = model_file['model'].predict(X)
        data_prediction_sensor.append(round(predict[0], 2))
    if error:
        error_mean = mean_absolute_percentage_error(np.array(у_true), np.array(data_prediction_sensor))
        error_mean = [round(error_mean, 2).tolist()]
    return data_prediction_sensor, error_mean


def get_pred_7(data_sensor, block, time, models_path=None):
    data_prediction_sensor_7 = []
    data_prediction_sensor_7_error = []
    for i in range(1, 8):
        data_sensor_filter, models = prepared_block_lag_data(data_sensor, 'lag_{}'.format(i), block, time,
                                                             models_path=models_path)
        data_prediction_sensor, error_mean = get_prediction(data_sensor_filter, models)
        data_prediction_sensor_7.append(data_prediction_sensor)
        if error_mean:
            data_prediction_sensor_7_error.append(error_mean)
    convert_data = list(zip(*data_prediction_sensor_7))
    if error_mean:
        covert_mean = list(zip(*data_prediction_sensor_7_error))
        covert_mean = list(covert_mean[0])

    return convert_data, covert_mean


# TODO добавить по датам отображение
def create_scatter_plot(convert_data, sensor_mean, block, time=None):
    traces = [go.Scatter(x=list(range(1, 8)), y=d,
                         mode='lines+markers',
                         name=block[ind],
                         marker_color=COLORS_SENS[ind]) for ind, d in enumerate(convert_data)]
    average = round(sum(sensor_mean) / len(sensor_mean), 4)
    fig = {
        # set data equal to traces
        'data': traces,
        # use string formatting to include all symbols in the chart title
        'layout': go.Layout(title='forecast on 7 days <br>mean error {}'.format(average),
                            autosize=True,
                            plot_bgcolor="#F7F7F7",
                            paper_bgcolor='#F7F7F7',
                            yaxis={'title': 'Time', 'autorange': True})
    }

    return fig


# TODO придумать что делать с константой газа
def anomalies_find(data_pivot_anom, block, func_anomal):
    anomal_data = dict()
    data_d = data_pivot_anom[block]
    data_d.index.names = ['Time']
    data_d.index = pd.to_datetime(data_d.index)
    df = validate_series(pd.DataFrame(data_d))

    anomalies = func_anomal.fit_detect(df)
    # TODO придумать что делать с константой газа
    anomal_points = data_d[anomalies.values] if block == 'GAZDTCT1' else data_d[anomalies[block]]
    anomal_data[block] = {'anom': anomal_points, 'data': data_d}
    return anomal_data


def create_plot(anomal_data):
    plot_anomal = []
    for d in anomal_data:
        data_d = anomal_data[d]
        traces = [go.Scatter(x=data_d['data'].index, y=data_d['data'].values,
                             mode='lines',
                             marker_color=COORS_ANOMAL[0],
                             name=d),
                  go.Scatter(x=data_d['anom'].index,
                             y=data_d['anom'].values,
                             marker_color=COORS_ANOMAL[1],
                             mode='markers',
                             name='value anomaly')]
        fig = {
            # set data equal to traces
            'data': traces,
            # use string formatting to include all symbols in the chart title
            'layout': go.Layout(title='{} anomaly'.format(d),
                                autosize=True,
                                plot_bgcolor=COLOR_BG,
                                paper_bgcolor=COLOR_BG,
                                yaxis={'autorange': True},
                                xaxis={'title': 'Time'})
        }
        plot_anomal.append(fig)
    return plot_anomal
