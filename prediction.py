import logging
import pickle
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from utils import sensor_data_mean, mean_absolute_percentage_error

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(filename="sample.log", level=logging.INFO)

tscv = TimeSeriesSplit(n_splits=5)
scaler = StandardScaler()
DROP_CLM = ['SRN1', 'TEMPSENS2', 'TEMPSENS11', 'TEMPSENS12', 'HUMSENS10', 'HUMSENS11', 'MOTSENS10', 'MOTSENS9']


# TODO проверить пути
def load_data():
    main_path = r'D:\Git_project\Jupyter_Projects\Smart_house\data\device_log_join_device_type.csv'
    data = pd.read_csv(main_path, encoding="cp1251", sep=',')
    logging.info("load main data size - {}".format(data.shape[0]))
    # for time-series cross-validation set 5 folds

    # Словарь для того чтобы смотреть какая комната
    device_name_dict = data[['device_name', 'desc']].drop_duplicates().set_index(keys='device_name').to_dict()['desc']
    device_name_dict = {i: '{}-{}'.format(i, j) for i, j in
                        zip(list(device_name_dict.keys()), list(device_name_dict.values()))}

    data['id_sp'] = data['id'].apply(lambda x: x.split(' ')[0])
    clm = ['un', 'id', 'device_name', 'value']
    data_pivot = data[clm].pivot('un', 'device_name', 'value')
    data_pivot_date = data_pivot.join(data[['un', 'id', 'id_sp']].set_index('un'))

    data_corr, data_pivot = sensor_data_mean('id_sp', data_pivot_date=data_pivot_date, fillna='bfill',
                                             DROP_CLM=DROP_CLM)
    logging.info("data regroup")

    corr = data_corr.corr()
    corr = pd.melt(corr.reset_index(),
                   id_vars='index')  # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']
    corr['value'] = corr['value'].abs()
    # фильтруем корреляцию
    CORR_COEFF = 0.7
    corr = corr[(corr['value'] >= CORR_COEFF) & (corr['value'] != 1.0)]

    corr_data_frame = corr.pivot('y', 'x')['value']
    logging.info("correlation found coef- {}".format(CORR_COEFF))
    # Трансформирование
    for i in data_pivot.columns:
        if 'MOT' in i:
            data_pivot[i] = data_pivot[i].apply(lambda x: 0 if x <= 0 else 1)
    data_pivot.to_csv(r'sensor_data_vis.csv', index=True, )
    pd.DataFrame(corr_data_frame).to_csv(r'sensor_data_corr_vis.csv', index=True)
    return data_pivot, corr_data_frame, device_name_dict


def load_data_pivot():
    path_pivot = r'D:\Git_project\Jupyter_Projects\Smart_house\sensor_data_vis.csv'
    data_pivot = pd.read_csv(path_pivot, index_col='id_sp')

    corr = data_pivot.corr()
    corr = pd.melt(corr.reset_index(),
                   id_vars='index')  # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']
    corr['value'] = corr['value'].abs()
    # фильтруем корреляцию
    # Кореляция и вытакскивание признаков
    CORR_COEFF = 0.70
    corr = corr[(corr['value'] >= CORR_COEFF) & (corr['value'] != 1.0)]

    corr_data_frame = corr.pivot('y', 'x')['value']
    print(corr_data_frame)
    return data_pivot, corr_data_frame


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

        # print(data_feature)
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
    print(data_sensor_lag['HUMSENS2'])
    data_sensor_block = itemgetter(*block)(data_sensor_lag)
    print(data_sensor_block[1].columns)
    data_sensor_filter = [filter_data_frame(data, time) for data in data_sensor_block]
    models = []
    for m in models_path:
        with open(m, 'rb') as f:
            models.append(pickle.load(f).get(lag))
    return data_sensor_filter, models


def get_prediction(data_sensor_filter, models, error=True):
    data_prediction_sensor = []
    у_true = []
    for data_sensor, model_file in zip(data_sensor_filter, models):
        y_clmn = data_sensor.columns[0]
        y = data_sensor[y_clmn]
        у_true.append(y.values[0])
        X = data_sensor.drop([y_clmn], axis=1)
        print(len(X.columns))
        print(y_clmn)
        print(X.columns)
        predict = model_file['model'].predict(X)
        data_prediction_sensor.append(predict[0])
    if error:
        error_mean = mean_absolute_percentage_error(np.array(у_true), np.array(data_prediction_sensor))
    return data_prediction_sensor, ['{}%'.format(round(error_mean, 2).tolist())]
