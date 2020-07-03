import logging

import pandas as pd

from prediction import DROP_CLM
from utils import sensor_data_mean


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
    path_corr = r'D:\Git_project\Jupyter_Projects\Smart_house\sensor_data_vis_corr.csv'
    data_corr = pd.read_csv(path_corr)
    corr = data_corr.corr()
    corr = pd.melt(corr.reset_index(),
                   id_vars='index')  # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']
    corr['value'] = corr['value'].abs()
    # фильтруем корреляцию
    # Кореляция и вытакскивание признаков
    CORR_COEFF = 0.7
    corr = corr[(corr['value'] >= CORR_COEFF) & (corr['value'] != 1.0)]

    corr_data_frame = corr.pivot('y', 'x')['value']
    return data_pivot, corr_data_frame
