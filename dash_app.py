from config import TEMPSENS, TEMEPSENS_MODELS
from prediction import load_data, create_data_for_pred, prepared_block_lag_data, get_prediction

data_pivot, corr_data_frame, device_name_dict = load_data()
print(device_name_dict)
data_sensor = dict(
    ('lag_{}'.format(i), create_data_for_pred(data_pivot, corr_data_frame, pred_d=i, lag=8, time=True)) for i in
    range(1, 8))

# Block
# TEMPSENS1
# TEMPSENS3
# TEMPSENS4
# TEMPSENS5
# TEMPSENS6


# Block

filter_data, models = prepared_block_lag_data(data_sensor, 'lag_7', TEMPSENS, '2020-04-22',
                                              models_path=TEMEPSENS_MODELS)
print(get_prediction(filter_data, models))
