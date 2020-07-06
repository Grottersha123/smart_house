import warnings

import dash
import dash_html_components as html
from adtk.detector import MinClusterDetector, QuantileAD
from dash.dependencies import Input, Output
from sklearn.cluster import KMeans

from block_humidity import HUMIDITY_MODULE
from block_other import SENSCOM_MODULE
from block_temperature import TEMPERATURE_MODULE
from config import device_name_dict, TEMPSENS, TEMEPSENS_MODELS, HUMSENS_MODELS, HUMSENS, SENSCOM, SENSCOM_MODELS
from load_data import load_data_pivot
from prediction import create_data_for_pred, prepared_block_lag_data, get_prediction, \
    create_scatter_plot, get_pred_7, find_diff, anomalies_find, create_plot

warnings.filterwarnings('ignore')

app = dash.Dash(__name__, external_stylesheets=[
    'https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css'
], external_scripts=['https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js'])

data_pivot, corr_data_frame = load_data_pivot()

print(device_name_dict)

data_sensor = dict(
    ('lag_{}'.format(i), create_data_for_pred(data_pivot, corr_data_frame, pred_d=i, lag=8, time=True)) for i in
    range(1, 8))

app.layout = html.Div([html.Div([
    html.H2('Smart_home sensors prediction app',
            style={'float': 'left',
                   }),
], className='row')] + TEMPERATURE_MODULE + HUMIDITY_MODULE + SENSCOM_MODULE, className="container")


# TEMPERATURE_MODULE
@app.callback(
    [Output(TEMPSENS[ind - 1], 'value') for ind in range(1, 6)] + [Output('mean_temp_error_temp', 'children')] + [
        Output('lag_7_temperature', 'figure')],
    [Input('demo-dropdown_lag_temp', 'value'), Input('demo-dropdown_date_temp', 'value')])
def callback_predict(value, value_date):
    filter_data, models = prepared_block_lag_data(data_sensor, value, TEMPSENS, value_date,
                                                  models_path=TEMEPSENS_MODELS)

    values, error_mean = get_prediction(filter_data, models)

    convert_data, covert_mean = get_pred_7(data_sensor, TEMPSENS, value_date, models_path=TEMEPSENS_MODELS)
    fig = create_scatter_plot(convert_data, covert_mean, TEMPSENS)
    result = values + error_mean + [fig]
    return result


# HUMIDITY_MODULE
@app.callback(
    [Output(HUMSENS[ind - 1], 'value') for ind in range(1, 6)] + [Output('mean_temp_error_hum', 'children')] + [
        Output('lag_7_hum', 'figure')],
    [Input('demo-dropdown_lag_hum', 'value'), Input('demo-dropdown_date_hum', 'value')])
def callback_predict(value, value_date):
    filter_data, models = prepared_block_lag_data(data_sensor, value, HUMSENS, value_date,
                                                  models_path=HUMSENS_MODELS)
    values, error_mean = get_prediction(filter_data, models)
    convert_data, covert_mean = get_pred_7(data_sensor, HUMSENS, value_date, models_path=HUMSENS_MODELS)
    fig = create_scatter_plot(convert_data, covert_mean, HUMSENS)
    result = values + error_mean + [fig]
    return result


# SENSCOM_MODULE
@app.callback(
    [Output(SENSCOM[ind - 1], 'value') for ind in range(1, 4)] + [Output('GAZRISE', 'color')] + [
        Output('lag_7_sens', 'figure')],
    [Input('demo-dropdown_lag_sens', 'value'), Input('demo-dropdown_date_sens', 'value')])
def callback_predict(value, value_date):
    filter_data, models = prepared_block_lag_data(data_sensor, value, SENSCOM, value_date,
                                                  models_path=SENSCOM_MODELS)
    val_gaz = filter_data[0].values[0][0]
    values, error_mean = get_prediction(filter_data, models)
    diff_gaz = find_diff(val_gaz, float(values[0]))
    convert_data, covert_mean = get_pred_7(data_sensor, SENSCOM, value_date, models_path=SENSCOM_MODELS)
    fig = create_scatter_plot(convert_data, covert_mean, SENSCOM)
    result = values + [diff_gaz] + [fig]
    return result


# TODO:// убрать хардкод с квантилями
@app.callback(
    Output('anomalies_hum', 'figure'),
    [Input('hum-slider', 'value')])
def update_figure(value):
    detector = QuantileAD(high=0.99, low=0.15)

    anomal_data = anomalies_find(data_pivot, TEMPSENS[value], detector)
    plots = create_plot(anomal_data)
    return plots[0]


# TODO:// убрать хардкод с квантилями
@app.callback(
    Output('anomalies_temp', 'figure'),
    [Input('temp-slider', 'value')])
def update_figure(value):
    detector = QuantileAD(high=0.99, low=0.15)

    anomal_data = anomalies_find(data_pivot, TEMPSENS[value], detector)
    plots = create_plot(anomal_data)

    return plots[0]


# TODO:// убрать хардкод с квантилями
@app.callback(
    Output('anomalies_sens', 'figure'),
    [Input('sens-slider', 'value')])
def update_figure(value):
    detector = MinClusterDetector(KMeans(n_clusters=2)) if value == 0 else QuantileAD(high=0.99, low=0.15)

    anomal_data = anomalies_find(data_pivot, SENSCOM[value], detector)
    plots = create_plot(anomal_data)
    return plots[0]


if __name__ == '__main__':
    app.run_server(debug=True)
