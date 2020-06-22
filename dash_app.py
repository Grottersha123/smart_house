import warnings

import dash
import dash_html_components as html
from dash.dependencies import Input, Output

from HUMIDITY import HUMIDITY_MODULE
from TEMPERATURE import TEMPERATURE_MODULE
from config import device_name_dict, TEMPSENS, TEMEPSENS_MODELS, HUMSENS_MODELS, HUMSENS
from prediction import create_data_for_pred, load_data_pivot, prepared_block_lag_data, get_prediction

warnings.filterwarnings('ignore')

app = dash.Dash('smart_house', external_stylesheets=[
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
], className='row')] + TEMPERATURE_MODULE + HUMIDITY_MODULE, className="container")


# TEMPERATURE_MODULE +
@app.callback(
    [Output(TEMPSENS[ind - 1], 'value') for ind in range(1, 6)] + [Output('mean_temp_error_temp', 'children')],
    [Input('demo-dropdown_lag_temp', 'value'), Input('demo-dropdown_date_temp', 'value')])
def callback_predict(value, value_date):
    filter_data, models = prepared_block_lag_data(data_sensor, value, TEMPSENS, value_date,
                                                  models_path=TEMEPSENS_MODELS)

    values, error_mean = get_prediction(filter_data, models)
    result = values + error_mean
    return result


# HUMIDITY_MODULE
@app.callback(
    [Output(HUMSENS[ind - 1], 'value') for ind in range(1, 6)] + [Output('mean_temp_error_hum', 'children')],
    [Input('demo-dropdown_lag_hum', 'value'), Input('demo-dropdown_date_hum', 'value')])
def callback_predict(value, value_date):
    filter_data, models = prepared_block_lag_data(data_sensor, value, HUMSENS, value_date,
                                                  models_path=HUMSENS_MODELS)
    values, error_mean = get_prediction(filter_data, models)
    result = values + error_mean
    return result


if __name__ == '__main__':
    app.run_server(debug=True)
