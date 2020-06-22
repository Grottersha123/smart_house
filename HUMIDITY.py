import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output

from config import device_name_dict, dropdown_list_date, dropdown_list_lag, lag, date, \
    HUMSENS

HUMIDITY_MODULE = [
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='demo-dropdown_lag_hum',
                options=dropdown_list_lag,
                value=lag,
                clearable=False
            ),
        ], className='col s1 m2 l4'),
        html.Div([
            dcc.Dropdown(
                id='demo-dropdown_date_hum',
                options=dropdown_list_date,
                value=date,
                clearable=False
            ),
        ], className='col s4 m2 l2'),
        html.Div([
            html.Span('mean absolute error - '),
            html.Span(id='mean_temp_error_hum', style={
                'color': 'red'

            })], className='col s4')
    ], className='row'),
    html.Div([
        html.Div([
            daq.Gauge(
                showCurrentValue=True,
                units="%",
                value=0,
                id=HUMSENS[ind],
                label=HUMSENS[ind],
                size=150,
                max=80,
                min=0,
            )
        ], className='col s12 m4 l2',
            style={
                'margin-right': 20
            })
        for ind in range(0, 5)
    ], className='row')
]
