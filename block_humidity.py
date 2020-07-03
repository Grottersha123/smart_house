import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output

from config import device_name_dict, dropdown_list_date, dropdown_list_lag, lag, date, \
    HUMSENS, displayModeBar

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
                size=140,
                max=80,
                min=0,
            )
        ], className='col s12 m4 l2')
                 for ind in range(0, 5)
             ] + [html.Div(dcc.Graph(
        id='lag_7_hum',
        config=dict(
            displayModeBar=displayModeBar
        ),
        figure={
            'data': [
                {'x': [1, 2], 'y': [3, 1]}
            ]
        }), className='col s12 m2 l8')] + [html.Div([

        dcc.Graph(
            id='anomalies_hum',

            config=dict(
                displayModeBar=displayModeBar
            ),
            figure={
                'data': [
                    {'x': [1, 2], 'y': [3, 1]}
                ]
            }),
        html.Div(dcc.Slider(
            id='hum-slider',
            step=None,
            min=0,
            max=len(HUMSENS),
            marks={ind: t for ind, t in enumerate(HUMSENS)},
            value=0
        ))], className='col s12 m12 l13')], className='row')
]
