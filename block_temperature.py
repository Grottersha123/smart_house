import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html

from config import TEMPSENS, dropdown_list_date, dropdown_list_lag, lag, date, \
    displayModeBar, style_block

TEMPERATURE_MODULE = [
    html.Div([
        html.Div([html.Img(src='assets/temp.svg', className="left-align image-pd"),
                  html.H5(children='Temperture', className='left-align text_block')]),
        html.Div([
            dcc.Dropdown(
                id='demo-dropdown_lag_temp',
                options=dropdown_list_lag,
                value=lag,
                clearable=False
            ),
        ], className='col s1 m2 l4'),
        html.Div([
            dcc.Dropdown(
                id='demo-dropdown_date_temp',
                options=dropdown_list_date,
                value=date,
                clearable=False
            ),
        ], className='col s4 m2 l2'),
        html.Div([
            html.Span('mean absolute error - '),
            html.Span(id='mean_temp_error_temp', style={
                'color': 'red'

            })], className='col s4')
    ], className='row', style=style_block),
    html.Div([
                 html.Div([
                     daq.Thermometer(
                         id=TEMPSENS[ind],
                         label=TEMPSENS[ind],
                         min=0,
                         max=30,
                         height=150,
                         value=0,
                         showCurrentValue=True,
                         units="C",
                         style={
                         }
                     )
                 ], className='col s12 m4 l2',
                     style={
                         'margin-right': 20
                     })
                 for ind in range(0, 5)
             ] + [html.Div(dcc.Graph(
        id='lag_7_temperature',

        config=dict(
            displayModeBar=displayModeBar
        ),
        figure={
            'data': [
                {'x': [1, 2], 'y': [3, 1]}
            ]
        }), className='col s12 m2 l8')] + [html.Div([

        dcc.Graph(
            id='anomalies_temp',

            config=dict(
                displayModeBar=displayModeBar
            ),
            figure={
                'data': [
                    {'x': [1, 2], 'y': [3, 1]}
                ]
            }),
        html.Div(dcc.Slider(
            id='temp-slider',
            step=None,
            min=0,
            max=len(TEMPSENS) - 1,
            marks={ind: t for ind, t in enumerate(TEMPSENS)},
            value=0
        ))], className='col s12 m12 l13')], className='row', style=style_block)
]
