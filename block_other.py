import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html

from config import SENSCOM, dropdown_list_date, dropdown_list_lag, lag, date, \
    displayModeBar

#         html.Div([
#             html.Span('mean absolute error - '),
#             html.Span(id='mean_temp_error_sens', style={
#                 'color': 'red'
#
#             })], className='col s4')
SENSCOM_MODULE = [
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='demo-dropdown_lag_sens',
                options=dropdown_list_lag,
                value=lag,
                clearable=False
            ),
        ], className='col s1 m2 l4'),
        html.Div([
            dcc.Dropdown(
                id='demo-dropdown_date_sens',
                options=dropdown_list_date,
                value=date,
                clearable=False
            ),
        ], className='col s4 m2 l2'),
    ], className='row'),
    html.Div([
                 html.Div([
                     daq.Gauge(
                         color={"gradient": True,
                                "ranges": {"green": [250, 350], "yellow": [350, 450], "red": [450, 610]}},
                         showCurrentValue=True,
                         units="%",
                         value=0,
                         id=SENSCOM[0],
                         label=SENSCOM[0],
                         size=140,
                         max=610,
                         min=250,
                     )
                 ], className='col s12 m4 l2'),
                 html.Div([
                     daq.Indicator(
                         id=SENSCOM[1],
                         label=SENSCOM[1],
                         labelPosition="bottom",
                         value=True
                     )

                 ], className='col s12 m4 l2'),
                 html.Div([
                     daq.Gauge(
                         color={"gradient": True,
                                "ranges": {"green": [700, 730], "yellow": [730, 760], "red": [760, 800]}},
                         showCurrentValue=True,
                         units="%",
                         value=0,
                         id=SENSCOM[2],
                         label=SENSCOM[2],
                         size=140,
                         max=800,
                         min=700,
                     )
                 ], className='col s12 m4 l2'),
                 html.Div([
                     daq.Indicator(
                         id='GAZRISE',
                         label='increase and decrease of GAZ',
                         labelPosition="bottom",
                         value=True
                     )

                 ], className='col s12 m4 l2'),

             ] + [html.Div(dcc.Graph(
        id='lag_7_sens',

        config=dict(
            displayModeBar=displayModeBar
        ),
        figure={
            'data': [
                {'x': [1, 2], 'y': [3, 1]}
            ]
        }), className='col s12 m2 l8')] + [html.Div([

        dcc.Graph(
            id='anomalies_sens',

            config=dict(
                displayModeBar=displayModeBar
            ),
            figure={
                'data': [
                    {'x': [1, 2], 'y': [3, 1]}
                ]
            }),
        html.Div(dcc.Slider(
            id='sens-slider',
            step=None,
            min=0,
            max=2,
            marks={
                0: SENSCOM[0],
                2: SENSCOM[2]
            },
            value=0
        ))], className='col s12 m12 l13')], className='row')
]
