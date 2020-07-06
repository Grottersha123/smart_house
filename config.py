import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(ROOT_DIR, r'sensor_models_8_8')
displayModeBar = True
###
# START VALUE
lag = 'lag_1'
date = '2019-03-1'
###
COLORS_SENS = ['#3883A3', '#FFB15A',
               '#37A463',
               '#57B9E4',
               '#6577E8']

style_block = {'background': '#F7F7F7',
               'box-shadow': '5px 8px 15px rgba(0, 0, 0, 0.25)',
               'padding': '10px 10px 10px 10px',
               'border-radius': '20px'}

COLOR_BG = '#F7F7F7'

COORS_ANOMAL = ['#69B3F8', '#F0A432']
# block_1 TempSENS
TEMPSENS = ['TEMPSENS1',
            'TEMPSENS3',
            'TEMPSENS4',
            'TEMPSENS5',
            'TEMPSENS6']

TEMEPSENS_MODELS = [os.path.join(models_dir, model) for model in os.listdir(models_dir) if
                    model.split('-')[0] in TEMPSENS]

HUMSENS = ['HUMSENS1',
           'HUMSENS2',
           'HUMSENS3',
           'HUMSENS4',
           'HUMSENS5']

HUMSENS_MODELS = [os.path.join(models_dir, model) for model in os.listdir(models_dir) if
                  model.split('-')[0] in HUMSENS]

SENSCOM = ['GAZDTCT1', 'MOTSENS1',
           'PRESSENS1']

SENSCOM_MODELS = [os.path.join(models_dir, model) for model in os.listdir(models_dir) if
                  model.split('-')[0] in SENSCOM]

device_name_dict = {'HUMSENS1': 'HUMSENS1-Влажность в котельной', 'TEMPSENS2': 'TEMPSENS2-Температура под крыльцом',
                    'TEMPSENS1': 'TEMPSENS1-Температура рядом с котлом',
                    'PRESSENS1': 'PRESSENS1-Давление наружного воздуха', 'GAZDTCT1': 'GAZDTCT1-Газ рядом с котлом',
                    'HUMSENS2': 'HUMSENS2-Влажность юго-восточная комната',
                    'HUMSENS3': 'HUMSENS3-Влажность юго-западная комната',
                    'HUMSENS4': 'HUMSENS4-Влажность северо-восточная комната',
                    'HUMSENS5': 'HUMSENS5-Влажность северо-западная комната',
                    'MOTSENS1': 'MOTSENS1-Присутствие юго-западная комната',
                    'TEMPSENS3': 'TEMPSENS3-Температура юго-восточная комната',
                    'TEMPSENS4': 'TEMPSENS4-Температура юго-западная комната',
                    'TEMPSENS5': 'TEMPSENS5-Температура северо-восточная комната',
                    'TEMPSENS6': 'TEMPSENS6-Температура северо-западная комната',
                    'MOTSENS10': 'MOTSENS10-Движение тамбур', 'HUMSENS10': 'HUMSENS10-Влажность тамбур',
                    'HUMSENS11': 'HUMSENS11-Влажность столовая', 'MOTSENS9': 'MOTSENS9-Движение столовая',
                    'TEMPSENS11': 'TEMPSENS11-Температура тамбур', 'TEMPSENS12': 'TEMPSENS12-Температура столовая',
                    'SRN1': 'SRN1-Сирена при входе'}

dropdown_list_lag = [{'label': '{} day ahead forecast'.format(i), 'value': 'lag_{}'.format(i)} for i in range(1, 8)]
dropdown_list_date = [{'label': '2019-03-{}'.format(i), 'value': '2019-03-{}'.format(i)} for i in range(1, 28)]
