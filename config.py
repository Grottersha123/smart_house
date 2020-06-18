import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(ROOT_DIR, r'sensor_models_8_8')

# block_1 TempSENS
TEMPSENS = ['TEMPSENS1',
            'TEMPSENS3',
            'TEMPSENS4',
            'TEMPSENS5',
            'TEMPSENS6']

TEMEPSENS_MODELS = [os.path.join(models_dir, model) for model in os.listdir(models_dir) if
                    model.split('-')[0] in TEMPSENS]
