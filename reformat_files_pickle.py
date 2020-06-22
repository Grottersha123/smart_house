import os
from config import models_dir, ROOT_DIR
import pickle as pk

models_dir = os.path.join(ROOT_DIR, r'sense')
# Для того чтобы законвертировать файлы
def save_models(lst, filename):
    filename = '{}.sav'.format(filename)
    pk.dump(lst, open(filename, 'wb'))
    print('save models', filename)


models = [os.path.join(models_dir, m) for m in os.listdir(models_dir)]

for m in models:
    with open(m, 'rb') as f:
        models_lst = pk.load(f)
        models_lst = dict(('lag_{}'.format(index + 1), m) for index, m in enumerate(models_lst))
        save_models(models_lst, m.split('.')[0])
