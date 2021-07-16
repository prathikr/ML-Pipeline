import sys
base_path = '/Users/prathikr/Desktop/CAIS/ML-Pipeline/'
sys.path.insert(1, base_path)

from utils.pipeline import Pipeline
from os import listdir
from os.path import isfile, join

config_files = [f for f in listdir('tests/configs') if isfile(join('tests/configs', f))]

print('config files:', config_files)

failed = 0
for config in config_files:
    print(f'--- running test for {config} ---')

    try:
        p = Pipeline(base_path + 'tests/configs/' + config)
        p.preprocess()
        p.model()
        p.postprocess()
    except Exception as e:
        print(f'{config} test failed')
        print(e)
        failed += 1

    print('\n')

if failed > 0:
    print('did NOT pass all tests')
else:
    print('all tests passed!')