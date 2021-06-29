import json

class Pipeline():

    def __init__(self, config_filename):
        print("--- __init__ ---")
        self.config_filename = config_filename
        self.validate_config()

    def validate_config(self):
        print("--- validate config ---")
        with open(self.config_filename) as f:
            data = json.load(f)
        print("Config:", data)
        return True

    def preprocess(self):
        print("--- preprocess ---")
        return True

    def model(self):
        print("--- model ---")
        return True

    def postprocess(self):
        print("--- postprocessing ---")
        return True

if __name__ == '__main__':
    p = Pipeline('/Users/prathikr/Desktop/CAIS/ML-Pipeline/example_config.json')