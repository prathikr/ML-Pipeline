import json

class Pipeline():

    def __init__(self, config_filename):
        self.config_filename = config_filename
        self.validate_config()

    def validate_config(self):
        with open(self.config_filename) as f:
            data = json.load(f)
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