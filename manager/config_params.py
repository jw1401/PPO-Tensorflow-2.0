import os
import yaml
from .env_vars import DESC_FILE_PATH
from utils import singleton


@singleton
class ConfigParams():

    def __init__(self):

        self.params = None
        self._set_path = ""
        self.running_config = ""

        try:
            with open(DESC_FILE_PATH) as file:
                self.description = yaml.safe_load(file)
        except:
            self.description = None
            print("DESCRIPTION FILE NOT FOUND ... ")
        
    def load_yaml_config(self, path= ""):

        self._set_path = path

        try:
            with open(path) as file:
                self.params = yaml.safe_load(file)
        except Exception as ex:
            print("ERROR >> " + str(ex))

        return [self.params, self.description, self._set_path]

    def save_yaml_config(self, new_params= None, path= ""):

        self._set_path = path

        self.update_dict(self.params, '_', "_", reset_bools=True)

        for key, value in new_params:
            self.update_dict(self.params, key, value, reset_bools= False)

        try:
            with open(path, 'w') as file:
                y = yaml.dump(self.params, file, default_flow_style= False)
        except Exception as ex:
            print("ERROR >> " + str(ex))

        return [self.params, self.description, self._set_path]

    def update_dict(self, d, serach_key, new_value, reset_bools=False):
        for k, v in d.items():
            if isinstance(v, dict):
                self.update_dict(v, serach_key, new_value, reset_bools)
            else:
                if reset_bools:
                    if isinstance(v, bool):
                        d[k] = False

                if k == serach_key:
                    if isinstance(v, list):
                        d[k] = list(eval(new_value))
                        continue
                    if isinstance(v, bool):
                        d[k] = eval(new_value)
                        continue
                    d[k] = type(v)(new_value)
