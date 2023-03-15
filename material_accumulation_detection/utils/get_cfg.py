import importlib
import os
import sys

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def get_cfgs(config_name):
    config_name = config_name.split("/")[-1].split(".")[0]
    module_name = "." + config_name
    package_name = "configs"
    cfg = importlib.import_module(module_name, package=package_name)
    return cfg
