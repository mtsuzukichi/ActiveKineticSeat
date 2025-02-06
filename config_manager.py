# config_manager.py

import configparser

CONFIG_FILE = 'config.ini'

def read_config():
    config = configparser.ConfigParser()
    config.optionxform = str  # キーを小文字に変換しない
    config.read(CONFIG_FILE)
    return config

def write_config(config):
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)
