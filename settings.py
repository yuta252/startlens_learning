import configparser
import os
from os.path import join, dirname

from dotenv import load_dotenv


conf = configparser.ConfigParser()
conf.read('settings.ini')

load_dotenv(verbose=True)
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

env = conf['environment']['env']
base_dir = os.path.dirname(os.path.abspath(__file__))

bucket = conf['storage']['bucket']
prefix_key = conf['storage']['key']
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

port = conf['webserver']['port']

hdf5_file_name = conf['hdf5']['name']
tflite_file_name = conf['tflite']['name']
