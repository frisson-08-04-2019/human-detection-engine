from urllib.parse import urlparse
import json
import os
import tarfile
import requests
from tqdm import tqdm
import sys

config_path = 'config.json'
nets_dir = 'nets'
chunk_size = 32 * 1024

with open(config_path, 'r') as f:
    config = json.load(f)
model_path = config['model_path']
model_url = config['model_url']

if not os.path.exists(nets_dir):
    os.makedirs(nets_dir)

file_name = os.path.basename(urlparse(model_url).path)
file_path = os.path.join(nets_dir, file_name)

with requests.get(model_url, stream=True) as response:
    total_length = int(response.headers.get('content-length', 0))

    with open(file_path, 'wb') as out_file:
        with tqdm(unit='MB', file=sys.stdout,
                  total=round(total_length / 1024 / 1024, 2)) as pbar:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    out_file.write(chunk)
                pbar.update(chunk_size / 1024 / 1024)

with tarfile.open(file_path) as tf:
    tf.extractall(nets_dir)
