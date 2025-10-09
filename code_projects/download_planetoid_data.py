import os
import urllib.request

DATASETS = ['cora', 'citeseer', 'pubmed']
BASE_URL = 'https://github.com/kimiyoung/planetoid/raw/master/data'
DATA_DIR = 'data'

os.makedirs(DATA_DIR, exist_ok=True)

def download_file(dataset, filename):
    url = f"{BASE_URL}/{filename}"
    out_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(out_path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, out_path)
    else:
        print(f"{filename} already exists.")

for dataset in DATASETS:
    for suffix in ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']:
        fname = f"ind.{dataset}.{suffix}"
        download_file(dataset, fname)

print("Download finished.")
