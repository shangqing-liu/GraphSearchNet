import argparse
import yaml
from core.model_handler import ModelHandler
from core.model_handler_extend import ModelHandlerExtend
from core.utils.search_engine import *
import pandas as pd
import os
import random
import gc
import json
from collections import OrderedDict


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    model = ModelHandler(config)
    model.train()


def test(config, gs=False):
    if not gs:
        print_config(config)
    set_random_seed(config['random_seed'])
    if config['out_dir'] is not None:
        config['pretrained'] = config['out_dir']
        config['out_dir'] = None
    model_handle = ModelHandlerExtend(config)
    model_handle.test()


def build_code_vec_database(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    if config['out_dir'] is not None:
        config['pretrained'] = config['out_dir']
        config['out_dir'] = None
    client = Elasticsearch(timeout=30)
    client.indices.delete(index=config['index_name'], ignore=[404])
    client = create_index(client, config['index_file'])
    model_handle = ModelHandlerExtend(config)
    for file in os.listdir(config['vector_db']):
        if file.endswith('.gz'):
            try:
                print(file)
                file_path = os.path.join(config['vector_db'], file)
                model_handle.prepare_vector_db(file_path)
                model_handle.build_code_vec_database(client)
                client.indices.refresh(index=config['index_name'])
            except:
                continue
    print('build index successfully')


def create_search_engine(config):
    set_random_seed(config['random_seed'])
    if config['out_dir'] is not None:
        config['pretrained'] = config['out_dir']
        config['out_dir'] = None
    model_handle = ModelHandlerExtend(config)
    se = search_engine(model_handle=model_handle, config=config)
    queries = pd.read_csv(config['query_file']).values.tolist()
    se.search(queries)


def create_index(client, index_file):
    with open(index_file) as index_file:
        source = index_file.read().strip()
        client.indices.create(index=config['index_name'], body=source)
    return client


def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")



def load_saved_models(dir):
    file_paths = os.listdir(dir)
    records = {}
    for file_path in file_paths:
        if 'Java_Graph2Search' in file_path:
            with open(os.path.join(dir, file_path, 'config.json'), 'r') as f:
                config = json.load(f)
            set_random_seed(config['random_seed'])
            if config['out_dir'] is not None:
                config['pretrained'] = config['out_dir']
                config['out_dir'] = None
            model_handle = ModelHandlerExtend(config)
            format_str = model_handle.test()
            records[file_path] = format_str
    for record in records:
        print(record)
        print(records[record])
        print('***********************************')


if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])
    main(config)
    test(config)
    build_code_vec_database(config)
    create_search_engine(config)