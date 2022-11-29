import os 
import json
from glob import glob
from transformers import GPT2Tokenizer

def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    f.close()
    return data


def gpt3_tokenizer(inp):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    res = tokenizer(inp)['input_ids']
    return len(res)


def load_meta_data(path):
    meta_data = {}
    file_paths = glob(os.path.join(path, '*-parsed.json'))
    for path in file_paths:
        if 'dev' in path:
            with open(path, 'r') as f:
                meta_data['dev'] = json.load(f)
            f.close()
        elif 'train' in path:
            with open(path, 'r') as f:
                meta_data['train'] = json.load(f)
            f.close()
        else:
            with open(path, 'r') as f:
                meta_data['test'] = json.load(f)
            f.close()
    return meta_data 


def choose_openpi_options(openpi_object):
    openpi_choices = openpi_object.split('|')
    openpi_object = openpi_choices[0].strip()
    return openpi_object


def parse_preds(preds):
    preds = preds.split('\n')
    preds_parsed = []
    for pred in preds:
        if pred[0] == '-':
            preds_parsed.append(pred.replace('-', '').strip())
    return preds_parsed