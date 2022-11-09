import os 
import json
import logging
import argparse
from tqdm import tqdm

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Path to the folder that stores the OpenPI dataset.')
parser.add_argument('--data_type', type=str, help = 'Choose data type from train, dev, and test.')

def load_data(data_path, data_type):
    with open(os.path.join(data_path, f'{data_type}.json'), 'r') as f:
        data = json.load(f)
    f.close()
    return data


if __name__ == '__main__':
    args = parser.parse_args()
    data = load_data(args.data_path, args.data_type)
    data_dict = {}
    for i, (key, value) in enumerate(tqdm(data.items())):
        temp_dict = {}
        cur_dict = value
        temp_dict['goal'] = value['goal']
        temp_dict['topics'] = value['topics']
        cur_states = value['states']
        cur_steps = value['steps']

        temp_dict['steps'] = [{} for i in range(len(cur_steps))]
        for j, step in enumerate(cur_steps):
            temp_dict['steps'][j]['description'] = step
            temp_dict['steps'][j]['state_changes'] = []
        

        for state_dict in cur_states:
            entity = state_dict['entity']
            attr = state_dict['attribute']
            ans_lst = state_dict['answers']
            for k, state_info in enumerate(ans_lst):
                if state_info:
                    before_state, after_state = state_info.split('after:')
                    before_state, after_state = before_state.split(':')[-1].strip()[:-1], after_state.replace('now', '')
                    temp_dict['steps'][k]['state_changes'].append(tuple((entity, attr, before_state.strip(), after_state.strip())))
    
        data_dict[str(i + 1)] = temp_dict

    with open(f'./data/openpi-{args.data_type}-parsed.json', 'w') as f:
        json.dump(data_dict, f, indent=4)
    f.close()
    


