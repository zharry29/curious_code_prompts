import time
import json
import utils
import pickle
import random
import openai
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score

random.seed(29)


class ANLI():
    def __init__(self, templates, idx):
        self.template = templates
        self.idx = idx
    
    def build_text_prompt(self):
        text_prompt = ""
        total_token = 0
        while total_token < args.window_size:
            example_index = random.sample(range(len(dataset[f'train_r{self.idx}'])), 1)[0]
            example = dataset[f'train_r{self.idx}'][example_index]
            input_text, output_text = self.template.apply(example)
            text_prompt += input_text + '\n\nAnswer: ' + output_text + '\n\n\n'
            total_token += len(text_prompt.split())
        return(text_prompt)
    
    def build_code_prompt(self):
        code_prompt = ""
        return code_prompt
    
    def run_llm(self, prompt, model, temperature=0, stop=['\n']):
        model_name = {
            "davinci": "text-davinci-002",
            "curie": "text-curie-001",
            "codex": "code-davinci-002",
            "ada": "text-ada-001",
        }
        while True:
            try:
                ret = openai.Completion.create(
                    engine=model_name[model],
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=300,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=stop
                )
                break
            except Exception as e:
                print(e)
                print("Retrying in 10 seconds")
                time.sleep(10)

        gen_text = ret["choices"][0]["text"].strip()#.split('\n')[0]
        return gen_text
    
    def parse_pred(self, pred):
        if pred.strip().lower() == 'true':
            return 0
        elif pred.strip().lower() == 'neither':
            return 1
        elif pred.strip().lower() == 'false':
            return 2
        else:
            return -1
    
    def predict(self):
        if args.prompt == "text":
            prompt = self.build_text_prompt()
        elif args.prompt == "code":
            prompt = self.build_code_prompt()

        val_data = dataset[f'dev_r{self.idx}']
        val_idx = np.random.choice(np.arange(len(val_data)), 333, replace=False)

        preds = []
        golds = []
        for idx in tqdm(val_idx):
            example = val_data[int(idx)]
            input_text, output_text = self.template.apply(example)
            pred = self.run_llm(prompt + input_text + '\n\nAnswer:', args.model)
            pred = self.parse_pred(pred)
            gold = example['label']
            preds.append(pred)
            golds.append(gold)
        return list(val_idx), preds, golds
    
    def evaluate(self):
        with open(f'{args.model}_pred.txt', 'r') as f:
            preds = [x.strip() for x in f.readlines()]
        with open(f'{args.model}_gold.txt', 'r') as f:
            golds = [x.strip() for x in f.readlines()]
        print("Accuracy", accuracy_score(golds, preds))
        return accuracy_score(golds, preds)


parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, help='Either text or code.')
parser.add_argument('--model', type=str, help='Either davinci, curie or codex.')
parser.add_argument('--window_size', type=int, help='Context window size of GPT3 model.')
#parser.add_argument('--dataset', type=str, help='Name of the datasset')
#parser.add_argument('--xxx', action='store_true', help='')
parser.add_argument('--key', type=str, help='The name of the OpenAI API key file.')


if __name__ == '__main__':
    args = parser.parse_args()
    openai.api_key_path = f'../../_private/{args.key}.key'

    data_name = 'anli'
    NUM_EXAMPLES_IN_PROMPT = 5

    dataset, templates = utils.load_data(data_name)
    val_idx_dict = []
    preds, golds = [], []
    for i in range(1, 4):
        inference_model = ANLI(templates, i)
        val_idx, pred, gold = inference_model.predict()
        val_idx_dict.append(val_idx)
        preds += pred
        golds += gold
    
    with open(f'{args.model}_pred.txt', 'w') as f:
        f.writelines([str(x) + '\n' for x in preds])
    with open(f'{args.model}_gold.txt', 'w') as f:
        f.writelines([str(x) + '\n' for x in golds])

    with open(f'{args.model}_val_idx.pkl', 'wb') as f:
        pickle.dump(val_idx_dict, f)
    f.close()

    inference_model.evaluate()










