import time
import json
import utils
import random
import pickle
import openai
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score


class HotpotQA():
    def __init__(self, templates):
        self.template = templates
    
    def build_text_prompt(self):
        text_prompt = ""
        total_token = 0
        while total_token < args.window_size:
            example_index = random.sample(range(len(dataset['train'])), 1)[0]
            example = dataset['train'][example_index]
            input_text, output_text = self.template.apply(example)
            if total_token + len(input_text.split()) + len(output_text.split()) > args.window_size:
                break
            text_prompt += input_text + '\n\nAnswer: ' + output_text + '\n\n\n'
            total_token += len(text_prompt.split())
        return(text_prompt)
    
    def build_code_prompt(self):
        code_prompt = ""
        return code_prompt
    
    def run_llm(self, prompt, model, temperature=0.7, stop=['\n']):
        model_name = {
            "davinci": "text-davinci-002",
            "curie": "text-curie-001",
            "codex": "code-davinci-002",
            "ada": "text-ada-001",
        }
        if model == 'codex':
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
        else:
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

        gen_text = ret["choices"][0]["text"].strip()#.split('\n')[0]
        return gen_text
    
    def predict(self):
        if args.prompt == "text":
            prompt = self.build_text_prompt()
        elif args.prompt == "code":
            prompt = self.build_code_prompt()

        val_data = dataset['validation']
        val_idx = np.random.choice(np.arange(len(val_data['answer'])), 1000, replace=False)
        with open(f'{args.model}_val_idx.pkl', 'wb') as f:
            pickle.dump(val_idx, f)
        f.close()
        
        preds = {}
        golds = []
        preds['answer'] = {}
        for i, idx in enumerate(tqdm(val_idx)):
            example = val_data[int(idx)]
            input_text, output_text = self.template.apply(example)
            try:
                pred = self.run_llm(prompt + input_text + '\n\nAnswer:', args.model)
            except openai.error.InvalidRequestError:
                print('Encountered lengthy input, skipping...')
                continue
            gold = example['answer']
            preds['answer'][f'seacow-{i}'] = pred
            golds.append({'_id': f'seacow-{i}', 'answer': gold})
        
        with open(f'{args.model}_pred.json', 'w') as f:
            json.dump(preds, f, indent=4)
        f.close()
        with open(f'{args.model}_gold.pkl', 'wb') as f:
            pickle.dump(golds, f)
        f.close()
    
    def evaluate():
        with open('pred.txt', 'r') as f:
            preds = [x.strip() for x in f.readlines()]
        with open('gold.txt', 'r') as f:
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

    data_name = 'hotpot_qa'
    NUM_EXAMPLES_IN_PROMPT = 5

    dataset, templates = utils.load_data(data_name)
    inference_model = HotpotQA(templates)
    inference_model.predict()
    # inference_model.evaluate()










