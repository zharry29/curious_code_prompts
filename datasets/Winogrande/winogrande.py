import argparse
import ast
import pickle
import random
import time

import numpy as np
import openai
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import utils
from codex_prompts import CodexPrompts


class Winogrande():
    def __init__(self, templates):
        self.template = templates

    def build_text_prompt(self):
        text_prompt = ""
        total_token = 0
        while total_token < args.window_size:
            example_index = random.sample(range(len(dataset['train'])), 1)
            example = dataset['train'][example_index]
            input_text, output_text = self.template.apply(example)
            text_prompt += input_text.replace('[', '').replace(']', '') + '\n\nAnswer: ' + output_text.replace('[', '').replace(']', '') + '\n\n\n'
            total_token += len(text_prompt.split())
        return text_prompt

    def build_code_prompt(self, style, train=True):
        text_prompt = self.build_text_prompt()
        text_prompt_lst = [s for s in text_prompt.split('\n') if s]
        codex_prompt = CodexPrompts(text_prompt_lst, train)
        code_prompt = eval('codex_prompt.' + str(style))()
        return code_prompt

    def parse_input(self, inp):
        return '- ' + "'" + inp.replace('-', '').strip() + "'"

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

    def predict(self):
        if args.prompt == "text":
            prompt = self.build_text_prompt()
        elif args.prompt == "code":
            prompt = self.build_code_prompt(args.style)

        val_data = dataset['validation']
        val_idx = np.random.choice(np.arange(len(val_data['answer'])), 1000, replace=False)
        with open(f'{args.model}_val_idx.pkl', 'wb') as f:
            pickle.dump(val_idx, f)
        f.close()

        preds = []
        golds = []
        for idx in tqdm(val_idx):
            example = val_data[int(idx)]
            input_text, output_text = self.template.apply(example)
            options = '[' + ','.join(['"' + e.replace('-', '').lower().strip() + '"' for e in input_text.split('\n')[-2:]]) + ']'
            options = ast.literal_eval(options)

            # if args.model != 'codex':
            input_text = input_text.split('\n')
            input_text[-2], input_text[-1] = self.parse_input(input_text[-2]), self.parse_input(input_text[-1])
            input_text = '\n'.join(input_text)
            pred = self.run_llm(prompt + input_text + '\n\nAnswer:', args.model)
            pred = pred.replace("'", '').replace('"', '').lower().strip()
            if pred in options:
                pred = str(options.index(pred) + 1)
            else:
                pred = str(-1)
            # else:
            #     codex_prompt = CodexPrompts(input_text.split('\n'), train=False)
            #     inference_prompt = eval('codex_prompt.' + str(args.style))()
            #     pred = self.run_llm(inference_prompt, args.model)
            gold = example['answer']
            preds.append(pred)
            golds.append(gold)

        with open(f'{args.model}_pred.txt', 'w') as f:
            f.writelines([x + '\n' for x in preds])
        with open(f'{args.model}_gold.txt', 'w') as f:
            f.writelines([x + '\n' for x in golds])

    def evaluate(self):
        with open(f'{args.model}_pred.txt', 'r') as f:
            preds = [x.strip() for x in f.readlines()]
        with open(f'{args.model}_gold.txt', 'r') as f:
            golds = [x.strip() for x in f.readlines()]
        print("Accuracy", accuracy_score(golds, preds))
        print("F1 Score", f1_score(golds, preds, average='macro'))
        return accuracy_score(golds, preds)


parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, default='text',help='Either text or code.')
parser.add_argument('--model', type=str, help='Either davinci, curie or codex.')
parser.add_argument('--style', type=str, help='choose style of code prompt from one of ["vanilla", "good_var_name", "with_comments", "class_obj"]')
parser.add_argument('--window_size', type=int, help='token threshold for GPT3 prompt.')
#parser.add_argument('--dataset', type=str, help='Name of the datasset')
#parser.add_argument('--xxx', action='store_true', help='')
parser.add_argument('--key', type=str, help='The name of the OpenAI API key file.')


if __name__ == '__main__':
    args = parser.parse_args()
    openai.api_key_path = f'../../_private/{args.key}.key'

    data_name = 'winogrande'

    dataset, templates = utils.load_data(data_name)
    inference_model = Winogrande(templates)

    # # !============ Code in development ===============
    # if args.prompt == 'text':
    #     test = inference_model.build_text_prompt()
    # elif args.prompt == 'code':
    #     test = inference_model.build_code_prompt(args.style)
    # print(test)
    # raise SystemExit()
    # # !============ Code in development ===============

    # inference_model.predict()
    inference_model.evaluate()










