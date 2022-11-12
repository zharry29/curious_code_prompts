import time
import utils
import random
import openai
import logging
import argparse
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score

from codex_prompts import CodexPrompts

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
random.seed(29)


class Winogrande():
    def __init__(self, templates):
        self.template = templates
    
    def build_text_prompt(self):
        text_prompt = ""
        example_indices = random.sample(range(len(dataset['train'])), NUM_EXAMPLES_IN_PROMPT)
        for example_index in example_indices:
            example = dataset['train'][example_index]
            input_text, output_text = self.template.apply(example)
            text_prompt += input_text + '\n\nAnswer: ' + output_text + '\n\n\n'
        return(text_prompt)
    
    def build_code_prompt(self, style, train=True):
        text_prompt = self.build_text_prompt()
        text_prompt_lst = [s for s in text_prompt.split('\n') if s]
        codex_prompt = CodexPrompts(text_prompt_lst, train)
        code_prompt = eval('codex_prompt.' + str(style))()
        return code_prompt
    
    def run_llm(self, prompt, model, temperature=0, stop=['\n']):
        model_name = {
            "davinci": "text-davinci-002",
            "curie": "text-curie-001",
            "codex": "code-davinci-002",
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

        preds = []
        golds = []
        c = 0
        for example in tqdm(dataset['validation']):
            input_text, output_text = self.template.apply(example)
            
            # print(prompt + input_text + '\n\nAnswer:')
            if args.model == 'text':
                pred = self.run_llm(prompt + input_text + '\n\nAnswer:', args.model)
            else:
                codex_prompt = CodexPrompts(input_text.split('\n'), train=False)
                inference_prompt = eval('codex_prompt.' + str(args.style))()
            prompt += inference_prompt
            pred = self.run_llm(prompt, args.model)
            print(prompt + pred)
            raise SystemExit()
            gold = example['answer']
            preds.append(pred)
            golds.append(gold)
        
        with open('pred.txt', 'w') as f:
            f.writelines([x + '\n' for x in preds])
        with open('gold.txt', 'w') as f:
            f.writelines([x + '\n' for x in golds])
    
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
parser.add_argument('--style', type=str, help='choose style of code prompt from one of ["vanilla", "good_var_name", "with_comments", "class_obj"]')
#parser.add_argument('--dataset', type=str, help='Name of the datasset')
#parser.add_argument('--xxx', action='store_true', help='')
parser.add_argument('--key', type=str, help='The name of the OpenAI API key file.')


if __name__ == '__main__':
    args = parser.parse_args()
    openai.api_key_path = f'../../_private/{args.key}.key'

    data_name = 'winogrande'
    NUM_EXAMPLES_IN_PROMPT = 5

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
    
    inference_model.predict()
    inference_model.evaluate()










