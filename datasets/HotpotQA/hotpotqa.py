import time
import utils
import random
import openai
import logging
import argparse
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score

openai.api_key_path = '../../_private/{args.key}.key'
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
random.seed(29)


class HotpotQA():
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
    
    def build_code_prompt(self):
        code_prompt = ""
        return code_prompt
    
    def run_llm(self, prompt, model, temperature=0, stop=['\n']):
        model_name = {
            "davinci": "text-davinci-002",
            "curie": "text-curie-001",
            "codex": "codex-davinci-002",
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
            prompt = self.build_code_prompt()

        preds = []
        golds = []
        for example in tqdm(dataset['validation']):
            input_text, output_text = self.template.apply(example)
            pred = self.run_llm(prompt + input_text + '\n\nAnswer:', args.model)
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
#parser.add_argument('--dataset', type=str, help='Name of the datasset')
#parser.add_argument('--xxx', action='store_true', help='')
parser.add_argument('--key', type=str, help='The name of the OpenAI API key file.')


if __name__ == '__main__':
    args = parser.parse_args()

    data_name = 'hotpot_qa'
    NUM_EXAMPLES_IN_PROMPT = 5

    dataset, templates = utils.load_data(data_name)
    inference_model = HotpotQA(templates)
    inference_model.predict()
    # inference_model.evaluate()










