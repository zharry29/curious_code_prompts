import argparse
from ast import main
import openai
from datasets import load_dataset
import random
#random.seed(29)
from promptsource.templates import DatasetTemplates
import time
from sklearn.metrics import accuracy_score
import pickle
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', required=True, type=str, help='Either text or code.')
parser.add_argument('--model', required=True, type=str, help='Either davinci, curie or codex.')
parser.add_argument('--max_prompt', type=int, help='Maximum number of tokens in the prompt.')
parser.add_argument('--key', required=True, type=str, help='The name of the OpenAI API key file.')

args = parser.parse_args()
openai.api_key = open(f'../../_private/{args.key}.key').read()

SELECTED_PROMPT_NAME = "based_on_that"

template = DatasetTemplates('yelp_review_full')[SELECTED_PROMPT_NAME]
dataset = load_dataset("yelp_review_full")

def predict():
    # Build prompt
    def build_text_prompt(inference_input_text):
        text_prompt = ""
        prev_prompt = ""
        example_indices = random.sample(range(len(dataset['train'])), 100)
        for example_index in example_indices:
            if len(tokenizer(text_prompt + inference_input_text)['input_ids']) > args.max_prompt:
                break
            example = dataset['train'][example_index]
            input_text, output_text = template.apply(example)
            prev_prompt = text_prompt
            text_prompt += input_text + ' ' + output_text + '.\n\n'

        return(prev_prompt + inference_input_text)

    def build_code_prompt():
        code_prompt = ""
        return code_prompt

    def run_llm(prompt, model, temperature=0, stop=['\n']):
        model_name = {
            "davinci": "text-davinci-002",
            "curie": "text-curie-001",
            "ada": "text-ada-001",
            "codex": "code-davinci-002",
        }
        while True:
            try:
                ret = openai.Completion.create(
                    engine=model_name[model],
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=5,
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

    preds = []
    golds = []
    print("Total examples: ", len(dataset['test']))
    count = 0
    #print(len(dataset['test']))
    #raise SystemExit
    with open("sampled_1000_indices.pkl", "rb") as f:
        indices = pickle.load(f)
    for index in indices:
        example = dataset['test'][index]
        count += 1
        print(count)
        input_text, output_text = template.apply(example)
        if args.prompt == "text":
            prompt = build_text_prompt(input_text)
        elif "code" in args.prompt:
            prompt = build_code_prompt(args.prompt, input_text)
        #print(prompt)
        #print(len(tokenizer(prompt)['input_ids']))
        #raise SystemExit
        pred_text = run_llm(prompt, args.model)
        pred = pred_text.strip()[0]
        gold = example['label']
        preds.append(pred)
        golds.append(gold)

    with open(f'pred_{args.model}_{args.prompt}.txt', 'w') as f:
        f.writelines([str(x) + '\n' for x in preds])
    with open('gold.txt', 'w') as f:
        f.writelines([str(x) + '\n' for x in golds])

def evaluate():
    with open(f'pred_{args.model}_{args.prompt}.txt', 'r') as f:
        preds = [int(l.strip()) for l in f.readlines()]
    with open('gold.txt', 'r') as f:
        golds = [int(l.strip()) + 1 for l in f.readlines()]
    print("Pearson's R", pearsonr(golds, preds)[0])
    return "Pearson's R", pearsonr(golds, preds)[0]

if __name__ == "__main__":
    predict()
    evaluate()