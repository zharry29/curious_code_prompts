import argparse
from ast import main
import openai
from datasets import load_dataset
import random
#random.seed(29)
from promptsource.templates import DatasetTemplates
import time
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', required=True, type=str, help='Either text or code.')
parser.add_argument('--model', required=True, type=str, help='Either davinci, curie or codex.')
#parser.add_argument('--dataset', type=str, help='Name of the datasset')
#parser.add_argument('--xxx', action='store_true', help='')
parser.add_argument('--key', required=True, type=str, help='The name of the OpenAI API key file.')

args = parser.parse_args()
openai.api_key = open(f'../../_private/{args.key}.key').read()

NUM_EXAMPLES_IN_PROMPT = 5
SELECTED_PROMPT_NAME = "how_ends"

template = DatasetTemplates('story_cloze')
print(template.templates)
raise SystemExit()
#[SELECTED_PROMPT_NAME]
dataset = load_dataset("story_cloze")

def predict():
    # Build prompt
    def build_text_prompt():
        text_prompt = ""
        example_indices = random.sample(range(len(dataset['train'])), NUM_EXAMPLES_IN_PROMPT)
        for example_index in example_indices:
            example = dataset['train'][example_index]
            input_text, output_text = template.apply(example)
            text_prompt += input_text + '\n\nAnswer: ' + output_text + '\n\n\n'
            #print(text_prompt)
        return(text_prompt)

    def build_code_prompt():
        code_prompt = ""
        return code_prompt

    def run_llm(prompt, model, temperature=0, stop=['\n']):
        model_name = {
            "davinci": "text-davinci-002",
            "curie": "text-curie-001",
            "ada": "text-ada-001",
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


    if args.prompt == "text":
        prompt = build_text_prompt()
    elif args.prompt == "code":
        prompt = build_code_prompt()
    #print(prompt)
    preds = []
    golds = []
    for example in dataset['validation']:
        input_text, output_text = template.apply(example)
        pred = run_llm(prompt + input_text + '\n\nAnswer: Ending', args.model)
        gold = example['label']
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
    return "Accuracy", accuracy_score(golds, preds)

if __name__ == "__main__":
    predict()
    evaluate()