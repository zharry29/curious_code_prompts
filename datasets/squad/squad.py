import argparse
import openai
from datasets import load_dataset
import random
random.seed(29)
from promptsource.templates import DatasetTemplates
import time
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', required=True, type=str, help='Either text or code.')
parser.add_argument('--model', required=True, type=str, help='Either davinci, curie or codex.')
parser.add_argument('--key', required=True, type=str, help='The name of the OpenAI API key file.')

args = parser.parse_args()
openai.api_key = open(f'../../_private/{args.key}.key').read()

NUM_EXAMPLES_IN_PROMPT = 5
SELECTED_PROMPT_NAME = "Questions with Context +unanswerable"

template = DatasetTemplates("squad_v2")[SELECTED_PROMPT_NAME]
dataset = load_dataset("squad_v2")

def predict():
    # Build prompt
    def build_text_prompt():
        text_prompt = ""
        example_indices = random.sample(range(len(dataset['train'])), NUM_EXAMPLES_IN_PROMPT)
        for example_index in example_indices:
            example = dataset['train'][example_index]
            input_text, output_text = template.apply(example)
            text_prompt += input_text + ' ' + output_text + '\n\n'
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
    #print("Total examples: ", len(dataset['validation']))
    count = 0
    for example in dataset['validation']:
        count += 1
        # if count > 20:
        #     break
        input_text, output_text = template.apply(example)
        pred = run_llm(prompt + input_text, args.model)
        gold = example["answers"]["text"]
        preds.append(pred if normalize_text(pred) != 'unanswerable' else '')
        golds.append(gold if len(gold) > 0 else [''])

    with open('pred.txt', 'w') as f:
        f.writelines([str(x) + '\n' for x in preds])
    with open('gold.txt', 'w') as f:
        f.writelines([str(x) + '\n' for x in golds])

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def evaluate():
    with open('pred.txt', 'r') as f:
        preds = [x.strip() for x in f.readlines()]
    with open('gold.txt', 'r') as f:
        golds = [eval(x.strip()) for x in f.readlines()]
    em_scores = []
    f1_scores = []
    for pred, gold in zip(preds,golds):
        em = max((compute_exact_match(pred, answer)) for answer in gold)
        f1 = max((compute_f1(pred, answer)) for answer in gold)
        em_scores.append(em)
        f1_scores.append(f1)
    em = sum(em_scores) / len(em_scores)
    f1 = sum(f1_scores) / len(f1_scores) #TODO: Double check that this is the right way to report this
    print("EM", em)
    print("F1", f1)
    return em, f1

if __name__ == "__main__":
    predict()
    evaluate()