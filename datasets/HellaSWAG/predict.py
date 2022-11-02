import argparse
import openai
from datasets import load_dataset
import random
random.seed(29)
from promptsource.templates import DatasetTemplates

template = DatasetTemplates('hellaswag')["how_ends"]
dataset = load_dataset("hellaswag")

NUM_EXAMPLES_IN_PROMPT = 5

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, default='', help='Either text or code.')
parser.add_argument('--model', type=str, default='dev', help='Either gpt3 or codex.')
parser.add_argument('--dataset', type=str, default='dev', help='Name of the datasset')
#parser.add_argument('--xxx', action='store_true', help='')
parser.add_argument('--key', type=str, default='harry_ccbgroup', help='The name of the OpenAI API key file.')

args = parser.parse_args()
openai.api_key = open(f'../../_private/{args.key}.key').read()

# Build prompt
text_prompt = ""
example_indices = random.sample(range(len(dataset['train'])), NUM_EXAMPLES_IN_PROMPT)
for example_index in example_indices:
    example = dataset['train'][example_index]
    input_text, output_text = template.apply(example)
    text_prompt += input_text + '\n\n' + output_text + '\n\n\n'
    print(text_prompt)
    break
    