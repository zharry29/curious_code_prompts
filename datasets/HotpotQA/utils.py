import os
import logging
from datasets import load_dataset
from transformers import GPT2Tokenizer
from promptsource.templates import DatasetTemplates
from promptsource.templates import TemplateCollection

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def gpt3_tokenizer(inp):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    res = tokenizer(inp)['input_ids']
    return len(res)


def check_subset(dataset):
    collection = TemplateCollection().datasets_templates

    subset_options = []
    for data_name, subset_name in collection.keys():
        if data_name == dataset:
            subset_options.append(subset_name)

    SELECTED_SUBSET_NAME = None
    if subset_options:
        print(f"\nChoose a subset of {dataset}:\n")
        for idx, option in enumerate(subset_options):
            print(str(idx) + '. ' + option)
        # select_idx = input("\nSelect the index: ")
        select_idx = 0
        SELECTED_SUBSET_NAME = subset_options[int(select_idx)]

    return SELECTED_SUBSET_NAME


def check_prompt_number(prompt_names, dataset):
    if len(prompt_names) > 1:
        print(f"\nChoose a prompt of {dataset}:\n")
        for idx, option in enumerate(prompt_names):
            print(str(idx) + '. ' + option)
        # select_idx = input("\nSelect the index: ")
        select_idx = 0
        SELECTED_PROMPT_NAME = prompt_names[int(select_idx)]
        return SELECTED_PROMPT_NAME
    else:
        return None

    
def load_data(data_name):
    SELECTED_SUBSET_NAME = check_subset(data_name)

    if SELECTED_SUBSET_NAME:
        templates = DatasetTemplates(f'{data_name}/{SELECTED_SUBSET_NAME}')
    else:
        templates = DatasetTemplates(data_name)

    prompt_names = templates.all_template_names
    SELECTED_PROMPT_NAME = check_prompt_number(prompt_names, data_name)

    templates = templates[SELECTED_PROMPT_NAME]
    if SELECTED_SUBSET_NAME:
        dataset = load_dataset(data_name, SELECTED_SUBSET_NAME)
    else:
        dataset = load_dataset(data_name)
    return dataset, templates