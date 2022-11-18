import re
import time
import utils
import random
import openai
import argparse
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score


class OpenPI():
    def __init__(self, metadata):
        self.metadata = metadata
    
    def build_text_prompt(self):
        train_dict = self.metadata['train'] 
        key_candidates = list(train_dict.keys())
        prompt_keys = random.sample(key_candidates, NUM_EXAMPLES_IN_PROMPT)
        sub_dict = [train_dict[k] for k in prompt_keys]
        prompt = ''
        for event_dict in sub_dict:
            goal = event_dict['goal']
            prompt += f'Goal: {goal}\n\n'
            steps = event_dict['steps']
            for i, step in enumerate(steps):
                step_state_changes = step['state_changes']
                step_desc = step['description']
                prompt += f'Step {i+1}: {step_desc}\n\n'
                prompt += f'Entity status changes:\n'
                if step_state_changes:
                    for state_change in step_state_changes:
                        entity, attr, pre, post = state_change
                        if '|' in entity:
                            entity = utils.choose_openpi_options(entity)
                        if '|' in attr:
                            attr = utils.choose_openpi_options(attr)
                        if '|' in pre:
                            pre = utils.choose_openpi_options(pre)
                        if '|' in post:
                            post = utils.choose_openpi_options(post)
                        prompt += f'- The {attr} of {entity} is {pre} before and {post} afterwards.\n'
                    prompt += '\n'
                else:
                    prompt += 'None\n'
            prompt += '\n\n'
        return prompt

    def build_code_prompt(self):
        code_prompt = ""
        return code_prompt
    
    def run_llm(self, prompt, model, temperature=0, stop=['\n']):
        model_name = {
            "davinci": "text-davinci-002",
            "curie": "text-curie-001",
            "codex": "codex-davinci-002",
            "ada": "text-ada-001"
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
        val_data = self.metadata['dev']
        if args.prompt == "text":
            prompt = self.build_text_prompt()
        elif args.prompt == "code":
            prompt = self.build_code_prompt()


        preds = []
        golds = []
        for example in tqdm(val_data.values(), position=0, leave=False):
            goal = example['goal']
            cur_prompt = prompt + f'Goal: {goal}\n\n'
            steps = example['steps']
            gold = [[] for _ in range(len(steps))]
            pred = [[] for _ in range(len(steps))]
            for i, step in enumerate(tqdm(steps, position=1, leave=False)):
                step_state_changes = step['state_changes']
                step_desc = step['description']
                cur_prompt += f'Step {i+1}: {step_desc}\n\n'
                cur_prompt += f'Entity status changes:\n'
                llm_pred = self.run_llm(cur_prompt, args.model, stop='\n\n')
                llm_pred = utils.parse_preds(llm_pred)
                for line in llm_pred:
                    print(line)
                    entries = re.match(r'The (.*) is (.*) before and (.*) afterwards.', line)
                    print((entries.group(1), entries.group(2), entries.group(3)))
                    cur_prompt += '- ' + line.strip() + '\n'
                cur_prompt += '\n'
                pred.append(llm_pred)
                
                state_change_lst = []
                if step_state_changes:
                    for state_change in step_state_changes:
                        entity, attr, pre, post = state_change
                        entity, attr, pre, post = entity.split('|'), attr.split('|'), pre.split('|'), post.split('|')
                        for e in entity:
                            for a in attr:
                                for pr in pre:
                                    for po in post:
                                        state_change_lst.append(f'The {a.strip()} of {e.strip()} is {pr.strip} before and {po.strip()} afterwards.')
                
                print(pred)
                print(gold)
                raise SystemExit()
            golds.append(gold)
            preds.append(pred)

        with open(f'pred_{self.idx}.txt', 'w') as f:
            f.writelines([x + '\n' for x in preds])
        with open(f'gold_{self.idx}.txt', 'w') as f:
            f.writelines([x + '\n' for x in golds])
    
    # TODO: How do we evaluate the result of OpenPI? 
    def evaluate(self):
        pass 


parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, help='Either text or code.')
parser.add_argument('--model', type=str, help='Either davinci, curie or codex.')
parser.add_argument('--data_path', type=str, help='Path to the folder that stores parsed OpenPI datasets.')
#parser.add_argument('--dataset', type=str, help='Name of the datasset')
#parser.add_argument('--xxx', action='store_true', help='')
parser.add_argument('--key', type=str, help='The name of the OpenAI API key file.')


if __name__ == '__main__':
    args = parser.parse_args()
    openai.api_key_path = f'../../_private/{args.key}.key'

    data_name = 'anli'
    NUM_EXAMPLES_IN_PROMPT = 2

    meta_data = utils.load_meta_data(args.data_path)

    inference_model = OpenPI(meta_data)
    inference_model.predict()
    # inference_model.evaluate()










