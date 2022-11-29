class CodexPrompts():
    def __init__(self, main_prompt: list, train: bool):
        self.main_prompt = main_prompt
        self.train = train
    
    def vanilla(self):
        prompt = ''
        options = []
        for entry in self.main_prompt:
            if not entry[0].isalnum():
                options.append(entry.replace('-', '').strip())
            elif entry[:6].lower() == 'answer':
                prompt += f'input3 = {options}\n'
                prompt += f"output = '{entry.replace('Answer:', '').strip()}'\n\n"
                options = []
            elif 'Replace the _' in entry:
                prompt += f"input2 = '{entry.replace(':', '.').strip()}'\n"
            else:
                prompt += f"input1 = '{entry}'\n"
        if not self.train:
            prompt += f'input3 = {options}\n'
            prompt += "answer ="
        return prompt

    def good_var_name(self):
        prompt = ''
        options = []
        for entry in self.main_prompt:
            if not entry[0].isalnum():
                options.append(entry.replace('-', '').strip())
            elif entry[:6].lower() == 'answer':
                prompt += f'options = {options}\n'
                prompt += f"answer = '{entry.replace('Answer:', '').strip()}'\n\n"
                options = []
            elif 'Replace the _' in entry:
                prompt += f"instruction = '{entry.replace(':', '.').strip()}'\n"
            else:
                prompt += f"sentence = '{entry}'\n"
        if not self.train:
            prompt += f'options = {options}\n'
            prompt += "answer ="
        return prompt

    def with_comments(self):
        if self.train:
            prompt = "'''\nThis is a coference resolution task. There will be a '_' in a given sentence and options will be provided. You need to choose from given options and fill in the '_'.\n'''\n\n"
        else:
            prompt = ''
        options = []
        for entry in self.main_prompt:
            if not entry[0].isalnum():
                options.append(entry.replace('-', '').strip())
            elif entry[:6].lower() == 'answer':
                prompt += f'options = {options}  # these are the options that you need to pick and fill in the blank with\n'
                prompt += f"answer = '{entry.replace('Answer:', '').strip()}'\n\n\n"
                options = []
            elif 'Replace the _' in entry:
                prompt += f"instruction = '{entry.replace(':', '.').strip()}'\n"
            else:
                prompt += f"sentence = '{entry}'  # the sentence with a blank '_' to be filled\n"
        if not self.train:
            prompt += f'options = {options}  # these are the options that you need to pick and fill in the blank with\n'
            prompt += "answer ="
        return prompt
    
    def class_obj(self):
        if self.train:
            prompt = f"import neuralcoref\n\ndef coreference_resolution():\n\t'''\n\tfunction to replace the _ in the above sentence with the correct option.\n\t'''\n"
            prompt += f"\tanswer = neuralcoref(sentence, options)['text']\n\treturn answer\n\n"
        else:
            prompt = ''
        options = []
        for entry in self.main_prompt:
            if not entry[0].isalnum():
                options.append(entry.replace('-', '').strip())
            elif entry[:6].lower() == 'answer':
                prompt += f"options = {options}\nanswer = coreference_resolution(sentence, options)\nassert answer == '{entry.replace('Answer:', '').strip()}'\n\n"
                options = []
            elif 'Replace the _' not in entry:
                prompt += f"sentence = '{entry}'\n"
        if not self.train:
            prompt += f"options = {options}\nanswer = coreference_resolution(sentence, options)\nassert answer =="    
        return prompt
                

            
        