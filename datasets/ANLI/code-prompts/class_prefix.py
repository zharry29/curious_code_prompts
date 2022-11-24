import neuralnli

class NaturalLanguageInference(): 
    '''function to answer the natural language inference task given premise and hypothesis.'''
    def __init__(self):
        self.model = neuralnli()
    
    def forward(self, premise, hypothesis):
        answer = self.model(premise, hypothesis)['answer'] 
        return answer

nli_model = NaturalLanguageInference()

