import neuralcoref

class CoreferenceResolution(): 
    '''function to replace the _ in the above sentence with the correct option. '''
    def __init__(self):
        self.model = neuralcoref()
    
    def forward(self, sentence, options):
        answer = self.model(sentence, options)['text'] 
        return answer

coref_res = CoreferenceResolution()

