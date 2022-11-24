import neuralqa

class QuestionAnswering(): 
    '''function to answer the question using information from the list of contexts. '''
    def __init__(self):
        self.model = neuralqa()
    
    def forward(self, question, contexts):
        answer = self.model(question, contexts)['answer'] 
        return answer

qa_model = QuestionAnswering()

