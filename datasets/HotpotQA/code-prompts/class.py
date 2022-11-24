question = "{question}"
contexts = [
    {supporting-documents}
]
answer = qa_model.forward(question, contexts) 
assert answer ==$ {answer}

