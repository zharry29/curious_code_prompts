premise = "{premise}"
hypothesis = "{hypothesis} True, False, or Neither?"
answer = nli_model.forward(premise, hypothesis) 
assert answer ==$ "{label}"