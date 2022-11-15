from sklearn.metrics import accuracy_score

with open("gold.txt") as f:
    golds = [int(l.strip()) + 1 for l in f.readlines()]

with open("pred.txt") as f:
    preds = [int(l.strip()) for l in f.readlines()]

print("Accuracy", accuracy_score(golds, preds))