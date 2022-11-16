from scipy.stats import pearsonr

with open("gold.txt") as f:
    golds = [int(l.strip()) + 1 for l in f.readlines()]

with open("pred.txt") as f:
    preds = [int(l.strip()) for l in f.readlines()]

print("Accuracy", pearsonr(golds, preds)[0])