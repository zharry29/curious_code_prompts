from datasets import load_dataset
import random
random.seed(29)

dataset = load_dataset("squad_v2", cache_dir="/nlp/data/huggingface_cache")
example_indices = random.sample(range(len(dataset['validation'])), 1000)
with open("sampled_1000_indices.txt","w+") as f:
    f.write("\n".join(map(str, example_indices)))