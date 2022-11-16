import random
import pickle
with open("sampled_1000_indices.pkl","wb") as f:
    pickle.dump(random.sample(range(50000),1000), f)