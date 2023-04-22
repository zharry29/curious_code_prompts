import pickle
import numpy as np


example_4000 = pickle.load(open('num_example_4000.pkl', 'rb'))
example_8000 = pickle.load(open('num_example_8000.pkl', 'rb'))

avg_4000 = np.mean(example_4000)
avg_8000 = np.mean(example_8000)

print(avg_4000, avg_8000)