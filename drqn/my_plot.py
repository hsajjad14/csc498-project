from matplotlib import pyplot as plt
import pickle
import numpy as np

with open('rewards.pickle', 'rb') as handle:
    b = pickle.load(handle)

print(b)

plt.plot(np.arange(len(b)), b)
plt.savefig('myRewards')