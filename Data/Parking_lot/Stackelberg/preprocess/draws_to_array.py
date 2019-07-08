import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

list = []

with open('beta_fee_n5r100.txt', 'r') as f:
    for line in f:
        list.extend(line.split())

list = [float(x) for x in list]
print(len(list))

xi = np.array((5, 100))
xi = np.reshape(list, (5, 100))

print(repr(xi))
