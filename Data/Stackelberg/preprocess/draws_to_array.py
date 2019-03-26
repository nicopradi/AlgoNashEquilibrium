import numpy as np
np.set_printoptions(threshold=np.nan)

list = []

with open('xi_n50r50.txt', 'r') as f:
    for line in f:
        list.extend(line.split())

list = [float(x) for x in list]
print(len(list))

xi = np.array((3, 50, 50))
xi = np.reshape(list, (3, 50, 50))

print(repr(xi))
