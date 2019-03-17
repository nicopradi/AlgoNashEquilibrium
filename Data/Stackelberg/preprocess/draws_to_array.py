import numpy as np
np.set_printoptions(threshold=np.nan)

list = []

with open('xi_n10r200.txt', 'r') as f:
    for line in f:
        list.extend(line.split())

list = [float(x) for x in list]
print(len(list))

xi = np.array((3, 10, 200))
xi = np.reshape(list, (3, 10, 200))

print(repr(xi))
