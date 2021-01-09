import numpy as np


a = [i for i in range(10)]
print(a)
b = [j for j in range(10, 20)]
print(b)
c = zip(a, b, b)
print(c)
d = [i for i in c]
print(d)
e = np.asarray(d)
print(e)
