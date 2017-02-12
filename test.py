import numpy as np

a = np.zeros(shape = (32,32,3))

print(a.shape)

b = np.transpose(a,(2,0,1))

print(b.shape)
print(a.shape)
