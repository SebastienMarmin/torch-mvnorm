a = (1,2,3)
b = [4,1,1,3]
import numpy as np
from itertools import zip_longest
def broadcast_shape(a,b):
    res = reversed(tuple(i if j==1 else (j if i==1 else (i if i==j else -1)) for i,j in zip_longest(reversed(a),reversed(b),fillvalue=1)))
    return list(res)

print(broadcast_shape(a,b))