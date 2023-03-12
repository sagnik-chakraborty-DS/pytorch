import torch
import numpy as np


x = torch.arange(11)
print(x)
print(x.dtype)
print(x.device)
print(x.shape)
print(x.requires_grad)
print(x.bool())
"""
torch.int64
cpu
torch.Size([11])
False
tensor([False,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True])
"""



"""
int16 int32 int64
float16 float32 float64
"""

x = x.int()
print(x)
print(x.dtype)
"""
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10], dtype=torch.int32)
torch.int32
"""
x = x.short()
print(x)
print(x.dtype)
"""
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10], dtype=torch.int16)
torch.int16
"""

x = x.long()
print(x)
print(x.dtype)
"""
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
torch.int64
"""
x = x.half()
print(x)
print(x.dtype)
"""
tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.],
       dtype=torch.float16)
torch.float16
"""
x = x.float()
print(x)
print(x.dtype)
"""
tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
torch.float32
"""

x = x.double()
print(x)
print(x.dtype)
"""
tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.],
       dtype=torch.float64)
torch.float64
"""



"""

TENSOR TO NUMPY ARRAY
"""


np_array = np.array([[1,2,3], [1,2,3]])

print(np_array)



tensor = torch.from_numpy(np_array)
print(tensor)

print(tensor.numpy())

"""
[[1 2 3]
 [1 2 3]]
tensor([[1, 2, 3],
        [1, 2, 3]], dtype=torch.int32)
[[1 2 3]
 [1 2 3]]
"""