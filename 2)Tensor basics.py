import torch




"""
What is tensor?
A kind of data structure => multidimensional arrays or matrices
With tensors you enocode all your parameters.
Type Conversions
Conversions from one datatype to another.
Conversions from torch tensors to numpy arrays and vice versa.

"""
device = "cuda" if torch.cuda.is_available() else "cpu"

print(device) # cuda

basic_tensor = torch.tensor([[1,2,3], [4,5,6]])
print(basic_tensor)
print(basic_tensor.dtype)
print(basic_tensor.device)
print(basic_tensor.shape)
print(basic_tensor.requires_grad) # Tensor. requires_grad. Is True if gradients need to be computed for this Tensor, False otherwise. The fact that gradients need to be computed for a Tensor do not mean that the grad attribute will be populated

"""
cuda
tensor([[1, 2, 3],
        [4, 5, 6]])
torch.int64
cpu
torch.Size([2, 3])
False

"""


basic_tensor = torch.tensor([[1,2,3.5], [4,5,6]])
print(basic_tensor)
print(basic_tensor.dtype)
print(basic_tensor.device)
print(basic_tensor.shape)
print(basic_tensor.requires_grad)

"""
tensor([[1.0000, 2.0000, 3.5000],
        [4.0000, 5.0000, 6.0000]])
torch.float32
cpu
torch.Size([2, 3])
False
"""

"""
basic_tensor = torch.tensor([["s","s"], ["a","b"]])
print(basic_tensor)
print(basic_tensor.dtype)
print(basic_tensor.device)
print(basic_tensor.shape)
print(basic_tensor.requires_grad)

ERROR:

Traceback (most recent call last):
  File "D:\pytorch\2)Tensor and Operation.py", line 55, in <module>
    basic_tensor = torch.tensor([["s","s"], ["a","b"]])
ValueError: too many dimensions 'str'
"""


# we can control parmetr for tensors
# below tnsor using gpu if we mention CPU in device parameter ther it will use CPU

device = "cuda" if torch.cuda.is_available() else "cpu"
basic_tensor = torch.tensor([[1,2,3],[11,22,33]],
                     dtype=torch.float,
                     device=device,
                     requires_grad=True)

print(basic_tensor)
print(basic_tensor.dtype)
print(basic_tensor.device)
print(basic_tensor.shape)
print(basic_tensor.requires_grad)

"""
tensor([[ 1.,  2.,  3.],
        [11., 22., 33.]], device='cuda:0', requires_grad=True)
torch.float32
cuda:0
torch.Size([2, 3])
True
"""

t