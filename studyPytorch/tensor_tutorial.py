from __future__ import print_function
import torch


x = torch.empty(5,3)
print("Generate uninitialized 5x3 matrix")
print(x)

x = torch.rand(5,3)
print("\nGenerate randomly initialized 5x3 matrix")
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print("\nGenerate 5x3 matrix filled with zero and dtype is long")
print(x)


x = torch.tensor([5.5, 3])
print("\nGenerate tensor from data directly")
print(x)

x = x.new_ones(5, 3, dtype = torch.double)
print("\nMake new tensor filled with one from existing tensor")
print(x)

x= torch.randn_like(x, dtype = torch.float)
print('\nMake new tensor which override dtype with exisiting tensor')
print(x)

print("\nPrint size of matrix")
print(x.size())


y = torch.rand(5,3)
print("\n Add two tensors")
print("\nx: ")
print(x)
print("\ny: ")
print(y)
print("\nx + y: ")
print(x+y)

# print(torch.add(x+y))                 Another Addition Operation

'''                                     Providing an output tensor as argument
result = torch.empty(5,3)
torch.add(x, y, out=result)
print(result)
'''

'''                                     Another Addition: In-place
y.add_(x)
print(y)
'''

print("\n Use Numpy-like indexing : x[:,1]")
print(x[:,1])

print("\nTorch Tensor to Numpy Array")
print("x as numpy")
x = x.numpy()
print(x)

print("\nNumpy Array to Torch Tensor")
print("x as Torch Tensor")
x = torch.from_numpy(x)
print(x)