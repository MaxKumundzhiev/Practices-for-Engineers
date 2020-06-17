# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


# In torch everything is operating with Tensors.

import torch

## torch.empty()

# Create an empty tensor with size 1D and 3 elements (1 rows by 3 columns)
x = torch.empty(3)
print(x, x.shape)

# Create an empty tensor with size 2D and 3 elements (2 rows by 3 columns)
x = torch.empty(2, 3)
print(x, x.shape)

# Create an empty tensor with size 3D and 3 elements (3 rows by 3 columns)
x = torch.empty(3, 3)
print(x, x.shape)

## torch.rand()
# Create a tensor with random values
x = torch.rand(2, 2)
print(x, x.shape)

## torch.zero()
# Create a tensor with zero values
x = torch.zeros(2, 2)
print(x, x.shape)

## torch.ones()
# Create a tensor with ones values and defined dtype
x = torch.ones(2, 2, dtype=float)
print(x, x.shape)

## torch.tensor()
# Create a tensor with defined by us values
x = torch.tensor([2, 2], dtype=float)

y = torch.tensor([[1, 1],
                  [2, 2]])
print(x, x.shape)
print(y, y.shape)


# Tensors operations.
## Addition and inplace addition torch.add(x, y) / x.add_(y)
## Subtraction and inplace substraction torch.sub(x, y) / x.sub_(y)
## Multiplication and inplace multiplication torch.mul(x, y) / x.mul_(y)
## Devision and inplace division torch.div(x, y) / x.div_(y)

print('Addition Sample \n')
## Addition and inplace addition
x = torch.rand(2, 2, dtype=float)
y = torch.rand(2, 2, dtype=float)
print(x, '\n', y)

# Addition
z = x + y
print(z)

z = torch.add(x, y)
print(z)

# Inplace addition
y.add_(x)
print(y)

# Tensors slicing. tensor[rows, columns]
x = torch.rand(5, 5)
print(x)

## Look on all column values for row 3
print(x[3, :])

## Look on first 3 column values for row 1
print(x[1, :3])

# Tensors reshaping. tensor.view()
x = torch.rand(3, 3)
print(f'Initial random tensor. Shape {x.shape} \n {x}')

print(f'Reshaped random tensor. Shape {x.view(9).shape} \n {x.view(9)}')
print(f'Reshaped random tensor. Shape {x.view(-1, 3).shape} \n {x.view(-1, 3)}')





