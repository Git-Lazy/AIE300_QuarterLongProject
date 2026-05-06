import torch

# --- Tensor creation ---
from_list = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print("Tensor from list:")
print(from_list)

random_tensor = torch.randn(3, 3)
print("\nRandom tensor (3x3):")
print(random_tensor)

zeros = torch.zeros(2, 4)
ones = torch.ones(2, 4)
print("\nZeros tensor:", zeros)
print("Ones tensor: ", ones)

# --- Basic operations ---
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("\nAddition:", a + b)
print("Element-wise multiply:", a * b)
print("Dot product:", torch.dot(a, b))

A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
print("\nMatrix multiplication A @ B:")
print(torch.matmul(A, B))

# --- Autograd ---
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1  # y = x^2 + 2x + 1  =>  dy/dx = 2x + 2
y.backward()
print(f"\nAutograd example:")
print(f"  x = {x.item()}, y = {y.item()}, dy/dx = {x.grad.item()}")  # dy/dx at x=3 => 8.0

# Multi-variable example
u = torch.tensor(2.0, requires_grad=True)
v = torch.tensor(3.0, requires_grad=True)
z = u ** 3 + v ** 2  # dz/du = 3u^2, dz/dv = 2v
z.backward()
print(f"\n  u={u.item()}, v={v.item()}, z=u^3+v^2={z.item()}")
print(f"  dz/du = {u.grad.item()} (expected {3 * u.item()**2})")
print(f"  dz/dv = {v.grad.item()} (expected {2 * v.item()})")
