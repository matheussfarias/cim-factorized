import torch

K = 316
N = 100
B = torch.rand(K,N)

l = 50

epochs = 25

B_prime = torch.rand(l,N)

B0 = B
error = []
for i in range(epochs):
    B_pos = torch.matmul(B, torch.linalg.pinv(B_prime))
    B_pos = torch.abs(B_pos)
    B = torch.matmul(B_pos, B_prime)
    B_prime = torch.linalg.lstsq(B_pos, B)
    error.append[torch.norm(B-B0)]

print(error)