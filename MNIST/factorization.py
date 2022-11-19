import torch

K = 316
N = 100
B = torch.rand(K,N)

l = 50

epochs = 25

B_prime = torch.rand(l,N)

B0 = B
print(B.shape)
print(B_prime.shape)
print(torch.linalg.pinv(B_prime).shape)
print(torch.linalg.pinv(B).shape)
print(torch.matmul(B, torch.linalg.pinv(B_prime)).shape)
error = []
for i in range(epochs):
    B_pos = torch.matmul(B, torch.linalg.pinv(B_prime))
    B_pos = torch.abs(B_pos)
    B = torch.matmul(B_pos, B_prime)
    B_prime,_,_,_ = torch.linalg.lstsq(B_pos, B)
    error.append(torch.norm(B-B0))

print(error)