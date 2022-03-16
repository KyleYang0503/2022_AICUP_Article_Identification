import torch

a = torch.tensor([[[1, 2, 3], [4, 5, 6]],[[4, 5, 6], [7, 8, 9]],[[1, 2, 3], [4, 5, 6]],[[4, 5, 6], [7, 8, 9]]])

print(a)
print(a.shape)

# b = a.view(a.size(0) * a.size(1), a.size(-1))
# print(b)
# print(b.shape)


b = torch.repeat_interleave(a, 2, dim=0)
print(b)
print(b.shape)