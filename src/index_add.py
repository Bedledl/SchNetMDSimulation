
import torch

x = torch.ones(3, 5)
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])

reference_res = x.index_add_(1, index, t)


def my_index_add(x: torch.Tensor, dim: int, index: torch.Tensor, source: torch.Tensor):
    if len(index) > len(source):
        raise ValueError("Index tensor must not be longer than source tensor.")

    if dim > 1:
        raise NotImplementedError

    if dim == 0:
        for i in range(len(index)):
            x[index[i]] += source[i]
    elif dim == 1:
        for i in range(len(index)):
            x[:, index[i]] += source[i]

    return x

print(my_index_add(x, 1, index, t))
print(reference_res)

