from __future__ import print_function
import torch


def initial_tutorial():
    x = torch.empty(5, 3)
    print(x)

    x = torch.zeros(5, 3, dtype=torch.long)
    print(x)

    x = torch.tensor([5.5, 3])
    print(x)

    x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
    print(x)

    x = torch.randn_like(x, dtype=torch.float)  # override dtype!
    print(x)  # result has the same

    print(x.size())

    y = torch.rand(5, 3)
    print(x + y)
    print(torch.add(x, y))

    result = torch.empty(5, 3)
    torch.add(x, y, out=result)
    print(result)

    # adds x to y
    y.add_(x)
    print(y)

    # Any operation that mutates a tensor in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), will change x
    print(x[:, 1])

    x = torch.randn(4, 4)
    y = x.view(16)
    print("y: ", y)

    z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
    print("z: ", z)
    print(x.size(), y.size(), z.size())

    a = torch.ones(5)
    print(a)

    b = a.numpy()
    print(b)

    # let us run this cell only if CUDA is available
    # We will use ``torch.device`` objects to move tensors in and out of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")  # a CUDA device object
        y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
        x = x.to(device)  # or just use strings ``.to("cuda")``
        z = x + y
        print(z)
        print(z.to("cpu", torch.double))  # ``.to`` can also change dtype together!



def neural_net_tutorial():
    x = torch.ones(2, 2, requires_grad=True)
    print(x)

    y = x + 2
    print(y)

    print(y.grad_fn)

    z = y * y * 3
    out = z.mean()

    print(z, out)

    a = torch.randn(2, 2)
    a = ((a * 3) / (a - 1))
    print(a.requires_grad)
    a.requires_grad_(True)
    print(a.requires_grad)
    b = (a * a).sum()
    print(b.grad_fn)

if __name__ == "__main__":
    neural_net_tutorial()
   # initial_tutorial()
