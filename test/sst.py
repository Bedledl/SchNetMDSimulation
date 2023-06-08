#!/usr/bin/env python3

# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import poptorch

# This simple example demonstrates compiling a model to add
# two tensors together using the IPU.
from torch_geometric.nn import Sequential


class SimpleAdder(nn.Module):
    def __init__(self):
        super(SimpleAdder, self).__init__()
        self.il = torch.nn.Linear(in_features=4, out_features=2)

        self.ol = torch.nn.Linear(in_features=2, out_features=1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x, gt):
        print(x.is_leaf)
        y = self.il(x)
        y[len(gt)-2] = 0
        x = self.il(x)
        y = self.ol(y)
        print(x.is_leaf)
        print(y.is_leaf)
        loss = self.loss_fn(y, gt)
        print("after loss")
        go = torch.ones_like(loss)
        g = torch.autograd.grad(inputs=x, outputs=loss, grad_outputs=go, create_graph=False, allow_unused=True)
        print("before return")
        return x, g


model = SimpleAdder()
inference_model = poptorch.inferenceModel(model)

#t1 = torch.tensor([1., 6., 3.])
#t2 = torch.tensor([5, 8, 4])


t1 = torch.randn(3, 4, requires_grad=True)
t2 = torch.randn(3, 1)
print(t1)
print(t2)
#print(model(t1, t2))
e = inference_model(t1, t2)
print(e)
print("Success")


