#!/usr/bin/env python3

# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import datetime
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import torch
import torch.nn as nn
import poptorch

# This simple example demonstrates compiling a model to add
# two tensors together using the IPU.
from torch_geometric.nn import Sequential


class SimpleAdder(nn.Module):
    def __init__(self):
        super(SimpleAdder, self).__init__()

    def forward(self, x, y):
        z = torch.ones(len(x))
        return x + z + y


model = SimpleAdder()
opts = poptorch.Options()
#opts.replicationFactor(1)


def iterate_adder(i_model, start_x, iter_y):
    x = start_x
    for y in iter_y:
        x = i_model(x, y)
    return x


max_pid = 3

def inference(pid: int):
    opts.Distributed.configureProcessId(pid, max_pid)

    inference_model = poptorch.inferenceModel(model, opts)

    start_x1 = torch.randn(100)
    start_x2 = torch.randn(100)

    iter_y1 = [torch.randn(100) for _ in range(1000)]
    iter_y2 = [torch.randn(100) for _ in range(1000)]

    a = datetime.datetime.now()
    for i in range(10):
        iterate_adder(inference_model, start_x1, iter_y1)
        iterate_adder(inference_model, start_x2, iter_y2)

    b = datetime.datetime.now()

    print(f"Nacheinanderausführung: {b - a}")
    return f"{a} - {b} also {b - a}"


results = []

with ProcessPoolExecutor(max_workers=max_pid) as executor:
    for result in executor.map(inference, range(max_pid)):
        print(result)
