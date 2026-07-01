import torch
import torch.nn as nn
import onnx

model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
dummy = torch.randn(1, 64)
torch.onnx.export(model, dummy, "toy.onnx", opset_version=17)

m = onnx.load("toy.onnx")
for init in m.graph.initializer:                  # the weights
    print(f"Weight: {init.name} {list(init.dims)}")
for node in m.graph.node:                         # the steps
    print(f"Op: {node.op_type} | in: {list(node.input)} | out: {list(node.output)}")