import torch.nn.functional as F
import torch
def preprocess(x):
    outputs = []
    for i in range(4):
        outputs.append(F.pad(x[..., i], (i + 1, 3 - i), "constant", 2048))
    return torch.stack(outputs, -1)


x = preprocess(torch.zeros([2, 3, 4]))
x = x.transpose(1, 2)
print(x[0, :4, :4])