import torch
from model.model import LSTN
import thop
import time


model = LSTN()
inputs = torch.randn(1, 3, 224, 224)
print(thop.profile(model, (inputs,)))
# time_s = time.time()
# _ = model(inputs, stn_mode=True)
# print(time.time() - time_s)
