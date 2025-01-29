import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100,100)
    
    def forward(self,x):
        return self.embed(x)
    
model = Model()

model.train()
for _, mod in model.named_modules():
    if isinstance(mod, torch.nn.Embedding):
        mod.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig
torch.quantization.convert(model)