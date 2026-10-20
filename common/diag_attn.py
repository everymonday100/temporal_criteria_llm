import sys, torch
sys.path.insert(0, '.')
from common.model import load_model

tok, m = load_model()
inp = tok('What is 2+2?', return_tensors='pt').to('cuda:0')
out = m(inp['input_ids'], output_attentions=True)
print('n_layers in attentions:', len(out.attentions))
a0 = out.attentions[0]
print('shape:', a0.shape, 'dtype:', a0.dtype)
print('has nan:', torch.isnan(a0).any().item())
print('has inf:', torch.isinf(a0).any().item())
pos = a0[a0 > 0]
print('min pos:', pos.min().item() if pos.numel() else 'n/a')
print('max:', a0.max().item())
print('num zeros:', (a0 == 0).sum().item(), 'of', a0.numel())