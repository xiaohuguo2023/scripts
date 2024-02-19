import torch
import torch.nn.functional as F

def random_tensors(shape, num=1, requires_grad=False):
  tensors = [torch.randn(shape, requires_grad=requires_grad) for i in range(0, num)]
  return tensors[0] if num == 1 else tensors

# Parameters
# -- [hidden_dimension]
bM, br, w = random_tensors([7], num=3, requires_grad=True)
# -- [hidden_dimension x hidden_dimension]
WY, Wh, Wr, Wt = random_tensors([7, 7], num=4, requires_grad=True)

# Single application of attention mechanism 
def attention(Y, ht, rt1):
  # -- [batch_size x hidden_dimension] 
  tmp = torch.einsum("ik,kl->il", [ht, Wh]) + torch.einsum("ik,kl->il", [rt1, Wr])
  Mt = F.tanh(torch.einsum("ijk,kl->ijl", [Y, WY]) + tmp.unsqueeze(1).expand_as(Y) + bM)
  # -- [batch_size x sequence_length]
  at = F.softmax(torch.einsum("ijk,k->ij", [Mt, w]), dim=1) 
  # -- [batch_size x hidden_dimension]
  rt = torch.einsum("ijk,ij->ik", [Y, at]) + F.tanh(torch.einsum("ij,jk->ik", [rt1, Wt]) + br)
  # -- [batch_size x hidden_dimension], [batch_size x sequence_dimension]
  return rt, at

# Sampled dummy inputs
# -- [batch_size x sequence_length x hidden_dimension]
Y = random_tensors([3, 5, 7])
# -- [batch_size x hidden_dimension]
ht, rt1 = random_tensors([3, 7], num=2)

rt, at = attention(Y, ht, rt1)
print(at)  # -- print attention weights
