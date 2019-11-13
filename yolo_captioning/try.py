# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

word_to_ix = {'hello': 0, 'world': 1}
t=torch.Tensor([-0.5736, -3.6566,  3.0850,  3.4097,  2.6072])
embeds = nn.Embedding(2, 5)
embeds.weight[0,:]=t
hello_idx = torch.LongTensor([word_to_ix['hello']])
hello_idx = Variable(hello_idx)
hello_embed = embeds(hello_idx)
print(hello_embed)