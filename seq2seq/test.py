import torch
import torch.nn as nn
from tqdm import tqdm
a = torch.tensor([[[1,2,3,4,5],[4,4,4,44,4]]])
b = torch.tensor([[[123,5,1,2,3],[8,5,2,3,1]]])

c = torch.cat((a,b),dim=0)
'''
print(c)

for i,(srg,trg) in tqdm(enumerate(c)):
    print(i)
    print('srg',srg)
    print('trg',trg)
    print('\n')
'''

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.tensor([1,2,3], dtype=torch.long).random_(5)
output = loss(input, target)
print(output)
output.backward()

