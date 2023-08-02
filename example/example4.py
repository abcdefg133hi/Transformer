import random
import torch
import torch.nn as nn
import torch.nn.functional as Fun
import numpy as np

from transformer import *

class Measure(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        word_length,
        src_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device='cuda',
        max_length=500
        ):
        super(Measure, self).__init__()
        self.problem = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )
        self.answer = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )
        self.word_length = word_length
        self.feed_forward = nn.Sequential(
            nn.Linear(self.word_length*(embed_size+embed_size), 2*forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(2*forward_expansion*embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, 2)
        )
        self.src_pad_idx = src_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #(N,1,1,src_len)
        return src_mask.to(self.device)
    def forward(self, pro, ans):
        pro_mask = self.make_src_mask(pro)
        ans_mask = self.make_src_mask(ans)
        pro_src = self.problem(pro, pro_mask)
        ans_src = self.answer(ans, ans_mask)
        #ans_src = ans_src.unsqueeze(1)
        x = torch.cat((pro_src, ans_src), dim=1)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        out = self.feed_forward(x)
        return out

#Config
pre_problems = [["Where are you from 0 0 0 0 0"], ["How olds are you 0 0 0 0 0"], 
["Where do you come from 0 0 0 0"], ["What is your age 0 0 0 0 0"],
["How are you 0 0 0 0 0 0"], ["Where are you from 0 0 0 0 0"],
["How olds are you 0 0 0 0 0"], ["Where are you from 0 0 0 0 0"]]
pre_answers = [["I am from Taiwan 0 0 0 0 0"], ["I am from China 0 0 0 0 0"],
["I do from Taiwan 0 0 0 0 0"], ["I am 13 0 0 0 0 0 0"],
["I am 9 0 0 0 0 0 0"], ["I from Taiwan 0 0 0 0 0 0"],
["20 0 0 0 0 0 0 0 0"], ["Taiwan 0 0 0 0 0 0 0 0"]]
scores = [[0.9, 0.9], [0.1, 0.1],
[0.8, 0.3], [0.8, 0.6],
[0.75, 0.6], [0.75, 0.2],
[0.70, 0.50], [0.70, 0.50]]
problems = []
answers = []
for problem in pre_problems:
    sentence = problem[0].split(" ")
    problems.append(sentence)
for answer in pre_answers:
    sentence = answer[0].split(" ")
    answers.append(sentence)
from sklearn.preprocessing import LabelEncoder
all_words = sum(problems,[])+sum(answers,[])

label_encoder = LabelEncoder()
label_encoder.fit(all_words)
for i in range(8):
    problems[i] = label_encoder.transform(problems[i]).tolist()
    answers[i] = label_encoder.transform(answers[i]).tolist()

print(problems)

src_vocab_size = 40
num_layers = 3
src_pad_idx = 0
word_length = 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 50
dropout = 0

#Example



problems = torch.tensor(problems).to(device)
answers = torch.tensor(answers).to(device)
scores = torch.tensor(scores).to(device)
#problems = torch.tensor([[1,5,6,4,3,9,5,2,0], [1,8,7,3,4,5,6,7,2]]).to(device)
#answers = torch.tensor([[1,7,4,2,3,5,9,2,0],[1,5,6,1,2,4,7,6,2]]).to(device)
#scores = torch.tensor([[0.2, 0.3], [0.1, 0.2]]).to(device)
model = Measure(src_vocab_size,word_length,src_pad_idx,num_layers=num_layers,device=device, dropout=dropout)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for i in range(epochs):
    out = model(problems, answers)
    loss = loss_fn(out, scores)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Loss of training "+str(i)+": ", loss.item())

print(out)



