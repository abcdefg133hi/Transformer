import os
import sys

import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
import torch.nn.functional as Fun

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

# Import Data
summaries_train_df=pd.read_csv("input/summaries_train.csv")
summaries_test_df=pd.read_csv("input/summaries_test.csv")
prompts_train_df=pd.read_csv("input/prompts_train.csv")
prompts_test_df=pd.read_csv("input/prompts_test.csv")
sample_submission_df=pd.read_csv("input/sample_submission.csv")


problems_train = prompts_train_df.values.tolist()
train_set = summaries_train_df[['prompt_id','text','content','wording']].values.tolist()
train_set = train_set[:100]   #Now, just use 100 training data, 0: prompt_id, 1: text, 2: content, 3:wording, 4: problems
val_set = train_set[-20:]   #Now, just use 20 training data, 0: prompt_id, 1: text, 2: content, 3:wording, 4: problems
#Append problem in train_set
problems = []
answers = []
scores = []
padding_length = 300
for sheet in train_set:
    for problem in problems_train:
        if sheet[0] == problem[0]:
            sheet.append(problem[1])
            break
    sentence = sheet[4].split(" ")
    if len(sentence) > padding_length:
        print("length", len(sentence))
        print("Senetence",sentence)
        print("Length is bigger then padding_length!!")
        sys.exit(1)
    for _ in range(padding_length-len(sentence)):
        sentence.append("0")
    problems.append(sentence)
    sentence = sheet[1].split(" ")
    if len(sentence) > padding_length:
        print("length", len(sentence))
        print("Sentence", sentence)
        print("Length is bigger then padding_length!")
        sys.exit(1)
    for _ in range(padding_length-len(sentence)):
        sentence.append("0")
    answers.append(sentence)
    scores.append(sheet[2:4])

val_problems = []
val_answers = []
val_scores = []
for sheet in val_set:
    for problem in problems_train:
        if sheet[0] == problem[0]:
            sheet.append(problem[1])
            break
    sentence = sheet[4].split(" ")
    if len(sentence) > padding_length:
        print("length", len(sentence))
        print("Senetence",sentence)
        print("Length is bigger then padding_length!!")
        sys.exit(1)
    for _ in range(padding_length-len(sentence)):
        sentence.append("0")
    val_problems.append(sentence)
    sentence = sheet[1].split(" ")
    if len(sentence) > padding_length:
        print("length", len(sentence))
        print("Sentence", sentence)
        print("Length is bigger then padding_length!")
        sys.exit(1)
    for _ in range(padding_length-len(sentence)):
        sentence.append("0")
    val_answers.append(sentence)
    val_scores.append(sheet[2:4])


from sklearn.preprocessing import LabelEncoder
all_words = sum(problems,[])+sum(answers,[])+sum(val_problems, [])+sum(val_answers, [])

label_encoder = LabelEncoder()
temp = label_encoder.fit_transform(all_words)
max_encoded_value = np.max(temp)
for i in range(len(train_set)):
    problems[i] = label_encoder.transform(problems[i]).tolist()
    answers[i] = label_encoder.transform(answers[i]).tolist()
    
for i in range(len(val_set)):
    val_problems[i] = label_encoder.transform(val_problems[i]).tolist()
    val_answers[i] = label_encoder.transform(val_answers[i]).tolist()

print("Src_vocab_size", np.max(temp)+1)

src_vocab_size = max_encoded_value+1

# Config
num_layers = 3
src_pad_idx = 0
embed_size = 256
word_length = padding_length
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 2
dropout = 0
print("Now using:", device)

problems = torch.tensor(problems).to(device)
answers = torch.tensor(answers).to(device)
scores = torch.tensor(scores).to(device)
val_problems = torch.tensor(problems).to(device)
val_answers = torch.tensor(answers).to(device)
val_scores = torch.tensor(scores).to(device)
#problems = torch.tensor([[1,5,6,4,3,9,5,2,0], [1,8,7,3,4,5,6,7,2]]).to(device)
#answers = torch.tensor([[1,7,4,2,3,5,9,2,0],[1,5,6,1,2,4,7,6,2]]).to(device)
#scores = torch.tensor([[0.2, 0.3], [0.1, 0.2]]).to(device)
model = Measure(src_vocab_size,word_length,src_pad_idx,embed_size=embed_size,
                num_layers=num_layers,device=device, dropout=dropout).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
for i in range(epochs):
    out = model(problems, answers)
    loss = loss_fn(out, scores)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print("Loss of training "+str(i)+": ", loss.item())
    with torch.no_grad():
        out = model(val_problems, val_answers)
        val_loss = loss_fn(out, val_scores)
        print("Loss of valid "+str(i)+": ", val_loss.item())