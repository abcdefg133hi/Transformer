import torch
import torch.nn as nn
import torch.nn.functional as Fun
import numpy as np

from transformer import *

#[[sentence, scores]], scores: 0 or 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
problems = [
    ["I am a doctor.", 1],
    ["He is a policeman.", 1],
    ["She is a server.", 1],
    ["Here is an apple.", 1],
    ["He are a nurse.", 0],
    ["They is a doctor.", 0],
    ["Here is a apple.", 0],
    ["They are a policeman.", 0],
    ["They are police officers.", 1],
    ["She are a server.", 0] ]

sentences = []
scores = []
all_words = []
epochs = 20
for pair in problems:
    sentence = pair[0].split(" ")
    sentences.append(sentence)
    scores.append([pair[1]])

#Padding(WIP)

from sklearn.preprocessing import LabelEncoder
all_words = sum(sentences,[])
label_encoder = LabelEncoder()
label_encoder.fit(all_words)

for i in range(len(sentences)):
    sentences[i] = label_encoder.transform(sentences[i]).tolist()
    
x = torch.tensor(sentences).to(device)
trg = torch.tensor(scores).to(device)
src_pad_idx = 0
trg_pad_idx = 0
src_vocab_size = 17
trg_vocab_size = 2
trg_label = Fun.one_hot(trg, num_classes=trg_vocab_size)
trg_label = trg_label*1.  #Long->Float
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for i in range(epochs):
    out = model(x, trg)
    loss = loss_fn(out, trg_label)
    loss.backward()
    optimizer.step()
    
#Total Loss 0/1 Error
max_, argmax_ = torch.max(out, dim=2)
numpy_trg = trg.numpy()
numpy_train = argmax_.numpy()
loss = 0.
for sentence, train_sentence in zip(numpy_trg, numpy_train):
    loss += np.abs((sentence-train_sentence)).sum()/len(sentence)
loss /= len(numpy_trg)

print("Training Loss:", loss)
