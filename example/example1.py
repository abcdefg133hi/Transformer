import torch
import torch.nn as nn
import torch.nn.functional as Fun
import numpy as np

from transformer import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10
x = torch.tensor([[1,5,6,4,3,9,5,2,0], [1,8,7,3,4,5,6,7,2]]).to(device)
trg = torch.tensor([[1,7,4,3,5,9,2,0],[1,5,6,2,4,7,6,2]]).to(device)
src_pad_idx = 0
trg_pad_idx = 0
src_vocab_size = 10
trg_vocab_size = 10
trg_label = Fun.one_hot(trg[:,:-1], num_classes=trg_vocab_size)
trg_label = trg_label*1.  #Long->Float
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for i in range(epochs):
    out = model(x, trg[:, :-1])
    loss = loss_fn(out, trg_label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    max_, argmax_ = torch.max(out, dim=2)
    numpy_trg = trg[:,:-1].numpy()
    numpy_train = argmax_.numpy()
    loss = 0.
    for sentence, train_sentence in zip(numpy_trg, numpy_train):
        loss += np.abs((sentence-train_sentence)).sum()/len(sentence)
    loss /= len(numpy_trg)
    print("Training Loss:"+str(i), loss)

#Total Loss 0/1 Error
max_, argmax_ = torch.max(out, dim=2)
numpy_trg = trg[:, :-1].numpy()
numpy_train = argmax_.numpy()
loss = 0.
for sentence, train_sentence in zip(numpy_trg, numpy_train):
    loss += np.abs((sentence-train_sentence)).sum()/len(sentence)
loss /= len(numpy_trg)
print("Training Loss:", loss)

#Validation
x_val = torch.tensor([[1,6,3,2,3,3,3,2,0], [1,9,7,2,4,5,2,7,2]]).to(device)
trg_val = torch.tensor([[1,8,1,3,5,2,2,0],[1,5,6,6,6,7,1,2]]).to(device)
out_val = model(x_val, trg_val[:,:-1])
max_val, argmax_val = torch.max(out_val, dim=2)
numpy_trg_val = trg_val[:, :-1].numpy()
numpy_val = argmax_val.numpy()
loss = 0.
for sentence, train_sentence in zip(numpy_trg_val, numpy_val):
    loss += np.abs((sentence-train_sentence)).sum()/len(sentence)
loss /= len(numpy_trg_val)
print("Validation Loss:", loss)