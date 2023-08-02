"""
WIP
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as Fun
import numpy as np

from transformer import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
subjects = ["I", "You", "He", "She", "It", "We", "They"]
verbs_singular = ["is", "likes", "loves", "admires", "appreciates"]
verbs_plural = ["are", "like", "love", "admire", "appreciate"]
jobs = ["teachers", "engineers", "doctors", "writers", "chef", "artists", "pilots", "scientists", "actors", "musicians"]
alls = subjects+verbs_singular+verbs_plural+jobs+["am"]
sentences = []
val_sentences = []
for _ in range(20):
    subject = random.choice(subjects)
    verb = random.choice(verbs_plural if subject == "They" or subject == "We" or subject == "You" else verbs_singular)
    job = random.choice(jobs)

    if subject == "I":
        if verb == "is":
            verb = "am"
        sentence = f"{subject} {verb} {job}"
    else:
        sentence = f"{subject} {verb} {job}"

    sentences.append([sentence, [random.uniform(0.99,1),random.uniform(0., 0.01)], [1]])

for _ in range(50):
    subject = random.choice(alls)
    verb = random.choice(alls)
    job = random.choice(alls)

    if subject == "I":
        sentence = f"{subject} {verb} {job}"
    else:
        sentence = f"{subject} {verb} {job}"

    sentences.append([sentence, [random.uniform(0.,0.01),random.uniform(0.99, 1.0)], [0]])
random.shuffle(sentences)
print(sentences)
for _ in range(20):
    subject = random.choice(subjects)
    verb = random.choice(verbs_plural if subject == "They" or subject == "We" or subject == "You" else verbs_singular)
    job = random.choice(jobs)

    if subject == "I":
        if verb == "is":
            verb = "am"
        sentence = f"{subject} {verb} {job}"
    else:
        sentence = f"{subject} {verb} {job}"

    val_sentences.append([sentence, [1]])

for _ in range(50):
    subject = random.choice(alls)
    verb = random.choice(alls)
    job = random.choice(alls)

    if subject == "I":
        sentence = f"{subject} {verb} {job}"
    else:
        sentence = f"{subject} {verb} {job}"

    val_sentences.append([sentence, [0]])
random.shuffle(val_sentences)
x = []
x_val = []
probabilities = []
scores = []
val_scores = []
all_words = []
epochs = 500
for pair in sentences:
    sentence = pair[0].split(" ")
    x.append(sentence)
    probabilities.append(pair[1])
    scores.append(pair[2])

for pair in val_sentences:
    sentence = pair[0].split(" ")
    x_val.append(sentence)
    val_scores.append(pair[1])

#Padding(WIP)

from sklearn.preprocessing import LabelEncoder
all_words = alls
label_encoder = LabelEncoder()
label_encoder.fit(all_words)

for i in range(len(sentences)):
    x[i] = label_encoder.transform(x[i]).tolist()
    x_val[i] = label_encoder.transform(x_val[i]).tolist()

x = torch.tensor(x).to(device)
x_val = torch.tensor(x_val).to(device)
trg = torch.tensor(scores).to(device)
trg_val = torch.tensor(val_scores).to(device)
trg_label = torch.tensor(probabilities).to(device).reshape(70,1,2)
src_pad_idx = 0
trg_pad_idx = 0
src_vocab_size = len(alls)

trg_vocab_size = 2
"""
trg_label = Fun.one_hot(trg, num_classes=trg_vocab_size)
trg_label = trg_label*1.  #Long->Float
"""
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
for i in range(epochs):
    out = model(x, trg)
    loss = loss_fn(out, trg_label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    max_, argmax_ = torch.max(out, dim=2)
    numpy_trg = trg.numpy()
    numpy_train = argmax_.numpy()
    loss = 0.
    for sentence, train_sentence in zip(numpy_trg, numpy_train):
        loss += np.abs((sentence-train_sentence)).sum()/len(sentence)
    loss /= len(numpy_trg)
    print("Training Loss:"+str(i), loss)
    
#Total Loss 0/1 Error
max_, argmax_ = torch.max(out, dim=2)
numpy_trg = trg.numpy()
numpy_train = argmax_.numpy()
loss = 0.
for sentence, train_sentence in zip(numpy_trg, numpy_train):
    loss += np.abs((sentence-train_sentence)).sum()/len(sentence)
loss /= len(numpy_trg)

print("Training Loss:", loss)
out = model(x_val, trg_val)
for i in range(10):
    print(label_encoder.inverse_transform(x_val[i]), "true" if out[i].tolist()[0][1]<out[i].tolist()[0][0] else "wrong")
#print(out[0])

