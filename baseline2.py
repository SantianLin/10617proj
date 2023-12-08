from cProfile import run
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence, PackedSequence
import seaborn as sn
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

dataset = load_dataset("midas/inspec")

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

training_data = dataset['train']
testing_data = dataset['test']
word_to_ix = {}

for sentence in training_data:
    sent = sentence['document']
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
for sentence in testing_data:
    sent = sentence['document']
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
tag_to_ix = {"O": 0, "I": 1, "B": 2}  # Assign each tag with a unique index

EMBEDDING_DIM = 300
HIDDEN_DIM = 64
N_EPOCH = 25

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, dropout=0.3)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.75)

train_loss = np.zeros(N_EPOCH)
test_loss = np.zeros(N_EPOCH)
train_acc = np.zeros(N_EPOCH)
test_acc = np.zeros(N_EPOCH)

for epoch in range(N_EPOCH):
    # running_loss = 0
    for sent_agg in training_data:
        sentence = sent_agg['document']
        tags = sent_agg['doc_bio_tags']
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        # running_loss += loss.item()
        optimizer.step()
    running_loss = 0
    acc = 0
    count = 0
    for sent_agg in training_data:
        sentence = sent_agg['document']
        tags = sent_agg['doc_bio_tags']
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        y_pred = model(sentence_in)
        loss = loss_function(y_pred, targets)
        acc += (torch.argmax(y_pred, 1) == targets).float().sum()
        
        running_loss += loss.item()
        count += len(targets)
    train_loss[epoch] = running_loss/len(training_data)
    acc /= count
    train_acc[epoch] = acc

    acc = 0
    count = 0
    running_loss = 0
    for sent_agg in testing_data:
        sentence = sent_agg['document']
        tags = sent_agg['doc_bio_tags']

        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        y_pred = model(sentence_in)
        loss = loss_function(y_pred, targets)
        
        acc += (torch.argmax(y_pred, 1) == targets).float().sum()
        running_loss += loss.item()
        count += len(targets)
    acc /= count
    test_acc[epoch] = acc
    test_loss[epoch] = running_loss/len(testing_data)
    print("Epoch %d: training accuracy %.2f%%" % (epoch, train_acc[epoch]*100))
    print("Epoch %d: testing accuracy %.2f%%" % (epoch, acc*100))

torch.save(model.state_dict(), "baseline.pth")

# model.load_state_dict(torch.load("baseline.pth"))

running_loss = 0
acc = 0
count = 0
f1 = 0
recall = 0
precision = 0

for sent_agg in training_data:
    sentence = sent_agg['document']
    tags = sent_agg['doc_bio_tags']
    sentence_in = prepare_sequence(sentence, word_to_ix)
    targets = prepare_sequence(tags, tag_to_ix)

    y_pred = model(sentence_in)

    f1 += f1_score(targets, torch.argmax(y_pred, 1), average='macro')/len(training_data)
    recall += recall_score(targets, torch.argmax(y_pred, 1), average='macro')/len(training_data)
    precision += precision_score(targets, torch.argmax(y_pred, 1), average='macro')/len(training_data)
    acc += accuracy_score(targets, torch.argmax(y_pred, 1))/len(training_data)

print("train_f1", f1)
print("train_recall", recall)
print("train_precision", precision)
print("train_acc", acc)

acc = 0
count = 0
f1 = 0
recall = 0
precision = 0
for sent_agg in testing_data:
    sentence = sent_agg['document']
    tags = sent_agg['doc_bio_tags']

    sentence_in = prepare_sequence(sentence, word_to_ix)
    targets = prepare_sequence(tags, tag_to_ix)
    y_pred = model(sentence_in)
    f1 += f1_score(targets, torch.argmax(y_pred, 1), average='macro')/len(testing_data)
    recall += recall_score(targets, torch.argmax(y_pred, 1), average='macro')/len(testing_data)
    precision += precision_score(targets, torch.argmax(y_pred, 1), average='macro')/len(testing_data)
    acc += accuracy_score(targets, torch.argmax(y_pred, 1))/len(testing_data)
print("test_f1", f1)
print("test_recall", recall)
print("test_precision", precision)
print("test_acc", acc)

# print("Epoch %d: training accuracy %.2f%%" % (epoch, train_acc[epoch]*100))
# print("Epoch %d: testing accuracy %.2f%%" % (epoch, acc*100))

plt.plot(range(1, N_EPOCH+1), train_loss, label = "train")
plt.plot(range(1, N_EPOCH+1), test_loss, label = "test")
plt.xlabel('epochs')
plt.ylabel('Average NLLLoss')
plt.title("train and test loss")
plt.legend()
plt.savefig("bl_Loss.png")
plt.show()

plt.clf()
plt.plot(range(1, N_EPOCH+1), train_acc, label = "train")
plt.plot(range(1, N_EPOCH+1), test_acc, label = "test")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title("train and test accuracy")
plt.legend()
plt.savefig("bl_Acc.png")
plt.show()