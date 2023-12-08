from cProfile import run
from xml.dom.minidom import Document
import torch
import torch.nn as nn
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
import fasttext
import fasttext.util
from poutyne import (
    set_seeds,
    Model,
    ModelCheckpoint,
    CSVLogger,
    Callback,
    ModelBundle,
    Metric,
    SKLearnMetrics,
    plot_history,
)

from datasets import load_dataset

dataset = load_dataset("midas/inspec")

cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

# print(len(dataset['train']))
# print(len(dataset['validation']))
# print(len(dataset['test']))
# print(len(dataset['train'][0]['doc_bio_tags']))

# for set in dataset['test']:
#     # print(set['id'])
#     if(len(set['document']) != len(set['doc_bio_tags'])):
#         print("co")

# quit()

batch_size = 32
lr = 0.1

class EmbeddingVectorizer:
    def __init__(self):
        """
        Embedding vectorizer
        """

        # fasttext.util.download_model('en', if_exists='ignore')
        self.embedding_model = fasttext.load_model("./cc.en.300.bin")

    def __call__(self, sentence):
        embeddings = []
        for word in sentence:
            embeddings.append(self.embedding_model[word])
        return embeddings

embedding_model = EmbeddingVectorizer()

class Vectorizer:
    def __init__(self, dataset, embedding_model, predict=False):
        self.data = dataset
        self.embedding_model = embedding_model
        self.predict = predict
        self.tags_set = {
            "I": 0,
            "O": 1,
            "B": 2,
        }

    def __len__(self):
        # for the dataloader
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]

        if not self.predict:
            sentence = data['document']
            sentence_vector = self.embedding_model(sentence)

            tags = data['doc_bio_tags']
            idx_tags = self._convert_tags_to_idx(tags)
            return sentence_vector, idx_tags

        sentence_vector = self.embedding_model(data)
        return sentence_vector

    def _convert_tags_to_idx(self, tags):
        idx_tags = []
        for tag in tags:
            idx_tags.append(self.tags_set[tag])
        return idx_tags

train_data_vectorize = Vectorizer(dataset['train'], embedding_model)
valid_data_vectorize = Vectorizer(dataset['validation'], embedding_model)
test_data_vectorize = Vectorizer(dataset['test'], embedding_model)

# print(train_data_vectorize)

def pad_collate_fn(batch):
    """
    The collate_fn that can add padding to the sequences so all can have
    the same length as the longest one.

    Args:
        batch (List[List, List]): The batch data, where the first element
        of the tuple are the word idx and the second element are the target
        label.

    Returns:
        A tuple (x, y). The element x is a tensor of packed sequence .
        The element y is a tensor of padded tag indices. The word vectors are
        padded with vectors of 0s and the tag indices are padded with -100s.
        Padding with -100 is done because of the cross-entropy loss and the
        accuracy metric ignores the targets with values -100.
    """

    # This gets us two lists of tensors and a list of integer.
    # Each tensor in the first list is a sequence of word vectors.
    # Each tensor in the second list is a sequence of tag indices.
    # The list of integer consist of the lengths of the sequences in order.
    sequences_vectors, sequences_labels, lengths = zip(*[
        (torch.FloatTensor(np.stack(seq_vectors)), torch.LongTensor(labels), len(seq_vectors))
        for (seq_vectors, labels) in sorted(batch, key=lambda x: len(x[0]), reverse=True)
    ])

    lengths = torch.LongTensor(lengths)

    padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True, padding_value=0)
    pack_padded_sequences_vectors = pack_padded_sequence(
        padded_sequences_vectors, lengths.cpu(), batch_first=True
    )  # We pack the padded sequence to improve the computational speed during training

    padded_sequences_labels = pad_sequence(sequences_labels, batch_first=True, padding_value=-100)

    return pack_padded_sequences_vectors, padded_sequences_labels

train_loader = DataLoader(train_data_vectorize, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
valid_loader = DataLoader(valid_data_vectorize, batch_size=batch_size, collate_fn=pad_collate_fn)
test_loader = DataLoader(test_data_vectorize, batch_size=batch_size, collate_fn=pad_collate_fn)
# 
# train_loader = DataLoader(train_data_vectorize, batch_size=batch_size, shuffle=True)
# valid_loader = DataLoader(valid_data_vectorize, batch_size=batch_size)
# test_loader = DataLoader(test_data_vectorize, batch_size=batch_size)

dimension = 300
num_layer = 1
bidirectional = False


lstm_network = nn.LSTM(input_size=dimension, hidden_size=dimension,
                       num_layers=num_layer, bidirectional=bidirectional,
                       batch_first=True)

input_dim = dimension # the output of the LSTM
tag_dimension = 3

fully_connected_network = nn.Linear(input_dim, tag_dimension)

class FullNetWork(nn.Module):
    def __init__(self, lstm_network, fully_connected_network):
        super().__init__()
        self.hidden_state = None

        self.lstm_network = lstm_network
        self.fully_connected_network = fully_connected_network

    def forward(self, pack_padded_sequences_vectors: PackedSequence):
        """
            Defines the computation performed at every call.
        """
        lstm_out, self.hidden_state = self.lstm_network(pack_padded_sequences_vectors)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        tag_space = self.fully_connected_network(lstm_out)
        return tag_space.transpose(-1, 1)  # We need to transpose since it's a sequence

full_network = FullNetWork(lstm_network, fully_connected_network)

optimizer = optim.SGD(full_network.parameters(), lr)
len_o = 0
len_b = 0
len_i = 0
for x in dataset['test']:
    # print(x)
    len_o += x['document'].count("O")
    len_b += x['document'].count("B")
    len_i += x['document'].count("I")
summed = len_i + len_b + len_o
print([len_i / summed, len_b / summed, len_o / summed])
weights = torch.tensor([len_i, len_b, len_o]) / summed


# quit()
weights = 1./weights
loss_function = nn.CrossEntropyLoss(ignore_index=-100, weight=weights)

model = Model(full_network, optimizer, loss_function,
              batch_metrics=['accuracy'],
              device=device)

# history = model.fit_generator(train_loader, valid_loader, epochs=10)
# model.save_weights("baseline.pth")
model.load_weights("baseline.pth")
# test_loss, test_acc = model.evaluate_generator(test_loader)
# quit()
predictions, ground_truth = model.predict_generator(test_loader, has_ground_truth =True, return_ground_truth=True, concatenate_returns=False)
# print(predictions[0:2], ground_truth[0:2])
# print(predictions[0].shape)
# print(ground_truth[0].shape)
# print(predictions[0][:, :, 0])
# print(predictions[1].shape)
# print(ground_truth[1].shape)
# _ = plot_history(
#     history,
#     metrics=['loss', 'acc'],
#     labels=['Loss', 'Accuracy'],
#     titles='Training of LSTM seq tagging',
#     save = True
# )

idx_to_tags = {
    0: "I",
    1: "O",
    2: "B",
    -100: "O",
}

idx_predictions = []
for batch in predictions:
    idx_predictions.extend(batch.argmax(axis=1).tolist())

# print(idx_predictions[0])

tags_predictions = []
for address in idx_predictions:
    # print(len(address))
    tags_predictions.append([idx_to_tags.get(tag) for tag in address])

# idx_gt = []
# for batch in ground_truth:
#     idx_gt.extend(batch.argmax(axis=1).tolist())

# print(ground_truth.)
len_o = 0
len_b = 0
len_i = 0
tags_gt = []
for mat in ground_truth:
    # print(np.unique(mat, return_counts=True))
    for i in range(mat.shape[0]):
        address = mat[i, :]
        temp = [idx_to_tags.get(tag) for tag in address]
        tags_gt.append(temp)
        len_o += temp.count("O")
        len_b += temp.count("B")
        len_i += temp.count("I")

print([len_i, len_b, len_o])

# for x in dataset['test']:
    # print(x)

# print(tags_gt[2])
# print(tags_predictions[2])

# plt.clf()
# cf_matrix = confusion_matrix(y_true, y_pred)
# df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes], columns = [i for i in classes])
# plt.figure(figsize = (12,7))
# sn.heatmap(df_cm, annot=True)
# plt.savefig('3b_2c.png')

# print("Training Started")
# accuracy_list = []
# loss_list = []
# epochs = 3
# running_loss = 0
# for epoch in range(epochs):
#     print(epoch)
#     acc = 0
#     loss = 0
#     i = 0
#     for index, (sentences, tags) in enumerate(generate_batch(train)): optimizer.zero_grad()
#     sentences_in, max_len = prepare_sequences (sentences, word_to_ix)
#     targets, max_len = prepare_sequences (tags, tag_to_ix)
#     #print(targets) 
#     tag_scores = model(sentences_in)
#     #print(tag_scores)
#     loss = loss_function(tag_scores.view(-1, 3), targets.view(-1))
#     loss.backward()
#     optimizer.step()
#     argmaxed = torch.argmax(tag_scores, dim=-1)
#     #print(argmaxed)
    
#     mask = targets > -1
#     relevant = argmaxed[mask]
#     #print(relevant)
#     acc = ((relevant == targets [mask]).sum()/relevant.shape[0]).item()
#     loss = loss.item()
#     accuracy_list.append(acc)
#     loss_list.append(loss)
#     running_loss = 0.99*running_loss + 0.01*loss
#     if index % 10 == 0:
#         print(f'running loss at batch {index} is {running_loss} and acc is {acc}')