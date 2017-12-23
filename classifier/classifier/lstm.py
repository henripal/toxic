import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class NER_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
            linear_dim, n_layers,
                 init_embedding, dropout=.2,  bidirectional=False,
                 gpu=False, maxpool=True):
        super(NER_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight = nn.Parameter(init_embedding)
        self.lstm1 = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                             bidirectional=bidirectional,
                            batch_first=True)
                            

        self.hidden_dim = hidden_dim
        self.num_directions = 1
        self.gpu = gpu
        self.linear_dim = linear_dim
        if bidirectional: self.num_directions = 2

        self.linear = nn.Linear(self.hidden_dim * self.num_directions,
                self.linear_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(self.linear_dim, 6)

        self.maxpool = maxpool

    def init_hidden(self, batch_size):
        if self.gpu:
            return (autograd.Variable(torch.zeros(self.num_directions,
                                                  batch_size,
                                                  self.hidden_dim)).cuda(),
                  autograd.Variable(torch.zeros(self.num_directions,
                                                batch_size,
                                                self.hidden_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(self.num_directions,
                                                  batch_size,
                                                  self.hidden_dim)),
                  autograd.Variable(torch.zeros(self.num_directions,
                                                batch_size,
                                                self.hidden_dim)))

    def forward(self, inputs):
        # inputs = batch x sentence_length
        embeds = self.embeddings(inputs)
        # embeds = batch x sentence_length x embedding_dim
        hidden = self.init_hidden(inputs.size(0))

        out_lstm, (h_n, c_n) = self.lstm1(embeds, hidden)
        # out_lstm = batch x sentence_length x hidden_size (last layer)

        if self.maxpool:
            maxpool = nn.MaxPool1d(inputs.size(1))
            out = out_lstm.permute(0, 2, 1) # batch x hidden x seqlen
            out = maxpool(out).squeeze() # batch x hidden
        else:
            out = out_lstm[:, -1, :].squeeze() # taking the last out;
            # out = batch x hidden_size

        out = self.linear(out.contiguous().view(-1, self.hidden_dim * self.num_directions))
        # we want to flatten out to be (batch) x hidden_size
        # linear wants (batch), hidden_dim and
        # outputs (batch) x linear_dim 

        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout2(out)

        log_probs = F.sigmoid(out)

        return log_probs
