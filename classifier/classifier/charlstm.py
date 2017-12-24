import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, char_size, embedding_dim, hidden_dim,
            linear_dim, n_layers,
                 init_embedding, char_embedding_dim, 
                 char_hidden_dim, char_n_layers, dropout=.2,  bidirectional=False,
                 char_bidirectional=False,
                 gpu=False, maxpool=True):
        super(CharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight = nn.Parameter(init_embedding)

        self.char_embeddings = nn.Embedding(char_size, char_embedding_dim)

        self.num_directions = 1
        self.char_num_directions = 1
        if bidirectional: self.num_directions = 2
        if char_bidirectional: self.char_num_directions = 2

        self.hidden_dim = hidden_dim
        self.char_hidden_dim = char_hidden_dim

        self.lstm1 = nn.LSTM(input_size=embedding_dim + self.char_hidden_dim*self.char_num_directions,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                             bidirectional=bidirectional,
                            batch_first=True)
                            

        self.lstm2 = nn.LSTM(input_size=char_embedding_dim,
                            hidden_size=char_hidden_dim,
                            num_layers=char_n_layers,
                             bidirectional=char_bidirectional,
                            batch_first=True)
        


        self.gpu = gpu

        self.linear_dim = linear_dim

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

    def char_init_hidden(self, batch_size):
        if self.gpu:
            return (autograd.Variable(torch.zeros(self.char_num_directions,
                                                  batch_size,
                                                  self.char_hidden_dim)).cuda(),
                  autograd.Variable(torch.zeros(self.char_num_directions,
                                                batch_size,
                                                self.char_hidden_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(self.char_num_directions,
                                                  batch_size,
                                                  self.char_hidden_dim)),
                  autograd.Variable(torch.zeros(self.char_num_directions,
                                                batch_size,
                                                self.char_hidden_dim)))
    def forward(self, inputs, char_inputs):
        # char_inputs = batch x seqlen x wordlen
        char_outputs = []
        i = 0
        for char_input in char_inputs.split(1, dim=1):
            # char_input = batch x wordlen
            char_embed = self.char_embeddings(char_input.squeeze(1))
            # char_embed = batch x wordlen x charembeddim
            char_hidden = self.char_init_hidden(inputs.size(0))

            char_out_lstm, (char_h_n, char_c_n) = self.lstm2(
                    char_embed, char_hidden)

            #char_h_n is dir x bs x hidden dim
            # after permute bs x hidden x direections
            # after view bs x 1 x hiddendim*directions
            char_outputs.append(
                    char_h_n.permute(1, 2, 0).contiguous().view(
                        inputs.size(0), 1, -1))

        char_output = torch.cat(char_outputs, 1)
        # this is bs x seqlen x hidden
        
        # inputs = batch x sentence_length
        embeds = self.embeddings(inputs)
        # embeds = batch x sentence_length x embedding_dim
        hidden = self.init_hidden(inputs.size(0))

        concat = torch.cat([embeds, char_output], 2)
        # concat is now bs x seqlen x embed + hidden_char

        out_lstm, (h_n, c_n) = self.lstm1(concat, hidden)
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
