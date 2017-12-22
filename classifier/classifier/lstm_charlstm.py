import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class NER_WordCharLSTM(nn.Module):
    """
    The bidirectionsl word LSTM with nested char LSTM model
    """
    def __init__(self, vocab_size, char_size, embedding_dim, hidden_dim, n_layers,
                 init_embedding, char_embedding_dim, char_hidden_dim,
                 char_n_layers, bidirectional=False, char_bidirectional=False):
        super(NER_WordCharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_hidden_dim = char_hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight = nn.Parameter(init_embedding)

        self.char_embeddings = nn.Embedding(char_size, char_embedding_dim)

        self.lstm1 = nn.LSTM(input_size=embedding_dim,
                             hidden_size=hidden_dim,
                             num_layers=n_layers,
                             bidirectional=bidirectional,
                             batch_first=True)
        self.char_lstm = nn.LSTM(input_size=char_embedding_dim,
                                 hidden_size=char_hidden_dim,
                                 num_layers=char_n_layers,
                                 bidirectional=char_bidirectional,
                                 batch_first=True)

        self.num_directions = 1
        self.char_num_directions = 1
        if bidirectional:
            self.num_directions = 2
        if char_bidirectional:
            self.char_num_directions = 2

        self.linear = nn.Linear(
            self.hidden_dim * self.num_directions\
            + self.char_hidden_dim * self.char_num_directions , 3)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(self.num_directions,
                                              batch_size,
                                              self.hidden_dim)).cuda(),
             autograd.Variable(torch.zeros(self.num_directions,
                                           batch_size,
                                           self.hidden_dim)).cuda())

    def char_init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(self.num_directions,
                                              batch_size,
                                              self.char_hidden_dim)).cuda(),
             autograd.Variable(torch.zeros(self.num_directions,
                                           batch_size,
                                           self.char_hidden_dim)).cuda())
    def forward(self, inputs, char_inputs):
        # inputs = batch x sentence_length
        embeds = self.embeddings(inputs)
        # embeds = batch x sentence_length x embedding_dim
        hidden = self.init_hidden(inputs.size(0))

        out_lstm, (h_n, c_n) = self.lstm1(embeds, hidden)
        # out_lstm = batch x sentence_length x hidden_size (last layer)

        # char_inputs = batch ix sentence_length x word_length
        char_outputs = []
        for char_input in char_inputs.split(1, dim=1):
            # char_input = batch x word_length
            char_embed = self.char_embeddings(char_input.squeeze(1))
            # char_embed = batch x word_length x charembed_dim
            char_hidden = self.char_init_hidden(inputs.size(0))

            char_out_lstm, (char_h_n, char_c_n) = self.char_lstm(
                char_embed, char_hidden)
            # char_h_n is directions x bs x hidden_dim
            # after permute, bs x hidden_dim x directions
            # after view bs x 1 x hiddendim*directions
            char_outputs.append(
                char_h_n.permute(1, 2, 0).contiguous().view(inputs.size(0), 1,  -1))

        char_output = torch.cat(char_outputs, 1)
        # char_output is bs x sentece_length x hidden

        concat = torch.cat([char_output, out_lstm], 2)

        #concat = nn.Dropout(0.5)(concat)


        out = self.linear(
            concat.contiguous().view(-1, self.hidden_dim * self.num_directions + self.char_hidden_dim * self.char_num_directions))
        # we want to flatten out_lstm to be (batch*seqlen) x hidden_size
        # linear wants (batch*sentence_len), hidden_dim and
        # outputs (batch*sentence_length) x 3

        log_probs = F.log_softmax(out)

        return log_probs
