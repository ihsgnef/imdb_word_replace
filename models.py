import torch
import torch.nn as nn
from torch.autograd import Variable

class ConvModel(nn.Module):
    
    def __init__(self, cfg):
        super(ConvModel, self).__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_size)
        convs = []
        for i, filter_size in enumerate(cfg.filter_sizes):
            pad = filter_size // 2
            conv = nn.Sequential(
                    nn.Conv1d(cfg.embed_size, cfg.hidden_size, filter_size, padding=pad),
                    nn.ReLU()
                    )
            convs.append(conv)
        self.convs = nn.ModuleList(convs)
        self.fconns = nn.Sequential(
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_size * len(cfg.filter_sizes), cfg.output_size)
                )

    def forward(self, inputs):
        inputs_embed = self.embed(inputs) 
        # length, batch_size, embed_size -> batch_size, embed_dim, length
        inputs_embed = inputs_embed.transpose(0, 1).transpose(1, 2)
        mots = []
        for conv in self.convs:
            conv_out = conv(inputs_embed)
            mot, _ = conv_out.max(2)
            mots.append(mot.squeeze(2))
        mots = torch.cat(mots, 1)
        output = self.fconns(mots)
        return output
                    

class LSTMModel(nn.Module):

    def __init__(self, cfg):
        super(LSTMModel, self).__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_size)
        self.lstm = nn.LSTM(cfg.embed_size, cfg.hidden_size, cfg.num_layers)
        self.fconns = nn.Sequential( 
                nn.BatchNorm1d(cfg.hidden_size),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_size, cfg.output_size)
                )

    def forward(self, inputs):
        length, batch_size = inputs.size()
        hidden = self.init_hidden(batch_size)
        inputs_embed = self.embed(inputs)
        output, hidden = self.lstm(inputs_embed, hidden)
        output = output[-1]
        output = self.fconns(output)
        return output
        
    def init_weights(self):
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range, init_range)
        self.embed.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size):
        w = next(self.parameters()).data
        num_layers = self.lstm.num_layers
        hidden_size = self.lstm.hidden_size
        return (Variable(w.new(num_layers, batch_size, hidden_size).zero_()),
                Variable(w.new(num_layers, batch_size, hidden_size).zero_()))

