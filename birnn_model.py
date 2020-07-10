import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, pad_idx)
        self.rnn = nn.RNN(input_size=embedding_dim,  hidden_size=hidden_dim, num_layers=n_layers, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, text, text_lengths):
        embedded_words = self.embedding(text)
        embedded_drop = self.dropout(embedded_words)
        inter_layer_packed = nn.utils.rnn.pack_padded_sequence(embedded_drop, text_lengths)
        inter_output, hidden = self.rnn(inter_layer_packed)
        output, output_length = nn.utils.rnn.pad_packed_sequence(inter_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))         
        return self.fc(hidden)