import math
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from music21 import converter, instrument, note, chord
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

class MidiDataset(Dataset):
    def __init__(self, data_path, seq_len, device, embedding_dim=128, num_embeddings=128):
        self.device = device
        self.seq_len = seq_len
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=seq_len)
        self.data = []  # в data лежат уже padded-последовательности
        self.load_data(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        # print('data:', len(x))
        x = torch.tensor(x, requires_grad=False)
        x = self.embedding(x).to(self.device)
        # print('embedding:', x.shape)
        x = self.positional_encoding(x)
        # print('positional encoding:', x.shape)
        return x

    def load_data(self, data_path):
        midi_files = glob.glob(os.path.join(data_path, '*.mid'))
        for file_path in midi_files:
            try:
                midi = converter.parse(file_path)
                notes = []
                for element in midi.flat:
                    if isinstance(element, note.Note):
                        notes.append(element.pitch.midi)
                    elif isinstance(element, chord.Chord):
                        notes.append(element.sortAscending()[0].pitch.midi)
                if len(notes) > self.seq_len:
                    notes = notes[:self.seq_len]  # cut sequence
                else:
                    notes = notes + [0] * (self.seq_len - len(notes))  # pad sequence
                self.data.append(notes)
            except Exception as e:
                print(f'Error processing {file_path}: {e}')


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    def decode(self, x):
        x = x - self.pe[:x.size(0), :]
        return x
