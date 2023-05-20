# import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pretty_midi

class MidiDataset(Dataset):
    def __init__(self, file_list, seq_len):
        self.file_list = file_list
        self.seq_len = seq_len

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        midi_file = self.file_list[idx]
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        piano_roll = midi_data.get_piano_roll(fs=10)
        # Transpose the piano roll to C major or A minor keys
        key = np.random.choice(['C', 'A'])
        if key == 'C':
            transposed_piano_roll = piano_roll[21:109, :]
        else:
            transposed_piano_roll = piano_roll[9:97, :]
        # Convert the piano roll to binary values
        binary_piano_roll = (transposed_piano_roll > 0).astype(np.float32)
        # Pad or truncate the piano roll to a fixed length
        target_length = self.seq_len
        if binary_piano_roll.shape[1] < target_length:
            padded_piano_roll = np.zeros((binary_piano_roll.shape[0], target_length))
            padded_piano_roll[:, :binary_piano_roll.shape[1]] = binary_piano_roll
            binary_piano_roll = padded_piano_roll
        else:
            binary_piano_roll = binary_piano_roll[:, :target_length]
        return torch.from_numpy(binary_piano_roll)

