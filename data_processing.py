import numpy as np
import torch


FILE_PATH = "data/input.txt"
WINDOW_SIZE = 10
TRAIN_SPLIT = 0.8
BATCH_SIZE = 32

def read_data(file_path=FILE_PATH):
    with open(file_path, "r") as file:
        data = file.read()
    return data

class CharLevelProcessor():
    def __init__(self, data):
        self.chars = sorted(list(set(data)))
        self.char_to_index = {char: i for i, char in enumerate(self.chars)}
        self.index_to_char = {i: char for i, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    
    def get_vocab_size(self):
        return self.vocab_size

    def get_chars(self):
        return self.chars
    
    def encode_data(self, data):
        return [self.char_to_index[char] for char in data]
    
    def decode_data(self, data):
        return [self.index_to_char[index] for index in data]

def window_data_test_train(data,train_split=TRAIN_SPLIT,window_size=WINDOW_SIZE):
    train = data[:int(len(data)*train_split)]
    test = data[int(len(data)*train_split):]
    train_window = np.lib.stride_tricks.sliding_window_view(train, window_shape=window_size, axis=0)
    test_window = np.lib.stride_tricks.sliding_window_view(test, window_shape=window_size, axis=0)
    X_train = train_window[:-1, :]
    X_test = test_window[:-1, :]
    Y_train = train_window[1:, :]
    Y_test = test_window[1:, :]
    return X_train, X_test, Y_train, Y_test

def batch_data(X:torch.tensor, Y:torch.tensor, batch_size=BATCH_SIZE):
    n_batches = X.shape[0]//batch_size * batch_size
    X_batches = X[:n_batches]
    Y_batches = Y[:n_batches]    
    X_batches = X_batches.view(-1, batch_size, *X_batches.shape[1:])
    Y_batches = Y_batches.view( -1, batch_size, *Y_batches.shape[1:])
    return X_batches, Y_batches

