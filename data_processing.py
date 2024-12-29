import numpy as np

FILE_PATH = "data/input.txt"
WINDOW_SIZE = 10
TRAIN_SPLIT = 0.8

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

def window_data_test_train(data, window_size=WINDOW_SIZE,train_split=TRAIN_SPLIT):
    data_window = np.lib.stride_tricks.sliding_window_view(data, window_shape=window_size, axis=0)
    X_train = data_window[:int(len(data)*train_split)-1, :]
    X_test = data_window[int(len(data)*train_split)-1:, :]
    Y_train = data_window[1:int(len(data)*train_split), :]
    Y_test = data_window[1+int(len(data)*train_split):, :]
    return X_train, X_test, Y_train, Y_test



#def preprocess_data(data):