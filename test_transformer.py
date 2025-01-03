import torch
from data_processing import CharLevelProcessor, window_data_test_train, batch_data,read_data
from transfomer import TransformerDecoder

def torch_to_str(X:torch.tensor, processor:CharLevelProcessor)->str:
    test_result = X[0].detach().cpu().numpy().tolist()
    test_result = processor.decode_data(test_result)
    return ''.join(test_result)



data = read_data()
processor = CharLevelProcessor(data)
vocab_size = processor.get_vocab_size()
chars = processor.get_chars()
vocab_size = processor.get_vocab_size()


d_k = vocab_size
d_v = vocab_size
emd_dim = 128
num_epochs = 1


MODEL_NAME = 'data/model/transformer_1.pth'
model = TransformerDecoder(num_heads=8,d_k=d_k,d_v=d_v,emd_dim=emd_dim,vocab_size=vocab_size,output_layer=True, input_layer=True)
model.load_state_dict(torch.load(MODEL_NAME, weights_only=True))
model.eval()
START_SENTENCE = 'The'
NUM_GEN_TOKENS = 10
WINDOW_SIZE = 10
input_seq = START_SENTENCE 


for index in range(WINDOW_SIZE - len(START_SENTENCE)):
    input_seq = ' ' + input_seq

X = input_seq
for _ in range(NUM_GEN_TOKENS):
    with torch.no_grad():
        X = torch.tensor([processor.encode_data(X)])
        X = model.generate(X)
        X = torch_to_str(X, processor)
        print(X)