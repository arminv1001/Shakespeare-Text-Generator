from data_processing import CharLevelProcessor, window_data_test_train, batch_data,read_data
import torch
from torch.nn import functional as F
from transfomer import TransformerDecoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

data = read_data()
processor = CharLevelProcessor(data)
vocab_size = processor.get_vocab_size()
chars = processor.get_chars()



d_k = vocab_size
d_v = vocab_size
emd_dim = 128
num_epochs = 1



data_encoded = processor.encode_data(data)
X_train, X_test, Y_train, Y_test = window_data_test_train(data_encoded)
X_train, Y_train = torch.tensor(X_train), torch.tensor(Y_train)
X_test, Y_test = torch.tensor(X_test), torch.tensor(Y_test)
X_train, Y_train = batch_data(X_train, Y_train)
X_test, Y_test = batch_data(X_test, Y_test)

model = TransformerDecoder(num_heads=8,d_k=d_k,d_v=d_v,emd_dim=emd_dim,vocab_size=vocab_size,output_layer=True, input_layer=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for i in range(X_train.shape[0]):
        inputs = X_train[i]
        targets = Y_train[i]

        
        # Forward pass
        outputs = model(inputs)
        outputs = outputs.view(-1, outputs.shape[-1])
        targets = targets.view(-1)
        
        loss = F.cross_entropy(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if i % 1000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{X_train.shape[0]}], Loss: {loss.item():.4f}')

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(X_train):.4f}')
    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i in range(X_test.shape[0]):
            inputs = X_test[i]
            targets = Y_test[i]


            # Forward pass
            outputs = model(inputs)
            
            outputs = outputs.view(-1, outputs.shape[-1])
            targets = targets.view(-1)
            loss = F.cross_entropy(outputs, targets)

            val_loss += loss.item()

    print(f'Validation Loss: {val_loss/len(X_test):.4f}')
torch.save(model.state_dict(), f"data/model/transformer_{num_epochs}.pth")
# Generate text
#model.eval()
#test = model.generate(Y_test[0],embedding)
#test_result = test[0].detach().cpu().numpy().tolist()
#test_result = processor.decode_data(test_result)
#print("".join(test_result))