import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as quantization

# Step 1: Define the Model
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# Hyperparameters
vocab_size = 1000
embedding_dim = 16
hidden_dim = 32
output_dim = 10

# Initialize the model
model = SimpleModel(vocab_size, embedding_dim, hidden_dim, output_dim)

# Step 2: Prepare the Model for QAT
for _, mod in model.named_modules():
   
    if isinstance(mod, nn.Embedding):
        mod.qconfig = quantization.float_qparams_weight_only_qconfig
    elif isinstance(mod, nn.Linear):
        mod.qconfig = quantization.get_default_qat_qconfig('fbgemm')

quantization.prepare_qat(model, inplace=True)





# Dummy input for demonstration
input_data = torch.randint(0, vocab_size, (1,))

# Step 3: Train the Model (Dummy training loop for demonstration)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy training loop
model.train()
for epoch in range(5):  # Train for 5 epochs
    optimizer.zero_grad()
    output = model(input_data)
    target = torch.randint(0, output_dim, (1,))
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Step 4: Convert the Model to Quantized Version
model.eval()
quantized_model = quantization.convert(model)
exit()

# Verify the quantized model
quantized_model(input_data)

print("Quantized model structure:")
print(quantized_model)
