import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
import os
# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)  # Fully connected layer
        self.fc2 = nn.Linear(256, 10)   # Fully connected layer
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the model
model = SimpleModel()

# Prepare the model for QAT
model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')  # Use a backend (e.g., 'fbgemm' or 'qnnpack')

model = torch.quantization.prepare_qat(model, inplace=True)

# Training loop (simplified)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Dummy input and target
input_data = torch.randn(32, 784)  # Batch of 32, each with 784 features (e.g., 28x28 images)
target_data = torch.randint(0, 10, (32,))  # Random target classes (0-9)

# Training step
optimizer.zero_grad()
output = model(input_data)
loss = criterion(output, target_data)
loss.backward()
optimizer.step()

# After training, convert the model to a quantized model
model.eval()
quantized_model = torch.quantization.convert(model, inplace=True)


# Now you can perform inference with the quantized model
output_quantized = quantized_model(input_data)




torch.save(output_quantized, 'q.pt')
