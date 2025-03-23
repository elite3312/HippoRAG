import torch
import torch.nn as nn
import torch.optim as optim
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create a model instance and move it to the GPU
model = SimpleNN().cuda()

# Create dummy input and target tensors and move them to the GPU
inputs = torch.randn(100, 10).cuda()
targets = torch.randn(100, 1).cuda()

# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10000):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

print("Training completed successfully.")