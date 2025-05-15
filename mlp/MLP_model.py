import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class MLP(nn.Module):
  def __init__(self, x_dim, h1_dim, h2_dim, out_dim):
    super().__init__()
    self.w1 = nn.Linear(x_dim, h1_dim, bias=True)
    self.w2 = nn.Linear(h1_dim, h2_dim, bias=True)
    self.w3 = nn.Linear(h2_dim, out_dim, bias=True)

  def forward(self, x):
    out = torch.relu(self.w1(x))
    out = torch.relu(self.w2(out))
    out = torch.relu(self.w3(out))
    return out

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='../data', train=True,
                                           download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='../data', train=False,
                                          download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model, loss and optimizer
input_dim = 28*28  # Flattened 28x28 images for MNIST
hidden1_dim = 128
hidden2_dim = 64
output_dim = 10  # 10 digits (0-9)

model = MLP(input_dim, hidden1_dim, hidden2_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(epochs=10):
  losses = []
  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
      # Reshape to flatten the images and move to device
      inputs = inputs.view(-1, 28*28).to(device)
      labels = labels.to(device)
      
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
    
    # Evaluate
    if (epoch+1) % 2 == 0:
      evaluate()
  
  return losses

def evaluate():
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for inputs, labels in test_loader:
      inputs = inputs.view(-1, 28*28).to(device)
      labels = labels.to(device)
      
      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  
  accuracy = 100 * correct / total
  print(f'Test Accuracy: {accuracy:.2f}%')

# Train the model
losses = train_model(epochs=10)

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Save the model
torch.save(model.state_dict(), 'digit_classification_model.pth')
