import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Fix the CNN class
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)  # Changed from 3 to 1 for MNIST
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16*4*4, 120)  # Adjusted size for MNIST
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)  # 10 digits

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16*4*4)  # Fixed view operation
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)  # Fixed typo
    return x

# Load and transform data
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='../data', train=True,
                    download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='../data', train=False,
                   download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
def train(epochs=5):
  for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data[0].to(device), data[1].to(device)
      
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      
      running_loss += loss.item()
      if i % 200 == 199:
        print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}')
        running_loss = 0.0
  
  print('Finished Training')
  torch.save(model.state_dict(), 'mnist_cnn.pth')

# Evaluate the model
def evaluate():
  correct = 0
  total = 0
  with torch.no_grad():
    for data in testloader:
      images, labels = data[0].to(device), data[1].to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  
  print(f'Accuracy on test set: {100 * correct / total:.2f}%')

# Run training and evaluation
train()
evaluate()
