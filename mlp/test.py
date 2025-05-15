import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys

# Define your model architecture
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

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize and load the state dict
# Initialize model, loss and optimizer
input_dim = 28*28  # Flattened 28x28 images for MNIST
hidden1_dim = 128
hidden2_dim = 64
output_dim = 10  # 10 digits (0-9)

model = MLP(input_dim, hidden1_dim, hidden2_dim, output_dim).to(device)
model.load_state_dict(torch.load("digit_classification_model.pth"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Same normalization as in training
])

# Load image
image_path = sys.argv[1]
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0).to(device)  # Move tensor to the same device as the model

# Reshape to flatten the input (convert from [1, 1, 28, 28] to [1, 784])
input_tensor = input_tensor.view(-1, 28*28)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    predicted = output.argmax(1).item()

print(f"Predicted Digit: {predicted}")
