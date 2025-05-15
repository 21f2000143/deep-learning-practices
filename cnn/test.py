import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import torch.nn.functional as F


# Define your model architecture
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

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize model and load the saved weights
model = CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

# Transform for input images - same as used during training
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.Resize((28, 28)),  # MNIST size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])

# Check if image path is provided
if len(sys.argv) < 2:
    print("Usage: python test.py <path_to_image>")
    sys.exit(1)

# Load image
image_path = sys.argv[1]
try:
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 1, 28, 28]
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(1).item()
        confidence = probabilities[0][predicted_class].item()
    
    print(f"Predicted Digit: {predicted_class} (Confidence: {confidence:.2%})")
except Exception as e:
    print(f"Error processing image: {e}")
    sys.exit(1)
