import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if CUDA is available
if torch.cuda.is_available():
    # Set the device to GPU (cuda)
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    # Fallback to CPU
    device = torch.device("cpu")
    print("Using CPU")

# Define your neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define layers of the model
        self.fc1 = nn.Linear(784, 512)  # First fully connected layer (example input size 784)
        self.fc2 = nn.Linear(512, 256)  # Second fully connected layer
        self.fc3 = nn.Linear(256, 10)   # Output layer (example output size 10)

    def forward(self, x):
        # Define the forward pass
        x = F.relu(self.fc1(x))  # Apply ReLU activation function after first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation function after second layer
        x = self.fc3(x)          # Output layer (no activation, assuming a classification task)
        return x

# Create the model and move it to the appropriate device
model = Net().to(device)

# Example: Create a tensor for a batch of input data and move it to the appropriate device
# Here, the size [64, 784] is an example, where 64 is the batch size, and 784 could be the input feature size (e.g., flattened 28x28 images)
data = torch.randn(64, 784).to(device)

# Forward pass
while True:
    output = model(data)

#