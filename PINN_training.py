## Imports:
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim

## Prepare Dataset
# Load data from .mat files
input_data = loadmat('Dataset/input_waves.mat')['input_waves'] # Adjust column range as needed
freq_response = loadmat('Dataset/forier_response.mat')['fourier_response']  # Adjust column range as needed
freq_progression = loadmat('Dataset/frequency_progression.mat')['freq_prog']  # Adjust column range as needed

# Separate real and imaginary parts
X_real = input_data.real
X_imag = input_data.imag

# Define noise parameters
noise_std = 1  # Standard deviation of the Gaussian noise

# Generate noise
noise_real = np.random.normal(0, noise_std, X_real.shape)
noise_imag = np.random.normal(0, noise_std, X_imag.shape)

# Add noise to the real and imaginary parts
X_real_noisy = X_real + 1*noise_real
X_imag_noisy = X_imag + 1*noise_imag

# Combine real and imaginary parts along the feature dimension (axis=1)
X = np.vstack((X_real_noisy, X_imag_noisy)).T
freq_response = freq_response.T
y = freq_progression.T*300000000/3000000000

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
print('Input shape and datatype: ', np.shape(X_tensor), type(X_tensor))
print('Output shape and datatype: ', np.shape(y_tensor), type(y_tensor))

# Define the split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create dataset and dataloader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=4001, shuffle=False)

# Get the number of samples
num_samples = len(dataset)

# Compute the number of samples for each split
num_train = int(train_ratio * num_samples)
num_val = int(val_ratio * num_samples)
num_test = num_samples - num_train - num_val  # Ensure all samples are used

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

# Create DataLoaders for each dataset
batch_size = 4001
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Example: Check the size of each split
print(f'Train samples: {len(train_dataset)}')
print(f'Validation samples: {len(val_dataset)}')
print(f'Test samples: {len(test_dataset)}')

## Create Model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, freq_response_dim, output_dim, dropout_prob=0.5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)  # Dropout layer
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)  # Dropout layer
        self.fc3 = nn.Linear(hidden_dim2, freq_response_dim)
        self.dropout3 = nn.Dropout(dropout_prob)  # Dropout layer
        self.fc4 = nn.Linear(freq_response_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)  # Apply dropout
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)  # Apply dropout
        x = self.fc3(x)
        x = self.dropout3(x)  # Apply dropout
        x = self.fc4(x)
        return x

# Define model parameters
input_dim = X_tensor.shape[1]  # Number of features in X
hidden_dim1 = 250
hidden_dim2 = 500
freq_response_dim = freq_response.shape[1]
output_dim = y_tensor.shape[1]  # Number of features in y
dropout_prob = 0.1  # Probability of dropout

# Instantiate the model
model = SimpleNN(input_dim, hidden_dim1, hidden_dim2, freq_response_dim, output_dim, dropout_prob)

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)

## Train
# Training settings
num_epochs = 1000  # Number of epochs
train_loss = []
val_loss = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    
    # Training phase
    for batch_X, batch_y in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_X)
        
        # Compute the loss
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate training loss
        running_train_loss += loss.item()
    
    # Calculate average training loss for the epoch
    avg_train_loss = running_train_loss / len(train_loader)
    train_loss.append(avg_train_loss)

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():  # No need to compute gradients
        for batch_X, batch_y in val_loader:
            # Forward pass
            outputs = model(batch_X)
            
            # Compute the loss
            loss = criterion(outputs, batch_y)
            
            # Accumulate validation loss
            running_val_loss += loss.item()
    
    # Calculate average validation loss for the epoch
    avg_val_loss = running_val_loss / len(val_loader)
    val_loss.append(avg_val_loss)

    # Print loss for every epoch
    if epoch%20==0: print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}')

print('Training finished')

## Evaluation
model.eval()
# Training
for batch_X, batch_y in train_loader:
    train_outputs = model(batch_X)
    train_gt = batch_y

# Validation and testing phase
model.eval()
with torch.no_grad():  # No need to compute gradients
    for batch_X, batch_y in val_loader:
        # Forward pass
        val_outputs = model(batch_X)
        val_gt = batch_y

    for batch_X, batch_y in test_loader:
        # Forward pass
        test_outputs = model(batch_X)
        test_gt = batch_y

print(train_outputs.shape, train_gt.shape)
print(val_outputs.shape, val_gt.shape)
print(test_outputs.shape, test_gt.shape)
print(criterion(train_gt, train_outputs), criterion(val_gt, val_outputs), criterion(test_gt, test_outputs))