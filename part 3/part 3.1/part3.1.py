import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import time

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # First 3x3 conv layer with 32 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second 3x3 conv layer with 32 filters
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        
        # Dynamically calculate the flattened size for the dense layer
        # LFW resized to 0.4 gives 50x37 images. 
        # Two 2x2 max pools roughly quarter the dimensions (12x9).
        self.fc = nn.Linear(32 * 12 * 9, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

def main():
    print("Loading LFW Dataset...")
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X = lfw_people.images
    y = lfw_people.target
    target_names = lfw_people.target_names
    num_classes = target_names.shape[0]

    # Preprocessing into 4D PyTorch Tensors [Batch, Channels, Height, Width]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train = torch.tensor(X_train[:, np.newaxis, :, :], dtype=torch.float32)
    X_test = torch.tensor(X_test[:, np.newaxis, :, :], dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = SimpleCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    print("Starting Training...")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        
    predictions = predicted.cpu().numpy()
    ground_truth = y_test.cpu().numpy()
    accuracy = np.sum(predictions == ground_truth) / len(ground_truth)
    
    print("\n--- CNN Classification Performance ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(ground_truth, predictions, target_names=target_names))
    
    torch.save(model.state_dict(), 'lfw_cnn_weights.pth')
    print("Model saved to 'lfw_cnn_weights.pth'")

if __name__ == "__main__":
    main()
