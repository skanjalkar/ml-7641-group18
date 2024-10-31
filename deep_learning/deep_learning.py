import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn

class ChessPositionDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        
         # Initialize lists to hold all features and labels
        self.all_features = []
        self.all_labels = []

        # Load all features and labels from files
        for filename in sorted(os.listdir(features_dir)):
            if filename.endswith('.npy'):
                file_path = os.path.join(features_dir, filename)
                features = np.load(file_path)
                self.all_features.append(features)
        
        for filename in sorted(os.listdir(labels_dir)):
            if filename.endswith('.npy'):
                file_path = os.path.join(labels_dir, filename)
                labels = np.load(file_path)
                self.all_labels.append(labels)

        # Convert lists of arrays into a single NumPy array
        self.features = np.concatenate(self.all_features, axis=0)
        self.labels = np.concatenate(self.all_labels, axis=0)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Load feature and label
        features_tensor = torch.from_numpy(self.features[idx]).float()  
        labels_tensor = torch.from_numpy(np.array(self.labels[idx].astype(np.int32))).long()  
        return features_tensor, labels_tensor

#working with only 1500-1600 elo range
#create a dataset
chesspositions = ChessPositionDataset('features/','labels')

#split into test and train dataset
train_size = int(0.8 * len(chesspositions))  # 80% for training
test_size = len(chesspositions) - train_size  # Remaining 20% for testing

# Perform the split
train_dataset, test_dataset = random_split(chesspositions, [train_size, test_size])

# Create DataLoaders for both datasets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8*8*17, 1088),
            nn.ReLU(),
            nn.Linear(1088, 1088),
            nn.ReLU(),
            nn.Linear(1088,1088),
            nn.ReLU(),
            nn.Linear(1088,1088),
            nn.ReLU(),
            nn.Linear(1088,1088),
            nn.ReLU(),
            nn.Linear(1088,10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
#create the model
model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")





