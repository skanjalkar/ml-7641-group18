import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from torch import optim
import torch.nn.functional as F
import argparse

# Global variables go here.
SCRIPT_ARGS = None
DIR_LIST = []
DATA_PATH = None

#### Parse the parameters for ML model ####
###########################################
# Initialize the parser
parser = argparse.ArgumentParser(description="Add arguments for training the deep neural network ML model")

# Define named arguments
parser.add_argument('--data_path', type=str, default="", help="Relative path to data directory")
parser.add_argument('--elo_list', type=str, default="1600-1700", help="Comma seperated ELO ranges to process.")

SCRIPT_ARGS = parser.parse_args()

DATA_PATH = SCRIPT_ARGS.data_path.split(',')
DIR_LIST = SCRIPT_ARGS.elo_list.split(',')

def get_files_to_process():
    """
    Helper method to fetch the numpy files corresponding to X and Y in the
    dataset.
    Returns:
        X_list, Y_list: List of numpy files in X and Y.
    """
    X_files = []
    Y_files = []
    for data_path in DATA_PATH:
        print("Compiling data for: ", data_path)
        total_files = 0
        # Iterate through each directory
        for entry in DIR_LIST:
            dir_name = os.path.join(data_path, entry)
            for filename in os.listdir(dir_name):
                # Get the X files
                if filename[0] == 'X':
                    X_files.append(os.path.join(dir_name, filename))
                    # Get the corresponding y file
                    y_file = 'y' + filename[1:]
                    Y_files.append(os.path.join(dir_name, y_file))
                    total_files = total_files + 1
        print("Number of files:", total_files)
    return X_files, Y_files

def get_numpy_data_from_files(X_files, Y_files):
    """
    Helper method to fetch data in numpy array form
    Returns:
        X, Y: two numpy arrays.
    """
    X = None
    Y = None
    for ii in range(len(X_files)):
        if X is None:
            X = np.load(X_files[ii])
            Y = np.load(Y_files[ii])
            continue
        X = np.concatenate((X, np.load(X_files[ii])))
        Y = np.concatenate((Y, np.load(Y_files[ii])))
    return X,Y

class ChessPositionDataset(Dataset):
    def __init__(self, X, Y):
        self.features = X
        self.labels = Y
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Load feature and label
        features_tensor = torch.from_numpy(self.features[idx]).float()  
        labels_tensor = torch.from_numpy(np.array(self.labels[idx].astype(np.int32))).long()  
        return features_tensor, labels_tensor

# Define neural network
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Input channels = 17, output channels = 64
        self.conv_layers = nn.Sequential()
        in_channels = 17
        out_channels = 64
        num_blocks = 6
        for i in range(num_blocks):
            self.conv_layers.add_module(f'conv{i+1}', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.conv_layers.add_module(f'bn{i+1}', nn.BatchNorm2d(out_channels))
            self.conv_layers.add_module(f'relu{i+1}', nn.ReLU())
            in_channels = out_channels  # For next block

        # Fully connected layer
        self.fc = nn.Linear(out_channels * 8 * 8, 1)  # Output is scalar (binary classification)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch_idx, (data, labels) in enumerate(dataloader):
        # Move data to device
        data = data.to(device)
        labels = labels.to(device).float()

        # If data shape is [batch_size, 8, 8, 17], rearrange it to [batch_size, 17, 8, 8]
        if data.shape[1:] == (8, 8, 17):
            data = data.permute(0, 3, 1, 2).contiguous()

        # Forward pass
        outputs = model(data)
        labels = labels.reshape(-1, 1)  # Ensure labels have shape [batch_size, 1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optional: print training progress
        if batch_idx % (10) == 0:
            print(f'Step [{batch_idx}], Loss: {loss.item():.4f}')

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.permute(0, 3, 1, 2).contiguous()
            y = y.reshape(-1,1).float()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            predicted = (pred > 0.5).float()
            correct += (predicted == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    X_files, Y_files = get_files_to_process()
    print("Test first 5 files from X: %s" % str(X_files[:5]))
    print("Test first 5 files from Y: %s" % str(Y_files[:5]))

    X, Y = get_numpy_data_from_files(X_files, Y_files)

    chesspositions = ChessPositionDataset(X,Y)

    #split into test and train dataset
    train_size = int(0.8 * len(chesspositions))  # 80% for training
    test_size = len(chesspositions) - train_size  # Remaining 20% for testing

    print(len(chesspositions), train_size, test_size)

    # Perform the split
    train_dataset, test_dataset = random_split(chesspositions, [train_size, test_size])

    # Create DataLoaders for both datasets
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Instantiate the model
    model = ChessNet()

    model.to(device)

    # Define the loss function
    criterion = nn.BCELoss()

    # Define the optimizer
    initial_lr = 0.1
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    epochs = 10

    for i in range(0,epochs):
        print(f"Epoch {i}---------")
        train(train_loader, model, criterion, optimizer)
        test(test_loader, model, criterion)
    print("Done!")





