import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from torch import optim
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import yaml

# Load the hyperparameters
with open("hyperparameters.yaml", "r") as file:
    hyperparameters = yaml.safe_load(file) 
# Global variables go here.
SCRIPT_ARGS = None
DIR_LIST = []
DATA_PATH = None

all_preds = []
all_labels = []
training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []

# Initialize the parser
parser = argparse.ArgumentParser(description="Add arguments for training the deep neural network ML model")

# Define named arguments
parser.add_argument('--data_path', type=str, default="", help="Relative path to data directory")
parser.add_argument('--elo_list', type=str, default="1600-1700", help="Comma seperated ELO ranges to process.")
parser.add_argument('--model_type', type=str, default="mlp", help="""Type mlp for simple multi layer neural network or cnn for a convolutional neural network with residual connections""")

SCRIPT_ARGS = parser.parse_args()

DATA_PATH = SCRIPT_ARGS.data_path.split(',')
DIR_LIST = SCRIPT_ARGS.elo_list.split(',')
MODEL_TYPE = SCRIPT_ARGS.model_type

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

def print_confusion_matrix(dataloader):
    # run a random test on the dataset to get the predicted and label values
    all_preds = []
    all_labels = []
    TP_samples, TN_samples, FP_samples, FN_samples = [], [], [], []
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    if MODEL_TYPE == "mlp":
        model = ChessNetSimple(hyperparameters['model']['mlp_input'], hyperparameters['model']['mlp_hidden'], hyperparameters['model']['mlp_output'])
    elif MODEL_TYPE == "cnn":
        model = ChessNet()
    model.load_state_dict(torch.load('best_chessnet.pth'))
    model.to(device)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.permute(0, 3, 1, 2).contiguous()
            y = y.reshape(-1,1).float()
            pred = model(X)
            predicted = (pred > 0.5).float()
            correct += (predicted == y).type(torch.float).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            for i in range(len(y)):
                sample = X[i].cpu().numpy()  # Get the sample
                if y[i].item() == 1 and predicted[i].item() == 1:
                    TP_samples.append(sample)  # True Positive
                elif y[i].item() == 0 and predicted[i].item() == 0:
                    TN_samples.append(sample)  # True Negative
                elif y[i].item() == 0 and predicted[i].item() == 1:
                    FP_samples.append(sample)  # False Positive
                elif y[i].item() == 1 and predicted[i].item() == 0:
                    FN_samples.append(sample) 
    conf_matrix = confusion_matrix(all_labels, all_preds)
    # Display the confusion matrix
    TN, FP, FN, TP = conf_matrix.ravel()
    # Print the confusion matrix in a visually pleasing format
    print("Confusion Matrix:")
    print(f"{'':<12}{'Predicted 0':<15}{'Predicted 1'}")
    print(f"{'True 0':<12}{TN:<15}{FP}")
    print(f"{'True 1':<12}{FN:<15}{TP}")
    np.save('TP_samples.npy', np.array(TP_samples))
    np.save('TN_samples.npy', np.array(TN_samples))
    np.save('FP_samples.npy', np.array(FP_samples))
    np.save('FN_samples.npy', np.array(FN_samples))


# CNN Neural Network
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = hyperparameters['model']['in_channels']
        out_channels = hyperparameters['model']['out_channels']
        num_blocks = hyperparameters['model']['num_blocks']
        
        for i in range(num_blocks):
            # Add convolutional block
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
            
            # Add a 1x1 conv layer to match dimensions for residual if needed
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
            )
            
            in_channels = out_channels  # Update in_channels for the next block

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Dropout(hyperparameters['model']['dropout_factor']),
            nn.Linear(out_channels * 8 * 8, 1)
        )

    def forward(self, x):
        residual = x  # Initial residual
        for i in range(0, len(self.conv_layers), 2):
            conv_block = self.conv_layers[i]  # Main conv block
            shortcut_layer = self.conv_layers[i + 1]  # Shortcut for residual matching
            
            out = conv_block(residual)  # Apply conv block
            residual = out + shortcut_layer(residual)  # Add residual connection

        x = residual.view(residual.size(0), -1)  # Flatten
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

# A simple Multi Layer Neural Network
class ChessNetSimple(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChessNetSimple, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.relu = nn.ReLU()           
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)   
        self.relu2 = nn.ReLU()               
        self.fc3 = nn.Linear(hidden_size//2, output_size) 
        self.sigmoid = nn.Sigmoid()
       
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss = 0
    correct = 0
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
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).type(torch.float).sum().item()

        # Optional: print training progress
        if batch_idx % (10) == 0:
            print(f'Step [{batch_idx}], Loss: {loss.item():.4f}')
    train_loss = train_loss / len(dataloader)
    training_loss.append(train_loss)
    correct = correct / len(dataloader.dataset)
    training_accuracy.append(correct)

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
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    print("correct predicitons:", correct)
    print("Total items", size)
    test_loss /= num_batches
    validation_loss.append(test_loss)
    correct /= size
    validation_accuracy.append(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

if __name__ == "__main__":
    X_files, Y_files = get_files_to_process()
    print("Test first 5 files from X: %s" % str(X_files[:5]))
    print("Test first 5 files from Y: %s" % str(Y_files[:5]))

    X, Y = get_numpy_data_from_files(X_files, Y_files)

    chesspositions = ChessPositionDataset(X,Y)

    #split into test and train dataset
    train_size = int(hyperparameters['training']['train_split'] * len(chesspositions))  
    test_size = len(chesspositions) - train_size

    print(len(chesspositions), train_size, test_size)

    # Perform the split
    train_dataset, test_dataset = random_split(chesspositions, [train_size, test_size])

    # Create DataLoaders for both datasets
    train_loader = DataLoader(train_dataset, batch_size=hyperparameters['model']['batch_size'], shuffle=True, num_workers=hyperparameters['model']['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=hyperparameters['model']['batch_size'], shuffle=False, num_workers=hyperparameters['model']['num_workers'])

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Instantiate the model
    if MODEL_TYPE == "mlp":
        print("selected multi layer neural network !")
        model = ChessNetSimple(hyperparameters['model']['mlp_input'], hyperparameters['model']['mlp_hidden'], hyperparameters['model']['mlp_output'])
    elif MODEL_TYPE == "cnn":
        print("selected convolutional neural network !")
        model = ChessNet()

    model.to(device)

    # Define the loss function
    criterion = nn.BCELoss()

    # Define the optimizer
    initial_lr = hyperparameters['model']['initial_learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    best_loss = float('inf')
    epochs_no_improve = 0
    patience = hyperparameters['model']['patience']# Number of epochs to wait before early stopping
    max_epochs = hyperparameters['model']['epochs']

    # Define a learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor= hyperparameters['scheduler']['factor'], patience= hyperparameters['scheduler']['patience'], verbose=True)

    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}-----------------")
        
        # Train and evaluate each epoch
        train(train_loader, model, criterion, optimizer)
        test_loss = test(test_loader, model, criterion)
        
        # Check if there is an improvement
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'best_chessnet.pth')  # Save the best model's parameters
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            # Adjust learning rate if no improvement
            scheduler.step(test_loss)
        
        # Check for early stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    print("Training Complete")

    print_confusion_matrix(test_loader)

    # write training and testing loss to a file
    with open('losses_cnn.txt', 'w') as f:
        f.write("Epoch\tTraining Loss\tTest Loss\n")  # Writing header
        for epoch_num in range(epoch+1):
            f.write(f"{epoch_num + 1}\t{training_loss[epoch_num]:.4f}\t{validation_loss[epoch_num]:.4f}\n")

    # write training and testing accuracy to a file
    with open('accuracy_cnn.txt', 'w') as f:
        f.write("Epoch\tTraining Loss\tTest Loss\n")  # Writing header
        for epoch_num_accurancy in range(epoch+1):
            f.write(f"{epoch_num + 1}\t{training_accuracy[epoch_num_accurancy]:.4f}\t{validation_accuracy[epoch_num_accurancy]:.4f}\n")





