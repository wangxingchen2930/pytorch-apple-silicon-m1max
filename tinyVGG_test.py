#######################
# Credit to mrdbourke #
#######################

import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from timeit import default_timer as timer 

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch import nn

from tqdm.auto import tqdm

import os

from pathlib import Path


# Helper function
def print_train_time(start, end, device=None, machine=None):
    """Prints difference between start and end time.
    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    if device:  
        print(f"\nTrain time on {machine} using PyTorch device {device}: {total_time:.3f} seconds\n")
    else:
        print(f"\nTrain time: {total_time:.3f} seconds\n")
    return round(total_time, 3)


# Create model
class TinyVGG(nn.Module):
    """Creates the TinyVGG architecture.

    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
    See the original architecture here: https://poloclub.github.io/cnn-explainer/

    Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),  
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                        stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
#             nn.Linear(in_features=hidden_units*5*5,out_features=output_shape) # image shape 32, 32
            nn.Linear(in_features=hidden_units*53*53,out_features=output_shape) # image shape 224, 224
        )

    def forward(self, x: torch.Tensor):
# #         print(x.shape)
#         x = self.conv_block_1(x)
# #         print(x.shape)
#         x = self.conv_block_2(x)
# #         print(x.shape)
#         x = self.classifier(x)
#         return x
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


# Set up training/testing loop
def train_step(model: torch.nn.Module, 
            dataloader: torch.utils.data.DataLoader, 
            loss_fn: torch.nn.Module, 
            optimizer: torch.optim.Optimizer,
            device: torch.device):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
#         X, y = X.to(device, non_blocking=True, memory_format=torch.channels_last), y.to(device, non_blocking=True)
#         X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
            dataloader: torch.utils.data.DataLoader, 
            loss_fn: torch.nn.Module,
            device: torch.device):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
#             X, y = X.to(device, non_blocking=True, memory_format=torch.channels_last), y.to(device, non_blocking=True)
#             X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
        train_dataloader: torch.utils.data.DataLoader, 
        test_dataloader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        device: torch.device):
    
    print(f"[INFO] Training model {model.__class__.__name__} on device '{device}' for {epochs} epochs...")
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
#     for epoch in range(epochs):
    for epoch in tqdm(range(epochs)):
        # Do eval before training (to see if there's any errors)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device)
        
        train_loss, train_acc = train_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=device)
        
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results





if __name__ == '__main__':  



    print(f"PyTorch version: {torch.__version__}")

    # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")

    # Set the device      
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")


    # Set hyperparameters
    BATCH_SIZE = 32 # good for your health: https://twitter.com/ylecun/status/989610208497360896
    IMAGE_SIZE = (224, 224) # (height, width) smaller images means faster computing 
    NUM_EPOCHS = 3 # only run for a short period of time... we don't have all day
    DATASET_NAME = "cifar10" # dataset to use (there are more in torchvision.datasets)
    MACHINE = "Apple M1 Max" # change this depending on where you're runing the code
    NUM_WORKERS = 2 # set number of cores to load data


    # Get data
    simple_transform = transforms.Compose([
        transforms.Resize(size=IMAGE_SIZE),
        transforms.ToTensor()
    ])

    # Get Datasets
    train_data = datasets.CIFAR10(root="data",
                                train=True,
                                transform=simple_transform,
                                download=True)

    test_data = datasets.CIFAR10(root="data",
                                train=False,
                                transform=simple_transform,
                                download=True)

    print(f"Number of training samples: {len(train_data)}, number of testing samples: {len(test_data)}")

    # Create DataLoaders
    train_dataloader = DataLoader(train_data,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=NUM_WORKERS,
                                pin_memory=True)

    test_dataloader = DataLoader(test_data,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=True)

    print("single data shape:", train_data[0][0].shape)

    # Get class names
    class_names = train_data.classes

    print("Class names:", class_names)


    # Do a dummy forward pass (to test the model) 
    model = TinyVGG(input_shape=3,
                hidden_units=10,
                output_shape=3)

    print("dummy forward pass test:", model(torch.randn(1, 3, 224, 224)))


    # Set random seed
    torch.manual_seed(42)

    # Create device list
    devices = ["mps","cpu"]

    for device in devices:

        # Recreate an instance of TinyVGG
        model = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                        hidden_units=10, 
                        output_shape=len(train_data.classes)).to(device)

        # Setup loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

        # Start the timer
        start_time = timer()

        # Train model
        model_results = train(model=model, 
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn, 
                            epochs=NUM_EPOCHS,
                            device=device)

        # End the timer
        end_time = timer()

        # Print out timer and results
        total_train_time = print_train_time(start=start_time,
                                            end=end_time,
                                            device=device,
                                            machine=MACHINE)
        
        # Create results dict
        results = {
        "machine": MACHINE,
        "device": device,
        "dataset_name": DATASET_NAME,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE[0],
        "num_train_samples": len(train_data),
        "num_test_samples": len(test_data),
        "total_train_time": round(total_train_time, 3),
        "time_per_epoch": round(total_train_time/NUM_EPOCHS, 3),
        "model": model.__class__.__name__
        }
        
        results_df = pd.DataFrame(results, index=[0])
        
        # Write CSV to file
        if not os.path.exists("results/"):
            os.makedirs("results/")

        results_df.to_csv(f"results/{MACHINE.lower().replace(' ', '_')}_{device}_{DATASET_NAME}_image_size.csv", 
                        index=False)

    # Inspect results
    # Get CSV paths from results
    results_paths = list(Path("results").glob("*.csv"))

    df_list = []
    for path in results_paths:
        df_list.append(pd.read_csv(path))
    results_df = pd.concat(df_list).reset_index(drop=True)
    print(results_df)

    # Get names of devices
    machine_and_device_list = [row[1][0] + " (" + row[1][1] + ")" for row in results_df[["machine", "device"]].iterrows()]

    # Plot and save figure
    plt.figure(figsize=(10, 7))
    plt.style.use('fivethirtyeight')
    plt.bar(machine_and_device_list, height=results_df.time_per_epoch)
    plt.title(f"PyTorch TinyVGG Training on CIFAR10 with batch size {BATCH_SIZE} and image size {IMAGE_SIZE}", size=16)
    plt.xlabel("Machine (device)", size=14)
    plt.ylabel("Seconds per epoch (lower is better)", size=14);
    save_path = f"results/{model.__class__.__name__}_{DATASET_NAME}_benchmark_with_batch_size_{BATCH_SIZE}_image_size_{IMAGE_SIZE[0]}.png"
    print(f"Saving figure to '{save_path}'")
    plt.savefig(save_path)