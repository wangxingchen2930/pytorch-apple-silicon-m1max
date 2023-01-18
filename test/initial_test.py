# Credit to mrdbourke

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

print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Create data and send it to the device
x = torch.rand(size=(3, 4)).to(device)
print("random data with size of 3*4:")
print(x)