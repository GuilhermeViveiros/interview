import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import time
from functools import wraps
import json
from typing import List, Dict, Optional, Generator
import unittest

# 1. Data Structures and Algorithms
def find_first_non_repeating(s: str) -> Optional[str]:
    """
    Find the first non-repeating character in a string.
    
    Args:
        s: Input string
    Returns:
        First non-repeating character or None if not found
        
    Examples:
        >>> find_first_non_repeating("leetcode")
        'l'
        >>> find_first_non_repeating("loveleetcode") 
        'v'
        >>> find_first_non_repeating("aabb")
        None
    """
    raise NotImplementedError("Not implemented")

# 2. Python-Specific Features
def measure_time(func):
    """
    Decorator that measures the execution time of a function.
    
    Examples:
        >>> @measure_time
        ... def slow_function():
        ...     time.sleep(1)
        ...     return "Done"
        >>> result = slow_function()
        Function 'slow_function' took 1.001 seconds to execute
        >>> result
        'Done'
        
        >>> @measure_time
        ... def fast_function():
        ...     return sum(range(1000))
        >>> result = fast_function() 
        Function 'fast_function' took 0.0001 seconds to execute
        >>> result
        499500
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise NotImplementedError("Not implemented")
    return wrapper

# 4. Error Handling and Testing
def read_json_file(filepath: str) -> Dict:
    """
    Read and parse a JSON file with error handling.
    
    Args:
        filepath: Path to the JSON file
    Returns:
        Parsed JSON data as dictionary
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    raise NotImplementedError("Not implemented")

# MNIST Related Tasks
def load_mnist():
    """
    Load and preprocess the MNIST dataset.
    
    Returns:
        train_loader, test_loader
    """
    raise NotImplementedError("Not implemented")

def visualize_mnist_samples(samples: int = 9):
    """
    Visualize random MNIST samples.
    
    Args:
        samples: Number of samples to visualize
    """
    raise NotImplementedError("Not implemented")

class SimpleCNN(nn.Module):
    # The architecture is not important, just a simple CNN
    # Modify as you see fit
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define your CNN architecture here
        pass

    def forward(self, x):
        # Implement the forward pass
        raise NotImplementedError("Not implemented")

def train_model(model: nn.Module, train_loader: DataLoader, epochs: int = 5):
    """
    Train the CNN model on MNIST.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        epochs: Number of training epochs
    """
    raise NotImplementedError("Not implemented")
