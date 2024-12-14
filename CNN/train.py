"""
This is the script used for training one of the neural networks
defined in model_defs.py
The usage is:

python train.py [model_id] [dataset] [save_folder] [n_epoch] -ch (True/False)

[model_id] is the id of the model used in model_defs.py
[dataset] is the name of a dataset from data/datasets
[save_folder] our results will be saved to models/[save_folder]
[n_epoch] We will iterate over the entire data set [n_epoch] many times

-ch: Flag indicating whether we want to start training from an earlier checkpoint
     WARNING: checkpoints are specific to the model_id and not to the experiment.
              If you have two different experiments using the same model_id running
              in parallel, their checkpoints will be in conflict.

In our paper, we trained our network using:

python train.py fcn_rffc4 brats_fold0 brats_fold0 600 -ch False
"""

#import break_handling
try:
    import pickle
except ImportError:
    print("Warning: pickle not found")
import json
import os
import datetime
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

#import matplotlib.pyplot as plt
import numpy as np
try:
    import gnumpy.pygpu as gnumpy
except ImportError:
    print("Warning: gnumpy not found, using numpy instead")
    gnumpy = np
import h5py

try:
    import climin.stops
    has_climin = True
except ImportError:
    print("Warning: climin not found, using basic training loop")
    has_climin = False

try:
    from breze.learn.trainer import report
    has_breze = True
except ImportError:
    print("Warning: breze not found, using basic reporting")
    has_breze = False

import ash
from ash import ModernPocketTrainer
from model_defs import get_model



from conv3d.model import SequentialModel
def report_fallback(msg):
    """Basic reporting function when breze is not available"""
    print(msg)

if not has_breze:
    report = report_fallback

def make_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(description='Train a segmentation model')
    parser.add_argument('--model', type=str, default='sequential',
                      help='model type (default: sequential)')
    parser.add_argument('--data', type=str, default='brats20_small',
                      help='dataset name (default: brats20_small)')
    parser.add_argument('--epochs', type=int, default=5,
                      help='number of epochs (default: 5)')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='path to checkpoint to resume from')
    parser.add_argument('--save_dir', type=str, 
                      default='/content/drive/MyDrive/RT-MyCopy/CNNbasedMedicalSegmentation/models',
                      help='directory to save models')
    return parser

def retrieve_data(data_code):
    """Load and preprocess the data"""
    print(f'Loading dataset: {data_code}')
    
    # Construct path to dataset in Google Drive
    base_dir = '/content/drive/MyDrive/RT-MyCopy/CNNbasedMedicalSegmentation'
    data_dir = os.path.join(base_dir, 'data/datasets')
    
    if data_code == 'brats20_small':
        data_path = os.path.join(data_dir, 'brats20_small.hdf5')
    else:
        data_path = os.path.join(data_dir, data_code + '.hdf5')
    
    print(f'Looking for dataset at: {data_path}')
    if not os.path.exists(data_path):
        raise ValueError(f'Dataset not found: {data_path}')
    
    # Load data from HDF5 file
    with h5py.File(data_path, 'r') as f:
        train_x = torch.FloatTensor(f['train_x'][:])
        train_y = torch.FloatTensor(f['train_y'][:])
        valid_x = torch.FloatTensor(f['valid_x'][:])
        valid_y = torch.FloatTensor(f['valid_y'][:])
        test_x = torch.FloatTensor(f['test_x'][:])
        test_y = torch.FloatTensor(f['test_y'][:])
    
    print(f'Data shapes:')
    print(f'Train: {train_x.shape}, {train_y.shape}')
    print(f'Valid: {valid_x.shape}, {valid_y.shape}')
    print(f'Test: {test_x.shape}, {test_y.shape}')
    
    return {
        'train_x': train_x,
        'train_y': train_y,
        'valid_x': valid_x,
        'valid_y': valid_y,
        'test_x': test_x,
        'test_y': test_y
    }

def build_model(model_code, checkpoint=None, info=None):
    """Build the model"""
    if info is None:
        info = {}
    
    # Default parameters for downsampled data
    info.setdefault('image_height', 120)  # Downsampled from 240
    info.setdefault('image_width', 120)   # Downsampled from 240
    info.setdefault('image_depth', 78)    # Downsampled from 155
    info.setdefault('n_channels', 4)      # 4 modalities (FLAIR, T1, T1ce, T2)
    info.setdefault('n_output', 2)        # Binary segmentation
    info.setdefault('batch_size', 2)      # Smaller batch size for memory
    info.setdefault('max_epochs', 5)
    
    if model_code == 'sequential':
        model = SequentialModel(**info)
    else:
        raise ValueError(f'Unknown model code: {model_code}')
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))
    
    return model

def setup_training(model_code, data_code, checkpoint=False, max_passes=5):
    """Setup the training process"""
    # Load data
    data = retrieve_data(data_code)
    
    # Build model
    model = build_model(model_code, checkpoint)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Move data to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_x = data['train_x'].to(device)
    train_y = data['train_y'].to(device)
    valid_x = data['valid_x'].to(device)
    valid_y = data['valid_y'].to(device)
    
    return model, criterion, optimizer, (train_x, train_y, valid_x, valid_y)

def train_epoch(model, train_data, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    train_x, train_y = train_data
    total_loss = 0
    batch_size = model.batch_size
    n_samples = 0
    
    for i in range(0, len(train_x), batch_size):
        # Get batch and move to device
        batch_x = train_x[i:i+batch_size].to(device)  # [B, C, H, W, D]
        batch_y = train_y[i:i+batch_size].to(device)  # [B, H, W, D]
        current_batch_size = len(batch_x)
        
        print(f'Input shape: {batch_x.shape}')
        
        # Forward pass
        outputs = model(batch_x)  # [B, n_classes, H, W, D]
        print(f'Model output shape: {outputs.shape}')
        
        # Reshape for CrossEntropyLoss:
        B, C, H, W, D = outputs.shape
        outputs = outputs.view(B * H * W * D, C)  # [B*H*W*D, C]
        batch_y = batch_y.view(-1).long()  # [B*H*W*D]
        
        print(f'Reshaped output shape: {outputs.shape}')
        print(f'Target shape: {batch_y.shape}')
        
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item() * current_batch_size
        n_samples += current_batch_size
        
        if (i // batch_size) % 10 == 0:
            print(f'Batch {i//batch_size}: loss = {loss.item():.4f}')
    
    return total_loss / n_samples

def validate(model, valid_data, criterion, device):
    """Validate the model"""
    # Check if validation data is empty
    if len(valid_data[0]) == 0:
        print("No validation data available. Skipping validation.")
        return 0.0  # Return 0 loss if no validation data
    
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for batch_x, batch_y in zip(valid_data[0], valid_data[1]):
            # Move data to device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Get current batch size
            current_batch_size = batch_x.size(0)
            
            # Forward pass
            outputs = model(batch_x)
            
            # Reshape for loss calculation
            B, C, H, W, D = outputs.shape
            outputs = outputs.view(B * H * W * D, C)
            batch_y = batch_y.view(-1).long()
            
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * current_batch_size
            n_samples += current_batch_size
    
    # Only return average loss if samples were processed
    return total_loss / n_samples if n_samples > 0 else 0.0

def start_training(model_code, data_code, checkpoint=None, epochs=100, save_dir='model_new_sample'):
    """Start the training process"""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print(f'Loading dataset: {data_code}')
    data = retrieve_data(data_code)
    
    # Move data to device
    data['train_x'] = data['train_x'].to(device)
    data['train_y'] = data['train_y'].to(device)
    data['valid_x'] = data['valid_x'].to(device)
    data['valid_y'] = data['valid_y'].to(device)
    
    # Calculate class weights for balanced loss
    train_y = data['train_y']
    n_zeros = torch.sum(train_y == 0).item()
    n_ones = torch.sum(train_y == 1).item()
    total = n_zeros + n_ones
    weights = torch.FloatTensor([total/(2*n_zeros), total/(2*n_ones)]).to(device)
    print(f'\nClass distribution:')
    print(f'Class 0 (non-tumor): {n_zeros/total*100:.2f}%')
    print(f'Class 1 (tumor): {n_ones/total*100:.2f}%')
    print(f'Using weighted loss with weights: {weights.cpu().numpy()}')
    
    # Create model
    model = build_model(model_code)
    model = model.to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Load checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    if checkpoint:
        if os.path.exists(checkpoint):
            print(f'Loading checkpoint: {checkpoint}')
            state = torch.load(checkpoint)
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            start_epoch = state['epoch']
            best_loss = state['loss']
        else:
            print(f'Warning: Checkpoint not found at {checkpoint}')
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        
        # Train
        train_loss = train_epoch(model, (data['train_x'], data['train_y']), criterion, optimizer, device)
        print(f'Training Loss: {train_loss:.4f}')
        
        # Validate
        val_loss = validate(model, (data['valid_x'], data['valid_y']), criterion, device)
        print(f'Validation Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
            }
            os.makedirs(save_dir, exist_ok=True)
            torch.save(state, os.path.join(save_dir, f'{model_code}_best.pth'))
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': val_loss,
            }
            os.makedirs(save_dir, exist_ok=True)
            torch.save(state, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print('\nTraining complete!')
    print(f'Best validation loss: {best_loss:.4f}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='sequential')
    parser.add_argument('--data', default='brats20_small')
    parser.add_argument('--checkpoint')
    parser.add_argument('--epochs', type=int, default=100)  # Back to 100 epochs
    parser.add_argument('--save_dir', default='models')
    args = parser.parse_args()

    start_training(
        model_code=args.model,
        data_code=args.data,
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        save_dir=args.save_dir
    )

if __name__ == '__main__':
    main()