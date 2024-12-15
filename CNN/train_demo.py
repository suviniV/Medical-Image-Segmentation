import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import os
from train import build_model

def retrieve_demo_data(data_code, num_samples=20):
    """Load and preprocess a small subset of data for demo"""
    print(f'Loading dataset: {data_code}')
    
    # Updated path to match Google Drive structure
    data_path = f'/content/drive/MyDrive/RT-MyCopy/CNNbasedMedicalSegmentation/data/datasets/{data_code}.hdf5'
    print(f'Looking for dataset at: {data_path}')
    
    with h5py.File(data_path, 'r') as f:
        # Load only the first num_samples samples
        train_x = torch.FloatTensor(f['train_x'][:num_samples])
        train_y = torch.FloatTensor(f['train_y'][:num_samples])
        valid_x = torch.FloatTensor(f['valid_x'][:5])  # Take only 5 validation samples
        valid_y = torch.FloatTensor(f['valid_y'][:5])
        test_x = torch.FloatTensor(f['test_x'][:5])    # Take only 5 test samples
        test_y = torch.FloatTensor(f['test_y'][:5])
    
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

def train_epoch(model, train_data, criterion, optimizer, device):
    """Train for one epoch with higher learning rate"""
    model.train()
    total_loss = 0
    
    # Get training data
    inputs = train_data['train_x']
    targets = train_data['train_y']
    
    # Process each sample
    for i in range(len(inputs)):
        # Get current sample
        input_sample = inputs[i:i+1]
        target = targets[i]
        
        # Forward pass
        optimizer.zero_grad()
        output = model(input_sample)
        
        # Reshape output and target for loss calculation
        output = output.view(-1, 2)  # Reshape to (N, 2) where N is the number of pixels
        target = target.view(-1)     # Flatten target
        
        # Compute loss
        loss = criterion(output, target.long())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(inputs)

def validate(model, valid_data, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        inputs = valid_data['valid_x']
        targets = valid_data['valid_y']
        
        for i in range(len(inputs)):
            input_sample = inputs[i:i+1]
            target = targets[i]
            
            output = model(input_sample)
            
            # Reshape output and target for loss calculation
            output = output.view(-1, 2)  # Reshape to (N, 2)
            target = target.view(-1)     # Flatten target
            
            loss = criterion(output, target.long())
            total_loss += loss.item()
    
    return total_loss / len(inputs)

def start_demo_training(model_code='sequential', data_code='brats20_new', epochs=500, save_dir='models/demo'):
    """Start the training process optimized for demo"""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load small dataset
    data = retrieve_demo_data(data_code, num_samples=10)  # Reduced to 10 samples for more focused learning
    
    # Move data to device
    for key in data:
        data[key] = data[key].to(device)
    
    # Calculate class weights with stronger weighting
    train_y = data['train_y']
    n_zeros = torch.sum(train_y == 0).item()
    n_ones = torch.sum(train_y == 1).item()
    total = n_zeros + n_ones
    # Stronger class weighting
    weights = torch.FloatTensor([1.0, 3.0]).to(device)  # Give more weight to tumor class
    
    print(f'\nClass distribution:')
    print(f'Class 0 (non-tumor): {n_zeros/total*100:.2f}%')
    print(f'Class 1 (tumor): {n_ones/total*100:.2f}%')
    print(f'Using weights: {weights.cpu().numpy()}')
    
    # Create model
    model = build_model(model_code)
    model = model.to(device)
    
    # Setup training with modified parameters
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    best_loss = float('inf')
    patience = 50  # Increased patience
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train and validate
        train_loss = train_epoch(model, data, criterion, optimizer, device)
        valid_loss = validate(model, data, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(valid_loss)
        
        # Save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }
            torch.save(checkpoint, os.path.join(save_dir, 'sequential_best.pth'))
            print(f'Saved new best model with loss: {best_loss:.4f}')
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Valid Loss: {valid_loss:.4f}')
            print(f'Best Loss: {best_loss:.4f}')
            print(f'Current Learning Rate: {optimizer.param_groups[0]["lr"]}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

if __name__ == '__main__':
    # Start training with demo settings
    start_demo_training()
