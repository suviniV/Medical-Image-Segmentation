import os
import time
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def fcn_cat_ce(pred, target):
    """
    Fully Convolutional Network Categorical Cross Entropy Loss
    
    Args:
        pred: Model predictions (N, C, H, W)
        target: Ground truth labels (N, H, W)
    Returns:
        Cross entropy loss value
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).long()
        
    # Ensure pred is in right shape (N, C, H, W)
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    
    # Ensure target is in right shape (N, H, W)
    if target.dim() == 2:
        target = target.unsqueeze(0)
        
    return F.cross_entropy(pred, target)

def categorical_crossentropy(pred, target, axis=1):
    """
    Categorical Cross Entropy Loss (wrapper for compatibility)
    """
    return fcn_cat_ce(pred, target)

# Alias for backward compatibility
cat_ce = categorical_crossentropy

def tensor_softmax(x, axis=1):
    """PyTorch compatible softmax function"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return F.softmax(x, dim=axis)

class TransFun:
    """Transfer function wrapper for compatibility"""
    def __init__(self, fun, *args, **kwargs):
        self.fun = fun
        self.args = args
        self.kwargs = kwargs
        
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return self.fun(x, *self.args, **self.kwargs)

def tanimoto(pred, target):
    """
    Compute Tanimoto coefficient (also known as Jaccard index)
    pred: prediction tensor
    target: ground truth tensor
    """
    smooth = 1e-5
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def dice_coefficient(pred, target):
    """
    Compute Dice coefficient
    pred: prediction tensor
    target: ground truth tensor
    """
    smooth = 1e-5
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

class PocketTrainer:
    """Legacy trainer for backward compatibility"""
    def __init__(self, model, data, stop, pause, score_fun, report_fun,
                 evaluate=True, test=False, batchnorm=False,
                 model_code=None, n_report=None):
        self.model = model
        self.data = data
        self.stop = stop
        self.pause = pause
        self.score_fun = score_fun
        self.report_fun = report_fun
        self.best_pars = None
        self.best_loss = float('inf')
        self.runtime = 0
        self.evaluate = evaluate
        self.test = test
        self.losses = []
        self.test_performance = []
        self.model_code = model_code
        self.n_epochs_done = 0
        self.n_iters_done = 0
        self.n_report = n_report
        self.using_bn = batchnorm

    def fit(self):
        print('Starting training with legacy trainer...')
        start = time.time()
        info = {'n_iter': 0}
        
        try:
            while not self.stop(info):
                # Training step using model's existing train function
                self.model.train(*self.data['train'])
                
                if self.pause(info):
                    # Calculate losses
                    train_loss = self.score_fun(self.model.score, *self.data['train'])
                    
                    if self.evaluate:
                        val_loss = self.score_fun(self.model.score, *self.data['val'])
                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            self.best_pars = self.model.parameters.data.copy()
                        self.losses.append((train_loss, val_loss))
                    else:
                        self.losses.append(train_loss)
                    
                    info.update({
                        'n_iter': info['n_iter'] + 1,
                        'loss': train_loss,
                        'val_loss': val_loss if self.evaluate else None,
                        'best_loss': self.best_loss,
                        'runtime': time.time() - start
                    })
                    
                    self.report_fun(info)
                
        except KeyboardInterrupt:
            print('Training interrupted by user.')
        
        self.runtime = time.time() - start
        print(f'Training completed in {self.runtime:.2f} seconds')

    def demo(self, predict, image, gt, size_reduction, im_name='test.png'):
        output_h = image.shape[2] - size_reduction
        output_w = image.shape[3] - size_reduction
        output_d = image.shape[4] if len(image.shape) > 4 else 1
        
        pred = predict(image)
        if hasattr(pred, 'as_numpy_array'):
            pred = pred.as_numpy_array()
        elif isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
            
        pred = pred.reshape(output_h, output_w, output_d, -1)
        gt = gt.reshape(output_h, output_w, output_d, -1)
        
        # Calculate dice coefficient
        dice = np.sum(2 * pred * gt) / (np.sum(np.square(pred)) + np.sum(np.square(gt)))
        
        # Visualization
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.title('Input Image')
        plt.imshow(image[0, 0], cmap='gray')
        
        plt.subplot(132)
        plt.title('Prediction')
        plt.imshow(pred[:,:,0].argmax(-1), cmap='gray')
        
        plt.subplot(133)
        plt.title('Ground Truth')
        plt.imshow(gt[:,:,0].argmax(-1), cmap='gray')
        
        plt.savefig(im_name)
        plt.close()
        
        return dice

class ModernPocketTrainer:
    def __init__(self, model, data, stop=None, pause=None, 
                 score_fun=None, report_fun=None, evaluate=True,
                 test=False, batchnorm=False, model_code=None, 
                 n_report=None, device=None):
        self.model = model
        self.data = data
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Training parameters
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        
        # Legacy compatibility
        self.stop = stop
        self.pause = pause
        self.score_fun = score_fun
        self.report_fun = report_fun
        self.evaluate = evaluate
        self.test = test
        self.using_bn = batchnorm
        self.model_code = model_code
        self.n_report = n_report
        
        # State tracking
        self.best_loss = float('inf')
        self.best_state = None
        self.losses = []
        self.test_performance = []
        
    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        
        x = torch.from_numpy(x).float().to(self.device)
        y = torch.from_numpy(y).long().to(self.device)
        
        output = self.model(x)
        loss = self.criterion(output, y)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, x, y):
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(self.device)
            y = torch.from_numpy(y).long().to(self.device)
            
            output = self.model(x)
            loss = self.criterion(output, y)
            
        return loss.item()
    
    def fit(self):
        print('Starting training...')
        train_x, train_y = self.data['train']
        valid_x, valid_y = self.data.get('val', self.data.get('valid', (None, None)))
        
        info = {'n_iter': 0}
        start_time = time.time()
        
        try:
            while not self.stop(info) if self.stop else True:
                # Training
                train_loss = self.train_step(train_x, train_y)
                
                # Validation if available
                if valid_x is not None and valid_y is not None:
                    val_loss = self.validate(valid_x, valid_y)
                else:
                    val_loss = train_loss
                
                # Update info
                info['n_iter'] += 1
                info['loss'] = train_loss
                info['val_loss'] = val_loss
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_state = self.model.state_dict().copy()
                
                self.losses.append((train_loss, val_loss))
                
                # Report progress
                if self.report_fun:
                    self.report_fun(info)
                else:
                    print(f'Iteration {info["n_iter"]} - '
                          f'Train Loss: {train_loss:.4f} - '
                          f'Val Loss: {val_loss:.4f}')
                
                # Pause if needed
                if self.pause and self.pause(info):
                    break
                    
        except KeyboardInterrupt:
            print('Training interrupted by user.')
        
        training_time = time.time() - start_time
        print(f'Training completed in {training_time:.2f} seconds')
        
        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
    
    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(self.device)
            output = self.model(x)
            return output.cpu().numpy()
    
    def demo(self, predict, image, gt, size_reduction, im_name='test.png'):
        pred = self.predict(image)
        
        # Calculate dice coefficient
        dice = np.sum(2 * pred * gt) / (np.sum(np.square(pred)) + np.sum(np.square(gt)))
        
        # Visualization
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.title('Input Image')
        plt.imshow(image[0, 0], cmap='gray')
        
        plt.subplot(132)
        plt.title('Prediction')
        plt.imshow(pred[0].argmax(0), cmap='gray')
        
        plt.subplot(133)
        plt.title('Ground Truth')
        plt.imshow(gt[0].argmax(0), cmap='gray')
        
        plt.savefig(im_name)
        plt.close()
        
        return dice
    
    def save_checkpoint(self, path):
        if self.best_state is None:
            print('No model to save.')
            return
            
        checkpoint = {
            'model_state': self.best_state,
            'best_loss': self.best_loss,
            'optimizer_state': self.optimizer.state_dict(),
            'losses': self.losses,
            'test_performance': self.test_performance
        }
        
        torch.save(checkpoint, path)
        print(f'Model saved to {path}')
    
    def load_checkpoint(self, path):
        if not os.path.exists(path):
            print('No checkpoint found.')
            return
            
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.best_loss = checkpoint['best_loss']
        self.losses = checkpoint.get('losses', [])
        self.test_performance = checkpoint.get('test_performance', [])
        print(f'Model loaded from {path}')