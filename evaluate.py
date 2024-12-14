
import torch
import numpy as np
import matplotlib.pyplot as plt
from train import retrieve_data, build_model
import h5py
import os

def calculate_metrics(pred, target):
    """Calculate Dice score and IoU"""
    smooth = 1e-6
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    iou = (intersection + smooth) / (pred.sum() + target.sum() - intersection + smooth)
    
    return dice, iou

def visualize_slice(image, seg, pred, slice_idx, save_path=None):
    """Visualize a slice of the image, ground truth, and prediction"""
    plt.figure(figsize=(15, 5))
    
    # Original image (FLAIR modality)
    plt.subplot(131)
    plt.imshow(image[0, :, :, slice_idx], cmap='gray')
    plt.title('FLAIR Image')
    plt.axis('off')
    
    # Ground truth
    plt.subplot(132)
    plt.imshow(seg[:, :, slice_idx], cmap='Reds')  # Changed from 'red' to 'Reds'
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Prediction
    plt.subplot(133)
    plt.imshow(pred[:, :, slice_idx], cmap='Reds')  # Changed from 'red' to 'Reds'
    plt.title('Prediction')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def evaluate_model(model_path, data_code='brats20_small', save_dir='results'):
    """Evaluate the model and generate visualizations"""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    model = build_model('sequential')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    data = retrieve_data(data_code)
    test_x = data['test_x'].to(device)
    test_y = data['test_y'].to(device)
    
    # Metrics storage
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for i in range(len(test_x)):
            # Get single sample
            image = test_x[i:i+1]
            target = test_y[i]
            
            # Forward pass
            output = model(image)
            pred = torch.argmax(output, dim=1)[0]
            
            # Calculate metrics
            dice, iou = calculate_metrics(pred.cpu().numpy(), target.cpu().numpy())
            dice_scores.append(dice)
            iou_scores.append(iou)
            
            # Visualize middle slice
            mid_slice = pred.shape[-1] // 2
            visualize_slice(
                image[0].cpu().numpy(),
                target.cpu().numpy(),
                pred.cpu().numpy(),
                mid_slice,
                save_path=os.path.join(save_dir, f'sample_{i}_slice_{mid_slice}.png')
            )
            
            # Print individual sample metrics
            print(f'\nSample {i}:')
            print(f'Dice Score: {dice:.4f}')
            print(f'IoU Score: {iou:.4f}')
    
    # Print overall results
    print('\nOverall Evaluation Results:')
    print(f'Average Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}')
    print(f'Average IoU Score: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}')
    
    return dice_scores, iou_scores

if __name__ == '__main__':
    model_path = '/content/drive/MyDrive/RT-MyCopy/CNNbasedMedicalSegmentation/models/brats20_new/sequential_best.pth'
    evaluate_model(model_path, data_code='brats20_new')  # Changed from default 'brats20_small'