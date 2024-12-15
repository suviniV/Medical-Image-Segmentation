# if __name__ == "__main__":
#     # Updated path to point to your new model
#     model_path = '/content/drive/MyDrive/RT-MyCopy/CNNbasedMedicalSegmentation/models/brats20_new/sequential_best.pth'
#     print_model_metrics(model_path)
import torch
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from train import build_model, retrieve_data

def visualize_predictions(image, target, pred, sample_idx, save_dir='visualization_results'):
    """
    Visualize input image, ground truth, and prediction
    
    Args:
    - image: input image (4D tensor or numpy array)
    - target: ground truth segmentation mask (3D tensor or numpy array)
    - pred: predicted segmentation mask (3D tensor or numpy array)
    - sample_idx: index of the current sample
    - save_dir: directory to save visualization results
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Print debug information about input shapes
    print(f"Debug - Sample {sample_idx}:")
    print(f"Image shape: {image.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Pred shape: {pred.shape}")
    
    # Select the first modality for visualization (assuming FLAIR)
    if image.ndim == 4:
        image = image[0]  # Select first modality
    
    # Handle prediction shape
    if pred.ndim == 4:
        if pred.shape[0] > 1:
            pred = pred[1]  # Take the second channel (foreground)
        else:
            pred = pred.squeeze()
    
    # Handle target shape
    target = target.squeeze()
    
    # Ensure both pred and target are 2D or 3D
    if pred.ndim not in [2, 3] or target.ndim not in [2, 3]:
        print(f"Warning: Unexpected array dimensions. Pred dim: {pred.ndim}, Target dim: {target.ndim}")
        return
    
    # Ensure binary masks
    target = (target > 0.5).astype(np.float32)
    pred = (pred > 0.5).astype(np.float32)
    
    # Transpose if needed to match image shape
    if pred.shape != image.shape:
        try:
            pred = np.transpose(pred, (1, 0, 2))
        except Exception as e:
            print(f"Could not transpose prediction. Error: {e}")
            return
    
    # Select middle slice for 3D volumes
    slice_idx = image.shape[-1] // 2
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(131)
    plt.imshow(image[:, :, slice_idx], cmap='gray')
    plt.title(f'Sample {sample_idx} - Input Image')
    plt.axis('off')
    
    # Ground Truth
    plt.subplot(132)
    plt.imshow(image[:, :, slice_idx], cmap='gray', alpha=0.5)
    plt.imshow(target[:, :, slice_idx], cmap='Reds', alpha=0.5)
    plt.title(f'Sample {sample_idx} - Ground Truth')
    plt.axis('off')
    
    # Prediction
    plt.subplot(133)
    plt.imshow(image[:, :, slice_idx], cmap='gray', alpha=0.5)
    plt.imshow(pred[:, :, slice_idx], cmap='Reds', alpha=0.5)
    plt.title(f'Sample {sample_idx} - Prediction')
    plt.axis('off')
    
    # Save the plot
    save_path = os.path.join(save_dir, f'sample_{sample_idx}_visualization.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"Visualization for Sample {sample_idx} saved to {save_path}")

def calculate_performance_metrics(pred, target):
    """Calculate comprehensive performance metrics for CNN models"""
    smooth = 1e-6
    
    # If prediction has multiple channels, take the argmax or second channel
    if pred.ndim == 4 and pred.shape[0] > 1:
        pred = pred[1]  # Take the second channel (assuming it's the foreground)
    
    # Ensure both pred and target are 3D
    pred = pred.squeeze()
    target = target.squeeze()
    
    # Ensure both are binary
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    
    # Flatten
    pred = pred.flatten()
    target = target.flatten()
    
    # True Positives, False Positives, False Negatives
    tp = (pred * target).sum()
    fp = pred.sum() - tp
    fn = target.sum() - tp
    tn = len(pred) - tp - fp - fn
    
    # Calculate metrics
    dice = (2. * tp + smooth) / (pred.sum() + target.sum() + smooth)
    iou = (tp + smooth) / (fp + tp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    accuracy = (tp + tn + smooth) / (len(pred) + smooth)
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy)
    }

# ... [rest of the code remains the same as in the previous submission]

def count_parameters(model):
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, input_shape=(1, 4, 120, 120, 78), num_runs=100):
    """Measure average inference time using dummy input"""
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)
    times = []
    model.eval()
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            times.append(time.time() - start)
    
    return {
        'avg_inference_time': np.mean(times),
        'std_inference_time': np.std(times)
    }

def print_model_metrics(model_path):
    """Print comprehensive metrics for the model"""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Load model
    model = build_model('sequential')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # If checkpoint contains full training state
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If checkpoint is just model weights
        model.load_state_dict(checkpoint)
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load data
    data = retrieve_data('brats20_small')
    test_x, test_y = data['test_x'], data['test_y']
    
    # Prepare for evaluation
    model.eval()
    all_metrics = []
    
    print(f"\nEvaluation Metrics Summary")
    print("==============================")
    print(f"Number of test samples: {len(test_x)}")
    
    # Create visualization directory
    visualization_dir = 'model_predictions'
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Compute metrics for each sample
    with torch.no_grad():
        for i, (x, y) in enumerate(zip(test_x, test_y)):
            x = x.unsqueeze(0).to(device)
            
            # Predict
            pred = model(x)
            
            # Convert to numpy and ensure correct shape
            pred_np = pred.sigmoid().cpu().numpy().squeeze()
            x_np = x.cpu().numpy().squeeze()
            y_np = y.cpu().numpy()
            
            # Ensure both pred and target are binary
            pred_binary = (pred_np > 0.5).astype(np.float32)
            target_binary = (y_np > 0.5).astype(np.float32)
            
            # Visualize predictions
            visualize_predictions(x_np, y_np, pred_binary, i, save_dir=visualization_dir)
            
            # Calculate metrics
            metrics = calculate_performance_metrics(pred_binary, target_binary)
            all_metrics.append(metrics)
    
    # Compute average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics]) 
        for key in all_metrics[0].keys()
    }
    
    # Print average metrics
    print("\nAverage Metrics:")
    print("--------------------")
    for key, value in avg_metrics.items():
        print(f"{key.capitalize()} Score: {value:.4f}")
    
    # Per-sample metrics
    print("\nPer-sample Metrics:")
    print("--------------------")
    for i, metrics in enumerate(all_metrics):
        print(f"Sample {i}:")
        for key, value in metrics.items():
            print(f"  {key.capitalize()} Score: {value:.4f}")
    
    # Additional model information
    print("\nModel Information:")
    print("--------------------")
    print(f"Total Parameters: {count_parameters(model)}")
    
    # Inference time
    inference_times = measure_inference_time(model)
    print(f"Avg Inference Time: {inference_times['avg_inference_time']:.4f} seconds")
    print(f"Std Inference Time: {inference_times['std_inference_time']:.4f} seconds")


if __name__ == "__main__":
    # Update this path to your model's checkpoint
    model_path = '/content/drive/MyDrive/RT-MyCopy/CNNbasedMedicalSegmentation/models/brats20_new/sequential_best.pth'
    print_model_metrics(model_path)