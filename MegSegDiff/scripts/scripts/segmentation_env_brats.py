import os
import argparse
import torch as th
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms
import sys
sys.path.append(".")
from guided_diffusion.guided_diffusion import dist_util, logger
from guided_diffusion.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.guided_diffusion.bratsloader import BRATSDataset3D
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.functional.classification import dice as dice_fn
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import time


def compute_hd95(pred, target):
    if not np.any(pred) or not np.any(target):
        return float('inf')
    
    pred_points = np.array(np.where(pred)).T
    target_points = np.array(np.where(target)).T
    
    d1, _, _ = directed_hausdorff(pred_points, target_points)
    d2, _, _ = directed_hausdorff(target_points, pred_points)
    return max(d1, d2)


def create_argparser():
    defaults = dict(
        data_dir="/content/drive/MyDrive/RT CW/BraTS2020/Testing/BraTS20_Training_121",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        fp16_scale_growth=1e-3,
        seed=101,
        model_path="/content/drive/MyDrive/RT_Codes_Files/MedSeggDiff/training_results2/savedmodel001000.pt",
        out_dir="./evaluation",
        image_size=256,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=50,  # Reduced from 1000 to speed up testing
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
        clip_denoised=True,
        use_ddim=False,
        in_ch=4,
        out_ch=4,
        num_heads_channels=-1,
        num_head_channels=-1,
        channel_mult="",
        num_channels_mult=2,
        dims=2,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    args = create_argparser().parse_args()
    logger.configure(dir=args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Loading dataset...")
    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size))
    ])
    
    ds = BRATSDataset3D(args.data_dir, transform_test, test_flag=False)
    
    total_slices = len(ds)
    start_slice = total_slices // 2 - 5
    end_slice = start_slice + 10
    
    indices = list(range(start_slice, end_slice))
    subset_ds = th.utils.data.Subset(ds, indices)
    
    print(f"Total dataset size: {total_slices} slices")
    print(f"Testing slices {start_slice} to {end_slice-1}")
    
    datal = th.utils.data.DataLoader(
        subset_ds,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    print(f"Loading model from {args.model_path}...")
    state_dict = th.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    iou_metric = BinaryJaccardIndex().to(device)
    
    dice_scores = []
    iou_scores = []
    hd95_scores = []
    
    print("\nStarting inference...")
    print("-" * 50)
    
    with th.no_grad():
        for i, (image, label, path) in enumerate(datal):
            try:
                slice_id = path[0].split("_slice")[-1].split(".")[0]
                print(f"\nProcessing slice {slice_id} ({i+1}/{len(datal)})")
                
                image = image.to(device)
                label = label.to(device)
                
                noise = th.randn_like(image[:, :1, ...])
                x = th.cat((image, noise), dim=1)
                                
                print("Starting diffusion sampling...")
                start_time = time.time()
                
                sample_fn = diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                sample, _, _, _, _ = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    x,
                    clip_denoised=args.clip_denoised,
                    model_kwargs={},
                )
                
                elapsed = time.time() - start_time
                print(f"Diffusion sampling completed in {elapsed:.1f}s")
                
                # Save prediction immediately
                pred = (sample[:, -1:, :, :] > 0).float()
                
                # Calculate metrics
                pred = (pred > 0.5).int().to(device)  # Convert to binary integer tensor
                gt = gt.int().to(device)  # Convert to integer tensor
                dice_score = dice_fn(pred.squeeze(), gt.squeeze())  # Ensure binary tensors
                iou_score = iou_metric(pred, gt)
                
                pred_np = pred.cpu().numpy().squeeze()
                gt_np = gt.cpu().numpy().squeeze()
                
                if np.any(pred_np) and np.any(gt_np):
                    hd95_score = compute_hd95(pred_np, gt_np)
                    hd95_scores.append(hd95_score)
                
                dice_scores.append(dice_score.item())
                iou_scores.append(iou_score.item())
                
                print(f"Metrics for slice {slice_id}:")
                print(f"  Dice score: {dice_score:.4f}")
                print(f"  IoU score: {iou_score:.4f}")
                if len(hd95_scores) > 0:
                    print(f"  HD95: {hd95_scores[-1]:.4f}")
                print("-" * 50)
                
            except Exception as e:
                import traceback
                print(f"Error processing batch {i}: {str(e)}")
                print("Full traceback:")
                print(traceback.format_exc())
                continue
    
    # Calculate and save final metrics
    avg_dice = np.mean(dice_scores) if dice_scores else 0
    avg_iou = np.mean(iou_scores) if iou_scores else 0
    avg_hd95 = np.mean(hd95_scores) if hd95_scores else float('inf')
    
    print("\nTesting completed!")
    print("=" * 50)
    print(f"Processed {len(dice_scores)} slices")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")
    if hd95_scores:
        print(f"Average HD95: {avg_hd95:.4f}")
    print("=" * 50)
    
    # Save metrics to file
    with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
        f.write("Evaluation Metrics Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Number of slices evaluated: {len(dice_scores)}\n\n")
        f.write("Average Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Dice Score: {avg_dice:.4f}\n")
        f.write(f"IoU Score: {avg_iou:.4f}\n")
        if hd95_scores:
            f.write(f"HD95: {avg_hd95:.4f}\n")
        f.write("\n")
        f.write("Per-slice Metrics:\n")
        f.write("-" * 20 + "\n")
        for i, (dice, iou) in enumerate(zip(dice_scores, iou_scores)):
            f.write(f"\nSlice {i}:\n")
            f.write(f"  Dice Score: {dice:.4f}\n")
            f.write(f"  IoU Score: {iou:.4f}\n")
            if i < len(hd95_scores):
                f.write(f"  HD95: {hd95_scores[i]:.4f}\n")

if __name__ == "__main__":
    main()