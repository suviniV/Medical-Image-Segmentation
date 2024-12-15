import argparse
import os
import random
import sys
sys.path.append(".")
import numpy as np
import torch as th
import torchvision.utils as vutils
from guided_diffusion.guided_diffusion import dist_util, logger
from guided_diffusion.guided_diffusion.bratsloader import BRATSDataset3D
from guided_diffusion.guided_diffusion.isicloader import ISICDataset
from guided_diffusion.guided_diffusion.custom_dataset_loader import CustomDataset
from guided_diffusion.guided_diffusion.utils import staple
from guided_diffusion.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from torchvision import transforms
from collections import OrderedDict

# Set random seeds for reproducibility
seed = 10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    """Normalize the image and ensure it is in 3D format (CxHxW)."""
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)

    # Convert single-channel or RGBA to RGB
    if normalized_img.shape[0] == 1:  # Single-channel to RGB
        normalized_img = normalized_img.repeat(3, 1, 1)  # Repeat to create RGB
    elif normalized_img.shape[0] == 4:  # RGBA to RGB
        normalized_img = normalized_img[:3, :, :]  # Drop the alpha channel

    return normalized_img


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    # Dataset selection
    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()]
        transform_test = transforms.Compose(tran_list)
        ds = ISICDataset(args, args.data_dir, transform_test, mode='Test')
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size, args.image_size))]
        transform_test = transforms.Compose(tran_list)
        ds = BRATSDataset3D(args.data_dir, transform_test)
        args.in_ch = 5
    else:
        tran_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()]
        transform_test = transforms.Compose(tran_list)
        ds = CustomDataset(args, args.data_dir, transform_test, mode='Test')
        args.in_ch = 4

        
    # Extract a subset of slices
    total_slices = len(ds)
    start_slice = total_slices // 2 - 5  # Adjust as per requirement
    end_slice = start_slice + 10  # Define the range of slices to test
    indices = list(range(start_slice, end_slice))
    subset_ds = th.utils.data.Subset(ds, indices)

    print(f"Total dataset size: {total_slices} slices")
    print(f"Testing slices {start_slice} to {end_slice-1}")

    # Create DataLoader using the subset dataset
    datal = th.utils.data.DataLoader(
        subset_ds,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Continue with the rest of the code
    data = iter(datal)

    logger.log("Creating model and diffusion...")

    # Load model and diffusion
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # Load model weights
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # Continue processing the data as before
    for _ in range(len(data)):
        b, m, path = next(data) # Input batch, ground truth, and paths
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)  # Add noise channel

        # Extract single image and ground truth from batch
        single_b = b[0]  # Extract the first image (CxHxW)
        single_m = m[0] if m is not None else None  # Extract first ground truth mask

        print(f"Shape of single input tensor (b): {single_b.shape}")
        if single_m is not None:
            print(f"Shape of single ground truth tensor (m): {single_m.shape}")

        # Save input and ground truth images
        slice_ID = path[0].split("_")[-1].split(".")[0]
        vutils.save_image(
            visualize(single_b), fp=os.path.join(args.out_dir, f"{slice_ID}_input.png"), nrow=1, padding=10
        )
        if single_m is not None:
            vutils.save_image(
                visualize(single_m), fp=os.path.join(args.out_dir, f"{slice_ID}_ground_truth.png"), nrow=1, padding=10
            )

        logger.log("Sampling...")

        enslist = []
        for i in range(args.num_ensemble):  # Generate an ensemble of 5 masks
            model_kwargs = {}
            sample_fn = diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            sample, _, _, _, _ = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                img,
                step=args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            enslist.append(sample[:, -1, :, :])

        # Ensemble result
        ensres = staple(th.stack(enslist, dim=0)).squeeze(0)
        vutils.save_image(ensres, fp=os.path.join(args.out_dir, f"{slice_ID}_output_ens.png"), nrow=1, padding=10)


def create_argparser():
    defaults = dict(
        data_name='BRATS',
        data_dir="/content/drive/MyDrive/RT CW/BraTS2020/Testing/BraTS20_Training_121",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="/content/drive/MyDrive/RT_Codes_Files/MedSeggDiff/training_results2/savedmodel001000.pt",
        num_ensemble=5,  # Number of samples in the ensemble
        gpu_dev="0",
        out_dir="./evaluation_imgs",
        multi_gpu=None,  # "0,1,2"
        debug=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
