import sys
import argparse
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")
from guided_diffusion.guided_diffusion import dist_util, logger
from guided_diffusion.guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.guided_diffusion.isicloader import ISICDataset
from guided_diffusion.guided_diffusion.custom_dataset_loader import CustomDataset
from guided_diffusion.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.guided_diffusion.train_util import TrainLoop
# from visdom import Visdom
# viz = Visdom(port=8850)
import torchvision.transforms as transforms
import os
import tempfile

def main():
    args = create_argparser().parse_args()
    
    # Set custom temporary directory
    # os.environ['TMPDIR'] = 'D:/temp_medical'
    # os.makedirs('D:/temp_medical', exist_ok=True)
    # tempfile.tempdir = 'D:/temp_medical'
    
    logger.configure(dir = args.out_dir)

    logger.log("creating data loader...")

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_train = transforms.Compose(tran_list)

        ds = BRATSDataset3D(args.data_dir, transform_train, test_flag=False)
        args.in_ch = 5
    else :
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)
        print("Your current directory : ",args.data_dir)
        ds = CustomDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
        
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    device = th.device('cuda', int(args.gpu_dev)) if th.cuda.is_available() else th.device('cpu')
    if args.multi_gpu:
        model = th.nn.DataParallel(model, device_ids=[int(id) for id in args.multi_gpu.split(',')])
    model.to(device)

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        max_train_steps=args.max_train_steps
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_name = 'BRATS',
        data_dir="/content/drive/MyDrive/RT CW/BraTS2020/Training",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1,
        save_interval=100,
        max_train_steps=1000,  # Set to 100 steps
        resume_checkpoint=None, #"/results/pretrainedmodel.pt"
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "0",
        multi_gpu = None, #"0,1,2"
        out_dir='./training_results2/'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
