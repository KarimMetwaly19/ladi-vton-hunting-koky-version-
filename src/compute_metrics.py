import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
from accelerate import Accelerator
from diffusers import DDIMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor

from dataset.dresscode import DressCodeDataset
from dataset.vitonhd import VitonHDDataset
from models.AutoencoderKL import AutoencoderKL
from src.utils.encode_text_word_embedding import encode_text_word_embedding
from utils.set_seeds import set_seed
from utils.val_metrics import compute_metrics
from vto_pipelines.tryon_pipe import StableDiffusionTryOnePipeline

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

import matplotlib.pyplot as plt
import numpy as np
def show_warped_cloth(warped_cloth):
    print("Size of warped_cloth tensor:", warped_cloth.size())
    # Assuming warped_cloth is your tensor after all the processing
    # Convert the tensor to a NumPy array
    warped_cloth_np = warped_cloth.cpu().numpy()
    # Normalize the image to the range 0-1
    warped_cloth_np = (warped_cloth_np + 1) / 2
    # Assuming warped_cloth_np has dimensions (batch_size, channels, height, width)
    # Select the first image from the batch
    first_image = warped_cloth_np[0]
    
    # Now, transpose the first image to (height, width, channels)
    first_image_transposed = np.transpose(first_image, (1, 2, 0))
    
    # Normalize the image to the range 0-1
    first_image_transposed = (first_image_transposed + 1) / 2
    
    # Save the image to a file
    plt.imsave('/kaggle/working/warped_cloth.png', first_image_transposed)



def parse_args():
    parser = argparse.ArgumentParser(description="Full inference script")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory",
    )

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )

    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size to use.")

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
    parser.add_argument('--vitonhd_dataroot', type=str, help='VitonHD dataroot')

    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the dataloader")

    parser.add_argument("--num_vstar", default=16, type=int, help="Number of predicted v* images to use")
    parser.add_argument("--test_order", type=str, required=True, choices=["unpaired", "paired"])
    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"], help="dataset to use")
    parser.add_argument("--category", type=str, choices=['all', 'lower_body', 'upper_body', 'dresses'], default='all')
    parser.add_argument("--use_png", default=False, action="store_true", help="Whether to use png or jpg for saving")
    parser.add_argument("--num_inference_steps", default=80, type=int, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", default=7.5, type=float, help="Guidance scale")
    parser.add_argument("--compute_metrics", default=False, action="store_true",
                        help="Compute metrics after generation")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


@torch.inference_mode()
def main():
    args = parse_args()

    # Check if the dataset dataroot is provided
    if args.dataset == "vitonhd" and args.vitonhd_dataroot is None:
        raise ValueError("VitonHD dataroot must be provided")
    if args.dataset == "dresscode" and args.dresscode_dataroot is None:
        raise ValueError("DressCode dataroot must be provided")

    # Enable TF32 for faster inference on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Setup accelerator and device.
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    

    # Enable xformers memory efficient attention if requested
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Load the datasets
    if args.category != 'all':
        category = [args.category]
    else:
        category = ['dresses', 'upper_body', 'lower_body']

    outputlist = ['image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name', 'cloth', 'warped_cloth']
    if args.dataset == "dresscode":
        test_dataset = DressCodeDataset(
            dataroot_path=args.dresscode_dataroot,
            phase='test',
            order=args.test_order,
            radius=5,
            outputlist=outputlist,
            category=category,
            size=(512, 384)
        )
    elif args.dataset == "vitonhd":
        test_dataset = VitonHDDataset(
            dataroot_path=args.vitonhd_dataroot,
            phase='test',
            order=args.test_order,
            radius=5,
            outputlist=outputlist,
            size=(512, 384),
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    save_dir = os.path.join(args.output_dir, args.test_order)
    os.makedirs(save_dir, exist_ok=True)

    
    # Compute metrics if requested
    if args.compute_metrics:
        metrics = compute_metrics(save_dir, args.test_order, args.dataset, args.category, ['all'],
                                  args.dresscode_dataroot, args.vitonhd_dataroot)

        with open(os.path.join(save_dir, f"metrics_{args.test_order}_{args.category}.json"), "w+") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
