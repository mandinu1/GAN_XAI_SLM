import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of GAN models "
    )

    
    # Model & dataset
   
    parser.add_argument(
        '--model',
        type=str,
        default='DCGAN',
        choices=['GAN', 'DCGAN', 'WGAN-CP', 'WGAN-GP']
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['cifar10', 'stl10'],
        help='Dataset name'
    )

    parser.add_argument(
        '--dataroot',
        type=str,
        required=True,
        help='Path to dataset'
    )

    
    # Training options
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )

    parser.add_argument(
        '--generator_iters',
        type=int,
        default=10000,
        help='Generator iterations for WGAN'
    )

    
    # Boolean flags 
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Enable training mode'
    )

    parser.add_argument(
        '--download',
        action='store_true',
        help='Download dataset if not found'
    )

    
    # Model loading
    
    parser.add_argument(
        '--load_D',
        type=str,
        default=None,
        help='Path to discriminator checkpoint'
    )

    parser.add_argument(
        '--load_G',
        type=str,
        default=None,
        help='Path to generator checkpoint'
    )

    args = parser.parse_args()
    return check_args(args)


def check_args(args):

    
    # Sanity checks
    if args.epochs < 1:
        raise ValueError("Epochs must be >= 1")

    if args.batch_size < 1:
        raise ValueError("Batch size must be >= 1")

    
    # Dataset channels
    
    if args.dataset in ['cifar', 'stl10']:
        args.channels = 3
    else:
        args.channels = 1

    # Device selection (
    if torch.backends.mps.is_available():
        args.device = 'mps'
    else:
        args.device = 'cpu'

    print(f"Using device: {args.device}")

    return args