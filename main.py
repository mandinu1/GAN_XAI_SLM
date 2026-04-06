import os
import argparse
from utils.data_loader import get_data_loader
from models.dcgan import DCGAN_MODEL
from models.styleGAN3 import StyleGAN3Model

def parse_args():
    parser = argparse.ArgumentParser(description="DCGAN Training and Evaluation")
    
    # Model options
    parser.add_argument('--model', type=str, default='DCGAN', choices=['GAN', 'DCGAN', 'STYLEGAN3'])
    
    # Dataset options
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','stl10','place365'], help='Dataset name')
    parser.add_argument('--dataroot', type=str, required=True, help='Root folder of dataset')
    parser.add_argument('--channels', type=int, default=3, help='Number of image channels')
    parser.add_argument('--target_class',type=str,default=None,help='Train on a single class only (e.g., airplane)')
    # Training options
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--train', action='store_true', help='Flag to train the model')

    # Model loading
    parser.add_argument('--load_G', type=str, default=None, help='Path to generator .pth file')
    parser.add_argument('--load_D', type=str, default=None, help='Path to discriminator .pth file')

    # XAI options
    parser.add_argument('--use_xai', action='store_true', help='Enable XAI-guided training')
    parser.add_argument('--lambda_xai', type=float, default=0.1, help='Weight for XAI loss')
    parser.add_argument('--xai_mode',type=str,default=None, choices=['gradcam', 'saliency', 'both'],help='Type of XAI method to use')

    parser.add_argument('--run_name',
    type=str,
    required=True,
    help='Name of the experiment run ')

    # StyleGAN3 options
    parser.add_argument('--image_size', type=int, default=256, help='Image size for StyleGAN3 dataset preparation')
    parser.add_argument('--stylegan3_repo', type=str, default='./external/stylegan3', help='Path to the official StyleGAN3 repository')
    parser.add_argument('--stylegan3_outdir', type=str, default='stylegan3_runs', help='Subfolder name for StyleGAN3 training outputs')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs for StyleGAN3 training')
    parser.add_argument('--gamma', type=float, default=6.6, help='R1 regularization gamma for StyleGAN3')
    parser.add_argument('--kimg', type=int, default=500, help='Training duration in kimg for StyleGAN3')
    parser.add_argument('--mirror', action='store_true', help='Enable x-flip augmentation for StyleGAN3')
    parser.add_argument('--cfg', type=str, default='stylegan3-t', help='StyleGAN3 config preset')
    parser.add_argument('--snap', type=int, default=10, help='Snapshot interval for StyleGAN3')
    parser.add_argument('--metrics', type=str, default='fid50k_full', help='Metrics used by StyleGAN3')
    parser.add_argument('--cond', action='store_true', help='Train StyleGAN3 as a conditional model')
    parser.add_argument('--workers', type=int, default=3, help='Number of dataloader workers for StyleGAN3')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for StyleGAN3')

    return parser.parse_args()

def main():
    args = parse_args()

    run_dir = os.path.join("experiments", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    args.run_dir = run_dir

    train_loader, test_loader = None, None
    if args.model != 'STYLEGAN3':
        train_loader, test_loader = get_data_loader(args, target_class=args.target_class)

    if args.model == 'STYLEGAN3':
        model = StyleGAN3Model(args)
    else:
        model = DCGAN_MODEL(
            args,
            use_xai=args.use_xai,
            lambda_xai=args.lambda_xai,
            xai_mode=args.xai_mode
        )

    # Load pre-trained models if specified
    if args.model == 'STYLEGAN3':
        if args.load_G:
            model.load_model(args.load_G, None)
    else:
        if args.load_G and args.load_D:
            model.load_model(args.load_G, args.load_D)

    
    # Train
    if args.train:
        model.train(train_loader, test_loader)
    else:
        model.evaluate(test_loader)

if __name__ == "__main__":
    main()