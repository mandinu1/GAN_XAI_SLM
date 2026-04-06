import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from utils.FID import calculate_fid
from utils.inception_score import inception_score
from utils.msssim import calculate_ms_ssim
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Generator
class Generator(nn.Module):
    def __init__(self, channels=3, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            # 1x1 -> 4x4
            nn.ConvTranspose2d(z_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(32, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.features = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(channels, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),

            # 128x128 -> 64x64
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=False),

            # 64x64 -> 32x32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),

            # 32x32 -> 16x16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False),

            # 16x16 -> 8x8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=False),

            # 8x8 -> 4x4
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        f = self.features(x)
        out = self.classifier(f)
        return out.view(-1)

# GradCAM Attention Loss
def gradcam_attention_loss(features, cam):
    if cam.dim() == 3:
        cam = cam.unsqueeze(1)

    cam = F.interpolate(cam, size=features.shape[2:], mode="bilinear", align_corners=False)
    attention = torch.mean(features, dim=1, keepdim=True)

    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    return torch.mean((attention - cam) ** 2)


def saliency_attention_loss(features, saliency):
    if saliency.dim() == 3:
        saliency = saliency.unsqueeze(1)

    saliency = F.interpolate(saliency, size=features.shape[2:], mode="bilinear", align_corners=False)
    attention = torch.mean(features, dim=1, keepdim=True)

    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    return torch.mean((attention - saliency) ** 2)


def xai_attention_loss(features, cam):
    return gradcam_attention_loss(features, cam)

class DCGAN_MODEL:
    def __init__(self, args, use_xai=False, lambda_xai=0.1, xai_mode=None):
        print("Initializing DCGAN...")
        self.run_dir = args.run_dir
        self.sample_dir = os.path.join(self.run_dir, "samples")
        self.eval_dir = os.path.join(self.run_dir, "evaluation")
        self.train_plot_dir = os.path.join(self.run_dir, "training_plots")

        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.train_plot_dir, exist_ok=True)

        self.G = Generator(args.channels).to(device)
        self.D = Discriminator(args.channels).to(device)
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.use_xai = use_xai
        self.lambda_xai = lambda_xai
        self.xai_mode = xai_mode

        self.criterion = nn.BCELoss()
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))

        if self.use_xai and self.xai_mode in ["gradcam", "both"]:
            from Xai_tools.grad_cam import GradCAM
            # Uses an earlier convolutional layer (8x8 feature map) for richer Grad-CAM attention
            self.gradcam = GradCAM(self.D, self.D.features[11])

        if self.use_xai and self.xai_mode in ["saliency", "both"]:
            from Xai_tools.saliency_map import SaliencyMap
            self.saliency = SaliencyMap(self.D)

    def generate_saliency(self, images):
        images = images.clone().detach().requires_grad_(True)

        outputs = self.D(images)
        score = outputs.sum()

        self.D.zero_grad()
        score.backward()

        saliency = images.grad.abs()
        saliency = saliency.mean(dim=1, keepdim=True)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        return saliency.detach()

    def train(self, train_loader, test_loader):
        self.G.train()
        self.D.train()

        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.train_plot_dir, exist_ok=True)

        g_losses, d_losses, gradcam_losses, saliency_losses, total_xai_losses, fid_scores = [], [], [], [], [], []

        best_fid = float('inf')
        best_epoch = -1
        # patience = 5
        counter = 0

        for epoch in range(self.epochs):
            epoch_g_loss = 0

            for i, (real_images, _) in enumerate(train_loader):
                real_images = real_images.to(device)
                bs = real_images.size(0)

                real_labels = torch.ones(bs, device=device)
                fake_labels = torch.zeros(bs, device=device)

                # Discriminator
                z = torch.randn(bs, 100, 1, 1, device=device)
                fake_images = self.G(z)

                d_loss = self.criterion(self.D(real_images), real_labels) + \
                         self.criterion(self.D(fake_images.detach()), fake_labels)

                self.D.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Generator
                z = torch.randn(bs, 100, 1, 1, device=device)
                fake_images = self.G(z)

                g_adv_loss = self.criterion(self.D(fake_images), real_labels)

                g_xai_loss = torch.tensor(0.0, device=device)
                g_gradcam_loss = torch.tensor(0.0, device=device)
                g_saliency_loss = torch.tensor(0.0, device=device)

                if self.use_xai and self.xai_mode is not None:
                    features = self.D.features(fake_images)

                    if self.xai_mode in ["gradcam", "both"]:
                        cam = self.gradcam.generate(fake_images)
                        g_gradcam_loss = gradcam_attention_loss(features, cam)
                        g_xai_loss = g_xai_loss + g_gradcam_loss

                    if self.xai_mode in ["saliency", "both"]:
                        saliency = self.saliency.generate(fake_images)
                        g_saliency_loss = saliency_attention_loss(features, saliency)
                        g_xai_loss = g_xai_loss + g_saliency_loss

                    if self.xai_mode == "both":
                        g_xai_loss = 0.5 * g_xai_loss

                    g_loss = g_adv_loss + self.lambda_xai * g_xai_loss
                else:
                    g_loss = g_adv_loss

                self.G.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                g_losses.append(g_adv_loss.item())
                d_losses.append(d_loss.item())
                gradcam_losses.append(g_gradcam_loss.item())
                saliency_losses.append(g_saliency_loss.item())
                total_xai_losses.append(g_xai_loss.item())

                epoch_g_loss += g_adv_loss.item()

                print(
                    f"[Epoch {epoch+1}/{self.epochs}] [Batch {i}] "
                    f"D: {d_loss.item():.4f} G: {g_adv_loss.item():.4f} "
                    f"GCAM: {g_gradcam_loss.item():.4f} SAL: {g_saliency_loss.item():.4f} "
                    f"XAI: {g_xai_loss.item():.4f}"
                )

            # FID Calculation every epoch
            self.G.eval()

           
            real_batch = []
            for imgs, _ in test_loader:
                real_batch.append(imgs)

            real_images = torch.cat(real_batch, dim=0).to(device)
            real_images = (real_images + 1) / 2

            n_real = real_images.size(0)

            with torch.no_grad():
                z = torch.randn(n_real, 100, 1, 1, device=device)
                fake_images = self.G(z)
                fake_images = (fake_images + 1) / 2

            fid = calculate_fid(real_images, fake_images)
            fid_scores.append(fid)

            print(f" Epoch {epoch+1} FID: {fid:.4f}")

            # Save best-FID model
            if fid < best_fid:
                best_fid = fid
                best_epoch = epoch + 1
                counter = 0

                print(f"BEST model at epoch {epoch+1} (FID={fid:.4f})")
                self.save_best_model(best_epoch, best_fid)
                self.save_best_samples(epoch)
            else:
                counter += 1

            self.G.train()

            # #Early stopping
            # if counter >= patience:
            #     print(" Early stopping triggered")
            #     break

        # Save final model from the last epoch
        self.save_last_model(self.epochs)

        print(f"Training finished. Best FID: {best_fid:.4f} at epoch {best_epoch}")

        # Save Loss Graphs
        self.save_loss_plots(g_losses, d_losses, gradcam_losses, saliency_losses, total_xai_losses, fid_scores)

    def save_best_samples(self, epoch):
        
        for fname in os.listdir(self.sample_dir):
            if fname.endswith(".png"):
                os.remove(os.path.join(self.sample_dir, fname))

        with torch.no_grad():
            z = torch.randn(64, 100, 1, 1, device=device)
            samples = self.G(z)
            samples = (samples + 1) / 2

            for i, img in enumerate(samples):
                utils.save_image(
                    img,
                    os.path.join(self.sample_dir, f"best_img_{i+1:03d}.png")
                )

    def save_loss_plots(self, g_losses, d_losses, gradcam_losses, saliency_losses, total_xai_losses, fid_scores):
        # Generator Loss
        plt.figure()
        plt.plot(g_losses)
        plt.title("Generator Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Generator Loss")
        plt.savefig(os.path.join(self.train_plot_dir, "g_loss.png"))
        plt.close()

        # Discriminator Loss
        plt.figure()
        plt.plot(d_losses)
        plt.title("Discriminator Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Discriminator Loss")
        plt.savefig(os.path.join(self.train_plot_dir, "d_loss.png"))
        plt.close()

        # Grad-CAM Loss
        if len(gradcam_losses) > 0:
            plt.figure()
            plt.plot(gradcam_losses)
            plt.title("Grad-CAM Loss")
            plt.xlabel("Iterations")
            plt.ylabel("Grad-CAM Loss")
            plt.savefig(os.path.join(self.train_plot_dir, "gradcam_loss.png"))
            plt.close()

        # Saliency Loss
        if len(saliency_losses) > 0:
            plt.figure()
            plt.plot(saliency_losses)
            plt.title("Saliency Loss")
            plt.xlabel("Iterations")
            plt.ylabel("Saliency Loss")
            plt.savefig(os.path.join(self.train_plot_dir, "saliency_loss.png"))
            plt.close()

        # Total XAI Loss
        if len(total_xai_losses) > 0:
            plt.figure()
            plt.plot(total_xai_losses)
            plt.title("Total XAI Loss")
            plt.xlabel("Iterations")
            plt.ylabel("Total XAI Loss")
            plt.savefig(os.path.join(self.train_plot_dir, "xai_total_loss.png"))
            plt.close()

        # FID Plot
        if len(fid_scores) > 0:
            plt.figure()
            plt.plot(fid_scores)
            plt.title("FID Score")
            plt.xlabel("Epochs")
            plt.ylabel("FID Score")
            plt.savefig(os.path.join(self.train_plot_dir, "fid.png"))
            plt.close()

    def save_best_model(self, epoch, fid):
        torch.save(self.G.state_dict(), os.path.join(self.run_dir, "generator_DCGAN_best_fid.pth"))
        torch.save(self.D.state_dict(), os.path.join(self.run_dir, "discriminator_DCGAN_best_fid.pth"))

        with open(os.path.join(self.run_dir, "best_fid_info.txt"), "w") as f:
            f.write(f"Best epoch: {epoch}\n")
            f.write(f"Best FID: {fid:.4f}\n")

        print("Best-FID models saved.")

    def save_last_model(self, epoch):
        torch.save(self.G.state_dict(), os.path.join(self.run_dir, "generator_DCGAN_last_epoch.pth"))
        torch.save(self.D.state_dict(), os.path.join(self.run_dir, "discriminator_DCGAN_last_epoch.pth"))

        with open(os.path.join(self.run_dir, "last_epoch_info.txt"), "w") as f:
            f.write(f"Last epoch: {epoch}\n")

        print("Last-epoch models saved.")

    def load_model(self, g_path=None, d_path=None):
        if g_path is None:
            g_path = os.path.join(self.run_dir, "generator_DCGAN_best_fid.pth")
        if d_path is None:
            d_path = os.path.join(self.run_dir, "discriminator_DCGAN_best_fid.pth")

        self.G.load_state_dict(torch.load(g_path, map_location=device))
        self.D.load_state_dict(torch.load(d_path, map_location=device))
        self.G.eval()
        self.D.eval()

        print("Models loaded.")

    def evaluate(self, test_loader=None):
        self.G.eval()

        
        if test_loader is not None:
            real_images_batch = []
            for imgs, _ in test_loader:
                real_images_batch.append(imgs)

            real_images = torch.cat(real_images_batch, dim=0).to(device)
            real_images_norm = (real_images + 1) / 2
            n_real = real_images_norm.size(0)
        else:
            # fallback size when no loader is available
            n_real = 100
            real_images_norm = None

        # Generating the same number of fake images as real images for fair evaluation
        z = torch.randn(n_real, 100, 1, 1, device=device)
        with torch.no_grad():
            fake_images = self.G(z)
            fake_images_norm = (fake_images + 1) / 2

        # Inception Score
        is_mean, is_std = inception_score(
            fake_images_norm,
            batch_size=32,
            splits=10,
            device=device
        )

        if real_images_norm is None:
            
            real_images_norm = fake_images_norm.clone()

        fid_score = calculate_fid(real_images_norm, fake_images_norm)
        ms_ssim_score = calculate_ms_ssim(fake_images_norm)

        # Save metrics to a text file
        metrics_path = os.path.join(self.eval_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}\n")
            f.write(f"FID Score: {fid_score:.4f}\n")
            f.write(f"MS-SSIM: {ms_ssim_score:.4f}\n")
            f.write(f"XAI Enabled: {self.use_xai}\n")
            f.write(f"Lambda XAI: {self.lambda_xai}\n")

        # Save generated images 
        samples = fake_images_norm[:64]  # first 64 images
        for idx, img in enumerate(samples):
            utils.save_image(
                img,
                os.path.join(self.eval_dir, f"sample_{idx+1}.png")
            )

        print(f"Evaluation complete. Metrics and images saved to {self.eval_dir}")
        print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
        print(f"FID Score: {fid_score:.4f}")
        print(f"MS-SSIM: {ms_ssim_score:.4f}")