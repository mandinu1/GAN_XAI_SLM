from pytorch_msssim import ms_ssim
import torch
import torch.nn.functional as F


def calculate_ms_ssim(fake_images):
    
    fake_images = fake_images.to(dtype=torch.float32)
    fake_images = torch.clamp(fake_images, 0.0, 1.0)

   
    if fake_images.shape[2] < 160 or fake_images.shape[3] < 160:
        fake_images = F.interpolate(
            fake_images,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

    if len(fake_images) < 2:
        return 0.0

    scores = []

    for i in range(len(fake_images) - 1):
        img1 = fake_images[i].unsqueeze(0)
        img2 = fake_images[i + 1].unsqueeze(0)

        score = ms_ssim(
            img1,
            img2,
            data_range=1.0,
            size_average=True
        )
        scores.append(score.item())

    return float(sum(scores) / len(scores))