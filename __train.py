import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure
from model import MHSAUNet
from losses import CombinedLoss
from dataloader import create_dataloaders
import os
import numpy as np
import lpips
from datetime import datetime

def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=True):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        img1 (torch.Tensor): First image (BxCxHxW)
        img2 (torch.Tensor): Second image (BxCxHxW)
        max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.

    Returns:
        float: The PSNR value.
    """
    if gt_mean:
        img1_gray = img1.mean(axis=1)
        img2_gray = img2.mean(axis=1)
        
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)
    
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=True):
    """
    Calculate SSIM (Structural Similarity Index) between two images.

    Args:
        img1 (torch.Tensor): First image (BxCxHxW)
        img2 (torch.Tensor): Second image (BxCxHxW)
        max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.

    Returns:
        float: The SSIM value.
    """
    if gt_mean:
        img1_gray = img1.mean(axis=1, keepdim=True)
        img2_gray = img2.mean(axis=1, keepdim=True)
        
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()

def validate(model, dataloader, device, result_dir):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    loss_fn = lpips.LPIPS(net='alex').to(device)

    with torch.no_grad():
        for idx, (low, high) in enumerate(dataloader):
            low, high = low.to(device), high.to(device)
            output = model(low)

            # Save the output image
            save_image(output, os.path.join(result_dir, f'result_{idx}.png'))

            # Calculate PSNR
            psnr = calculate_psnr(output, high)
            total_psnr += psnr

            # Calculate SSIM
            ssim = calculate_ssim(output, high)
            total_ssim += ssim

            # Calculate LPIPS
            lpips_score = loss_fn.forward(high, output)
            total_lpips += lpips_score.item()


    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    avg_lpips = total_lpips / len(dataloader)
    return avg_psnr, avg_ssim, avg_lpips

class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, base_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr + (self.max_lr - self.base_lr) * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return self.current_epoch > self.warmup_epochs

def main():
    # Hyperparameters
    train_low = 'data/LOLv1/Train/input'
    train_high = 'data/LOLv1/Train/target'
    test_low = 'data/LOLv1/Test/input'
    test_high = 'data/LOLv1/Test/target'
    base_lr = 1e-5  # Warmup 起始學習率
    max_lr = 2e-4   # 最大學習率
    num_epochs = 1000  # 總 epoch 數改為 1000
    warmup_epochs = 50  # Warmup 階段保持 50 epoch
    cosine_epochs = 650  # CosineAnnealingLR 階段改為 650 epoch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Base LR: {base_lr}; Max LR: {max_lr}; Epochs: {num_epochs}')

    result_dir = '/content/drive/MyDrive/MHSAUNet/results/training/output'

    # Data loaders
    train_loader, test_loader = create_dataloaders(train_low, train_high, test_low, test_high, crop_size=256, batch_size=1)
    print(f'Train loader: {len(train_loader)}; Test loader: {len(test_loader)}')

    # Model, loss, optimizer, and scheduler
    model = MHSAUNet().to(device)
    criterion = CombinedLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)  # 初始學習率設為 base_lr
    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs, base_lr, max_lr)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs)  # T_max 改為 650
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=50, verbose=True)  # mode='max' 因為 SSIM 越大越好

    scaler = torch.amp.GradScaler('cuda')
    best_psnr = 0
    best_ssim = 0
    best_lpips = 1

    print('Training started.')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # 計算每個 epoch 的平均訓練損失
        avg_train_loss = train_loss / len(train_loader)

        # 驗證階段
        avg_psnr, avg_ssim, avg_lpips = validate(model, test_loader, device, result_dir)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{num_epochs}, LR: {current_lr:.6f}, Loss: {avg_train_loss:.6f}, PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}, LPIPS: {avg_lpips:.6f}')

        # 學習率調度邏輯
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        elif epoch < warmup_epochs + cosine_epochs:
            cosine_scheduler.step()
        else:
            plateau_scheduler.step(avg_ssim)  # 根據驗證集 SSIM 動態調整學習率

        # 儲存最佳模型
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            model_path = "/content/drive/MyDrive/MHSAUNet/best_psnr_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f'Saving model with PSNR: {best_psnr:.6f}')

        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            model_path = "/content/drive/MyDrive/MHSAUNet/best_ssim_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f'Saving model with SSIM: {best_ssim:.6f}')
        
        if avg_lpips < best_lpips:
            best_lpips = avg_lpips
            model_path = "/content/drive/MyDrive/MHSAUNet/best_lpips_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f'Saving model with LPIPS: {best_lpips:.6f}')

        # 寫入日誌
        now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        file_path = "/content/drive/MyDrive/MHSAUNet/results/training/metrics.md"
        file_exists = os.path.exists(file_path)

        with open(file_path, "a") as f:
            if not file_exists:
                f.write("|   Timestemp   |   Epoch   |    LR    |   Loss   |   PSNR   |   SSIM   |   LPIPS   |\n")
                f.write("|---------------|-----------|----------|----------|----------|----------|-----------|\n")
            f.write(f"|   {now}   | {epoch + 1} | {current_lr:.6f} | {avg_train_loss:.6f} |  {avg_psnr:.6f}  |  {avg_ssim:.6f}  |  {avg_lpips:.6f}  |\n")

if __name__ == '__main__':
    main()