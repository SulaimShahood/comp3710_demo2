import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

class OASISSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Sort files to ensure images and masks align perfectly
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            # Resize mask with nearest neighbor to preserve discrete class values
            mask = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST)(mask)
            mask = transforms.PILToTensor()(mask).squeeze(0)
        
        # The OASIS masks typically use 4 grayscale values for classes (0, 85, 170, 255)
        # We divide by 85 and cast to long to get class indices [0, 1, 2, 3]
        mask = (mask.long() / 85).long()
        return image, mask

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_classes=4):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        
        self.pool = nn.MaxPool2d(2)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)
        
        self.out_conv = nn.Conv2d(32, out_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        
        # Decoder
        x = self.up1(x3)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up2(x)
        
        # Categorical logits [Batch, out_classes, H, W]
        return self.out_conv(x)

def dice_coefficient(pred, target, num_classes=4):
    """Calculate DSC for all labels > 0.9 requirement"""
    pred = torch.argmax(pred, dim=1)
    dices = []
    for cls in range(num_classes):
        pred_c = (pred == cls)
        target_c = (target == cls)
        intersection = (pred_c & target_c).float().sum((1, 2))
        union = pred_c.float().sum((1, 2)) + target_c.float().sum((1, 2))
        
        # Add epsilon to prevent division by zero
        dice = (2. * intersection + 1e-8) / (union + 1e-8)
        dices.append(dice.mean().item())
    return dices

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training UNet on: {device}")
    
    img_dir = '/home/groups/comp3710/OASIS/keras_png_slices_train'
    mask_dir = '/home/groups/comp3710/OASIS/keras_png_slices_seg_train'
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    dataset = OASISSegmentationDataset(img_dir, mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = UNet(in_channels=1, out_classes=4).to(device)
    # CrossEntropyLoss expects target class indices, perfectly matching our mask format
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        all_dices = []
        
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            all_dices.append(dice_coefficient(outputs, masks))
            
        avg_loss = epoch_loss / len(dataloader)
        avg_dice = np.mean(all_dices, axis=0)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        print(f"DSC - Bg: {avg_dice[0]:.4f} | CSF: {avg_dice[1]:.4f} | GM: {avg_dice[2]:.4f} | WM: {avg_dice[3]:.4f}")

    # Visualization for Demonstration
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(dataloader))
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(images[0].cpu().squeeze(), cmap='gray')
        axes[0].set_title('Original MRI')
        axes[1].imshow(masks[0].cpu(), cmap='viridis')
        axes[1].set_title('Ground Truth Mask')
        axes[2].imshow(preds[0].cpu(), cmap='viridis')
        axes[2].set_title('UNet Prediction')
        plt.savefig('unet_segmentation_result.png')
        print("\nSaved segmentation comparison to 'unet_segmentation_result.png'")

    torch.save(model.state_dict(), 'unet_weights.pth')

if __name__ == "__main__":
    main()
