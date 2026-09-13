import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class OASISDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L') 
        if self.transform:
            image = self.transform(image)
        return image, 0 

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvVAE, self).__init__()
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enc_fc_mu = nn.Linear(64 * 16 * 16, latent_dim)
        self.enc_fc_logvar = nn.Linear(64 * 16 * 16, latent_dim)
        
        self.dec_fc = nn.Linear(latent_dim, 64 * 16 * 16)
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        
    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = x.view(x.size(0), -1)
        return self.enc_fc_mu(x), self.enc_fc_logvar(x)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = x.view(x.size(0), 64, 16, 16)
        x = F.relu(self.dec_conv1(x))
        return torch.sigmoid(self.dec_conv2(x))
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training VAE on device: {device}")
    
    data_dir = '/home/groups/comp3710/OASIS/keras_png_slices_train'
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    print("Loading OASIS Dataset...")
    dataset = OASISDataset(root_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        
    model = ConvVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 20
    
    model.train()
    for epoch in range(1, epochs + 1):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            
            train_loss += loss.item()
            optimizer.step()
            
        print(f'Epoch [{epoch}/{epochs}] | Average Loss: {train_loss / len(dataloader.dataset):.4f}')
        
    model.eval()
    with torch.no_grad():
        sample = torch.randn(64, 128).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 64, 64), 'vae_brain_manifold.png')
        print("Successfully saved VAE manifold to 'vae_brain_manifold.png'")

if __name__ == '__main__':
    main()
