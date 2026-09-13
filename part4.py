import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvVAE, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1) # 64x64 -> 32x32
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # 32x32 -> 16x16
        self.enc_fc_mu = nn.Linear(64 * 16 * 16, latent_dim)
        self.enc_fc_logvar = nn.Linear(64 * 16 * 16, latent_dim)
        
        # Decoder
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
    # Kullback-Leibler divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training VAE on device: {device}")
    
    # ⚠️ We will update this path based on your `ls` command output
    data_dir = '/home/groups/comp3710/OASIS_PLACEHOLDER' 
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    
    # Assumes standard image directory structure. Will adjust if data is in .npy format.
    try:
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    except FileNotFoundError:
        print(f"Waiting for correct data path to be confirmed...")
        return
        
    model = ConvVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 30
    
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
        
    # Generate and save the 2D Manifold visualization
    model.eval()
    with torch.no_grad():
        # Sample 64 random vectors from the latent space
        sample = torch.randn(64, 128).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 64, 64), 'vae_brain_manifold.png')
        print("Successfully saved VAE manifold to 'vae_brain_manifold.png'")

if __name__ == '__main__':
    main()