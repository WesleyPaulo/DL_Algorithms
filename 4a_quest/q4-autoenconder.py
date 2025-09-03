#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
import random
import os

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class Config:
    BATCH_SIZE = 128
    LATENT_DIM = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    NOISE_FACTOR = 0.3
    DATA_DIR = "./fashion_mnist_data"
    CLASS_NAMES = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

def prepare_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST(
        root=Config.DATA_DIR, train=True, download=True, transform=transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root=Config.DATA_DIR, train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )
    
    print(f"Dataset Fashion-MNIST:")
    print(f"   Treino: {len(train_dataset)} imagens")
    print(f"   Teste: {len(test_dataset)} imagens")
    
    return train_loader, test_loader

def add_noise(images, noise_factor=0.3):
    noise = torch.randn_like(images) * noise_factor
    noisy_images = images + noise
    return torch.clamp(noisy_images, -1.0, 1.0)

class AutoencoderMLP(nn.Module):
    def __init__(self, latent_dim=32):
        super(AutoencoderMLP, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28*28),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(x.size(0), 1, 28, 28)

def train_autoencoder(model, train_loader, test_loader, epochs, is_denoising=False):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    train_losses = []
    test_losses = []
    
    print(f"Iniciando treinamento do {'Denoising ' if is_denoising else ''}Autoencoder...")
    print("=" * 60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            if is_denoising:
                noisy_data = add_noise(data, Config.NOISE_FACTOR)
                noisy_data = noisy_data.to(device)
                optimizer.zero_grad()
                outputs = model(noisy_data)
                loss = criterion(outputs, data)
            else:
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, data)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                
                if is_denoising:
                    noisy_data = add_noise(data, Config.NOISE_FACTOR)
                    noisy_data = noisy_data.to(device)
                    outputs = model(noisy_data)
                    loss = criterion(outputs, data)
                else:
                    outputs = model(data)
                    loss = criterion(outputs, data)
                
                test_loss += loss.item()
        
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] | '
                  f'Train Loss: {train_loss:.6f} | '
                  f'Test Loss: {test_loss:.6f}')
    
    print("=" * 60)
    print("Treinamento concluído!")
    
    return train_losses, test_losses

def visualize_results(original_images, reconstructed_images, title, is_denoising=False):
    n_images = min(10, len(original_images))
    
    if is_denoising:
        fig, axes = plt.subplots(3, n_images, figsize=(20, 8))
        noisy_images = add_noise(original_images, Config.NOISE_FACTOR)
        
        for i in range(n_images):
            axes[0, i].imshow(original_images[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(noisy_images[i].cpu().squeeze(), cmap='gray')
            axes[1, i].set_title('Com Ruído')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(reconstructed_images[i].cpu().squeeze(), cmap='gray')
            axes[2, i].set_title('Reconstruída')
            axes[2, i].axis('off')
            
        plt.suptitle(f'{title} - Denoising Autoencoder', fontsize=16)
        
    else:
        fig, axes = plt.subplots(2, n_images, figsize=(20, 6))
        for i in range(n_images):
            axes[0, i].imshow(original_images[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(reconstructed_images[i].cpu().squeeze(), cmap='gray')
            axes[1, i].set_title('Reconstruída')
            axes[1, i].axis('off')
        
        plt.suptitle(f'{title} - Autoencoder', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_loss_curve(train_losses, test_losses, title):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Treino', linewidth=2, color='blue')
    plt.plot(test_losses, label='Teste', linewidth=2, color='red')
    plt.xlabel('Época')
    plt.ylabel('Erro Médio (MSE Loss)')
    plt.title(f'Curva do Erro Médio - {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'curva_erro_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

train_loader, test_loader = prepare_data()


# PARTE 4(A) - AUTOENCODER REGULAR

print("\n" + "="*60)
print("4(A) - AUTOENCODER REGULAR")
print("="*60)

ae_model = AutoencoderMLP(latent_dim=Config.LATENT_DIM).to(device)
ae_train_losses, ae_test_losses = train_autoencoder(
    ae_model, train_loader, test_loader, Config.EPOCHS, is_denoising=False
)

plot_loss_curve(ae_train_losses, ae_test_losses, "Autoencoder Regular")

test_images = []
test_labels = []
with torch.no_grad():
    for data, labels in test_loader:
        test_images.append(data)
        test_labels.append(labels)
        if len(torch.cat(test_images)) >= 10:
            break

test_images = torch.cat(test_images)[:10]
test_labels = torch.cat(test_labels)[:10]

ae_model.eval()
with torch.no_grad():
    reconstructed = ae_model(test_images.to(device))

visualize_results(test_images, reconstructed, "Resultados Autoencoder Regular", is_denoising=False)


# PARTE 4(B) - DENOISING AUTOENCODER

print("\n" + "="*60)
print("4(B) - DENOISING AUTOENCODER")
print("="*60)

dae_model = AutoencoderMLP(latent_dim=Config.LATENT_DIM).to(device)
dae_train_losses, dae_test_losses = train_autoencoder(
    dae_model, train_loader, test_loader, Config.EPOCHS, is_denoising=True
)

plot_loss_curve(dae_train_losses, dae_test_losses, "Denoising Autoencoder")

dae_model.eval()
with torch.no_grad():
    noisy_test_images = add_noise(test_images, Config.NOISE_FACTOR)
    denoised_reconstructed = dae_model(noisy_test_images.to(device))

visualize_results(test_images, denoised_reconstructed, "Resultados Denoising Autoencoder", is_denoising=True)


print("\n" + "="*60)
print("COMPARAÇÃO QUALITATIVA")
print("="*60)

def calculate_mse(original, reconstructed):
    return F.mse_loss(original, reconstructed).item()

ae_mse = calculate_mse(test_images.to(device), reconstructed)
dae_mse = calculate_mse(test_images.to(device), denoised_reconstructed)

print(f"MSE Autoencoder Regular: {ae_mse:.6f}")
print(f"MSE Denoising Autoencoder: {dae_mse:.6f}")

fig, axes = plt.subplots(4, 10, figsize=(20, 8))

for i in range(10):
    axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
    axes[0, i].set_title(f'Original\n{Config.CLASS_NAMES[test_labels[i]]}')
    axes[0, i].axis('off')
    
    noisy_img = add_noise(test_images[i].unsqueeze(0), Config.NOISE_FACTOR).squeeze()
    axes[1, i].imshow(noisy_img.cpu().squeeze(), cmap='gray')
    axes[1, i].set_title('Com Ruído')
    axes[1, i].axis('off')
    
    axes[2, i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
    axes[2, i].set_title(f'AE Rec\nMSE: {calculate_mse(test_images[i].unsqueeze(0).to(device), reconstructed[i].unsqueeze(0)):.4f}')
    axes[2, i].axis('off')
    
    axes[3, i].imshow(denoised_reconstructed[i].cpu().squeeze(), cmap='gray')
    axes[3, i].set_title(f'DAE Rec\nMSE: {calculate_mse(test_images[i].unsqueeze(0).to(device), denoised_reconstructed[i].unsqueeze(0)):.4f}')
    axes[3, i].axis('off')

plt.suptitle('Comparação: Autoencoder vs Denoising Autoencoder', fontsize=16)
plt.tight_layout()
plt.savefig('comparacao_final.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nArquivos gerados:")
print("1. resultados_autoencoder_regular.png")
print("2. resultados_denoising_autoencoder.png")
print("3. curva_erro_autoencoder_regular.png")
print("4. curva_erro_denoising_autoencoder.png")
print("5. comparacao_final.png")

print("\nAnálise completa concluída!")
