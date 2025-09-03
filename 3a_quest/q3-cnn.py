#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import json
import random
import os
from dataclasses import dataclass
from typing import Tuple

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    step_size: int = 15
    gamma: float = 0.1
    augment: bool = True
    num_workers: int = 2
    data_dir: str = "./cifar10_data"
    model_out: str = "best_model.pth"

cfg = TrainConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def make_transforms(augment: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf

def make_loaders(cfg: TrainConfig):
    train_tf, test_tf = make_transforms(cfg.augment)

    train_ds = CIFAR10(root=cfg.data_dir, train=True, download=True, transform=train_tf)
    test_ds = CIFAR10(root=cfg.data_dir, train=False, download=True, transform=test_tf)

    val_size = 5000
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                             num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, 
                           num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, 
                            num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = make_loaders(cfg)
print(f"Treino: {len(train_loader.dataset)} imagens")
print(f"Validação: {len(val_loader.dataset)} imagens")
print(f"Teste: {len(test_loader.dataset)} imagens")

def show_sample_images(loader):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        ax = axes[i//5, i%5]
        img = images[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f'{class_names[labels[i]]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

show_sample_images(train_loader)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.block3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = CIFAR10CNN(num_classes=10).to(device)
print(f"Modelo criado com {sum(p.numel() for p in model.parameters()):,} parâmetros")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        acc = correct / labels.size(0)
        
        running_loss += loss.item() * images.size(0)
        running_acc += acc * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            acc = correct / labels.size(0)
            
            running_loss += loss.item() * images.size(0)
            running_acc += acc * images.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)

    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

print("Iniciando treinamento...\n")

best_val_acc = 0.0
history_train_loss, history_val_loss = [], []
history_train_acc, history_val_acc = [], []

for epoch in range(1, cfg.epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
    
    history_train_loss.append(train_loss)
    history_val_loss.append(val_loss)
    history_train_acc.append(train_acc)
    history_val_acc.append(val_acc)
    
    scheduler.step()
    
    print(f"Época {epoch:02d}/{cfg.epochs} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc,
        }, cfg.model_out)
        print(f"  → Novo melhor modelo salvo! (Acc: {val_acc:.4f})")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_train_loss, label='Treino', linewidth=2, color='blue')
plt.plot(history_val_loss, label='Validação', linewidth=2, color='red')
plt.xlabel('Época')
plt.ylabel('Erro Médio (Loss)')
plt.title('Curva do Erro Médio durante o Treinamento')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history_train_acc, label='Treino', linewidth=2, color='green')
plt.plot(history_val_acc, label='Validação', linewidth=2, color='orange')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.title('Curva de Acurácia durante o Treinamento')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('curva_erro_medio.png', dpi=300, bbox_inches='tight')
plt.show()

print("Avaliando no conjunto de teste...")

test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)

print(f"\nResultados no Teste:")
print(f"Erro Médio (Loss): {test_loss:.4f}")
print(f"Acurácia: {test_acc:.4f}")

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Matriz de Confusão - Conjunto de Testes CIFAR-10', fontsize=14, pad=20)
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha='right')
plt.yticks(tick_marks, class_names)

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8)

plt.tight_layout()
plt.ylabel('Rótulo Verdadeiro', fontsize=12)
plt.xlabel('Rótulo Predito', fontsize=12)
plt.savefig('matriz_confusao_testes.png', dpi=300, bbox_inches='tight')
plt.show()

metrics = {
    'erro_medio_teste': round(test_loss, 4),
    'acuracia_teste': round(test_acc, 4),
    'melhor_acuracia_validacao': round(best_val_acc, 4),
    'epochs_treinamento': cfg.epochs,
    'batch_size': cfg.batch_size,
    'learning_rate': cfg.lr,
}

print("\nMétricas Principais:")
print("=" * 40)
for key, value in metrics.items():
    print(f"{key}: {value}")

with open('metricas_avaliacao.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nArquivos gerados:")
print("1. curva_erro_medio.png - Curva do erro médio durante o treinamento")
print("2. matriz_confusao_testes.png - Matriz de confusão do conjunto de testes")
print("3. metricas_avaliacao.json - Métricas de avaliação")
print("4. best_model.pth - Melhores pesos do modelo")

print("\n Treinamento e avaliação concluídos com sucesso!")
