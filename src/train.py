"""
Training di un modello ResNet per la classificazione di immagini Fashion-MNIST.
Questo modulo contiene funzioni per caricare, addestrare e valutare un modello
di classificazione di abbigliamento basato su ResNet18.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image

def cretate_model_dir():
	"""Crea la directory models/ se non esiste."""
	if not os.path.exists("models"):
		os.makedirs("models")
		print("Directory 'models/' creata.")

def get_data_loaders(batch_size=64, subset_size=30000):
    """Carica e preprocessa Fashion-MNIST"""
    
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)), # 224x224 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Dataset
    train_dataset = torchvision.datasets.FashionMNIST(
        './data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        './data', train=False, download=True, transform=transform_test
    )
    
	# Riduci dataset per training veloce
    train_indices = torch.randperm(len(train_dataset))[:subset_size]
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    
    test_indices = torch.randperm(len(test_dataset))[:1000]  
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    print(f"Using reduced dataset: {len(train_dataset)} train, {len(test_dataset)} test")

    # Split train/validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = random_split(train_dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_model():
    """Crea modello ResNet18"""
    #model = models.resnet18(pretrained=True)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Adatta per grayscale e 10 classi
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Training per una epoca"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion, device):
    """Validation"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), 100 * correct / total

def train_model(epochs=10):
    """Funzione principale di training"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dati
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Modello
    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0
    
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Salva metriche
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Salva miglior modello
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"✓ Best model saved! Acc: {val_acc:.2f}%")
    
    # Test finale
    print("\nTesting...")
    model.load_state_dict(torch.load('models/best_model.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Grafici
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Val')
    plt.title('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_plots.png')
    print("Plots saved to models/training_plots.png")
    
    return model

def predict_image(image_path):
    """Predizione su singola immagine"""
    
    # Carica modello
    model = get_model()
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.eval()
    
    # Preprocessa immagine
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0)
    
    # Predizione
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(dim=1).item()
    
    # Classi
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    return classes[prediction]

def compute_metrics_pytorch(y_true, y_pred):
    """Calcola metriche usando solo PyTorch/NumPy"""
    
    # Converti in numpy se necessario
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Per classe
    num_classes = len(np.unique(y_true))
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for class_id in range(num_classes):
        # True/False Positives/Negatives
        tp = np.sum((y_true == class_id) & (y_pred == class_id))
        fp = np.sum((y_true != class_id) & (y_pred == class_id))
        fn = np.sum((y_true == class_id) & (y_pred != class_id))
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
    
    # Media pesata (weighted average)
    class_counts = [np.sum(y_true == i) for i in range(num_classes)]
    total_samples = len(y_true)
    
    weighted_precision = np.average(precision_per_class, weights=class_counts)
    weighted_recall = np.average(recall_per_class, weights=class_counts)
    weighted_f1 = np.average(f1_per_class, weights=class_counts)
    
    return {
        "accuracy": accuracy,
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1": weighted_f1
    }

def create_classification_report(y_true, y_pred, class_names):
    """Crea un report di classificazione semplice"""
    
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    report = "\nClassification Report:\n"
    report += f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n"
    report += "-" * 65 + "\n"
    
    for class_id, class_name in enumerate(class_names):
        tp = np.sum((y_true == class_id) & (y_pred == class_id))
        fp = np.sum((y_true != class_id) & (y_pred == class_id))
        fn = np.sum((y_true == class_id) & (y_pred != class_id))
        support = np.sum(y_true == class_id)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        report += f"{class_name:<15} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10}\n"
    
    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    report += f"\nOverall Accuracy: {accuracy:.3f}"
    
    return report

def evaluate_model(model, test_loader):
    """Valuta il modello senza sklearn"""
    print("\nValutazione del modello sul test set...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calcola metriche con PyTorch nativo
    test_results = compute_metrics_pytorch(all_labels, all_predictions)
    
    print("\n===== RISULTATI DELLA VALUTAZIONE =====")
    for metric_name, metric_value in test_results.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Report per classe
    report = create_classification_report(all_labels, all_predictions, class_names)
    print(report)
    
    return test_results

if __name__ == "__main__":
    print("🚀 Starting Fashion-MNIST training...")
    model = train_model(epochs=10)
    print("✅ Training completed!")