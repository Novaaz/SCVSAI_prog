"""
Exploratory Data Analysis (EDA) per il dataset Fashion-MNIST.
Questo modulo contiene funzioni per caricare, analizzare e visualizzare
il dataset Fashion-MNIST utilizzando torchvision.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import numpy as np
from collections import Counter

def create_data_dir():
    """
    Crea la directory data/ se non esiste.
    
    """
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Directory 'data/' creata.")

def load_data():
    """
    Carica il dataset Fashion-MNIST da torchvision.
    
    Returns:
        tuple: (train_dataset, test_dataset) contenenti i dati.
    """
    print("Caricamento del dataset Fashion-MNIST...")
    
    transform = transforms.ToTensor()
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    print(f"Dataset caricato. Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, test_dataset

def analyze_data(train_dataset, test_dataset):
    """
    Esegue un'analisi esplorativa sul dataset.
    
    Args:
        train_dataset: Dataset di training.
        test_dataset: Dataset di test.
        
    Returns:
        pd.DataFrame: DataFrame con i dati per ulteriori analisi.
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Estrai le etichette
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]
    
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    
    print("\n===== STATISTICHE DI BASE =====")
    print(f"Dimensione train set: {len(train_dataset)} esempi")
    print(f"Dimensione test set: {len(test_dataset)} esempi")
    print(f"Classi disponibili: {class_names}")
    
    # Verifica bilanciamento
    is_balanced = len(set(train_counts.values())) == 1
    print(f"Dataset bilanciato: {'Sì' if is_balanced else 'No'}")
    
    # Analisi pixel su campione
    sample_size = 2000
    pixels = []
    image_means = []
    
    print(f"\n===== ANALISI PIXEL (campione {sample_size}) =====")
    for i in range(min(sample_size, len(train_dataset))):
        image, _ = train_dataset[i]
        image_np = image.squeeze().numpy()
        image_means.append(image_np.mean())
        
        if i < 500:  # Limita per memoria
            pixels.extend(image_np.flatten()[::50])
    
    pixels = np.array(pixels)
    image_means = np.array(image_means)
    
    print(f"Media pixel globale: {pixels.mean():.4f}")
    print(f"Deviazione standard: {pixels.std():.4f}")
    print(f"Range valori: [{pixels.min():.3f}, {pixels.max():.3f}]")
    print(f"Media delle medie per immagine: {image_means.mean():.4f}")
    
    # Crea DataFrame per analisi
    train_df = pd.DataFrame({
        'class_id': list(range(len(class_names))),
        'class_name': class_names,
        'train_count': [train_counts[i] for i in range(len(class_names))],
        'test_count': [test_counts[i] for i in range(len(class_names))]
    })
    
    train_df['train_percentage'] = (train_df['train_count'] / len(train_dataset) * 100).round(2)
    
    return train_df, {'pixels': pixels, 'image_means': image_means}

def visualize_data(train_dataset, train_df, pixel_stats):
    """
    Crea e salva visualizzazioni del dataset.
    
    Args:
        train_dataset: Dataset di training.
        train_df: DataFrame contenente le statistiche delle classi.
        pixel_stats: Dizionario con statistiche sui pixel.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Distribuzione delle classi
    plt.figure(figsize=(12, 6))
    bars = plt.bar(train_df['class_name'], train_df['train_count'])
    plt.title('Distribuzione delle classi nel dataset')
    plt.xlabel('Classe')
    plt.ylabel('Conteggio')
    plt.xticks(rotation=45)
    
    for bar, count in zip(bars, train_df['train_count']):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                f'{count}', ha='center', va='bottom')
    
    try:
        plt.tight_layout()
    except UserWarning:
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig('data/class_distribution.png')
    print("Salvato grafico: data/class_distribution.png")
    
    # 2. Esempi per classe
    plt.figure(figsize=(15, 8))
    _, axes = plt.subplots(2, 5, figsize=(15, 8))
    axes = axes.ravel()
    
    class_examples = {}
    for i, (image, label) in enumerate(train_dataset):
        if label not in class_examples:
            class_examples[label] = image.squeeze().numpy()
        if len(class_examples) == 10:
            break
    
    for idx in range(10):
        axes[idx].imshow(class_examples[idx], cmap='gray')
        axes[idx].set_title(f'{train_df.iloc[idx]["class_name"]}')
        axes[idx].axis('off')
    
    plt.suptitle('Esempi rappresentativi per classe')
    plt.tight_layout()
    plt.savefig('data/class_examples.png')
    print("Salvato grafico: data/class_examples.png")
    
    # 3. Distribuzione valori pixel
    plt.figure(figsize=(12, 6))
    plt.hist(pixel_stats['pixels'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribuzione dei valori pixel')
    plt.xlabel('Valore pixel')
    plt.ylabel('Frequenza')
    plt.axvline(pixel_stats['pixels'].mean(), color='red', linestyle='--',
               label=f"Media: {pixel_stats['pixels'].mean():.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/pixel_distribution.png')
    print("Salvato grafico: data/pixel_distribution.png")
    
    # 4. Campioni multipli per alcune classi
    plt.figure(figsize=(12, 10))
    selected_classes = [0, 1, 2, 7]  # T-shirt, Trouser, Pullover, Sneaker
    
    class_samples = {i: [] for i in selected_classes}
    for idx, (image, label) in enumerate(train_dataset):
        if label in selected_classes and len(class_samples[label]) < 6:
            class_samples[label].append(image.squeeze().numpy())
        
        if all(len(samples) >= 6 for samples in class_samples.values()):
            break
    
    _, axes = plt.subplots(len(selected_classes), 6, figsize=(12, 8))
    
    for row, class_idx in enumerate(selected_classes):
        for col in range(6):
            axes[row, col].imshow(class_samples[class_idx][col], cmap='gray')
            if col == 0:
                axes[row, col].set_ylabel(train_df.iloc[class_idx]['class_name'])
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    
    plt.suptitle('Variazioni all\'interno delle classi')
    plt.tight_layout()
    plt.savefig('data/class_variations.png')
    print("Salvato grafico: data/class_variations.png")

def run_eda():
    """
    Esegue l'intero processo di EDA.
    """
    create_data_dir()
    train_dataset, test_dataset = load_data()
    train_df, pixel_stats = analyze_data(train_dataset, test_dataset)
    visualize_data(train_dataset, train_df, pixel_stats)
    
    print("\n===== RIEPILOGO FINALE =====")
    print("✅ Dataset Fashion-MNIST analizzato")
    print(f"✅ {len(train_dataset):,} immagini di training, {len(test_dataset):,} di test")
    print("✅ Tutti i grafici salvati in data/")
    
    print("\nAnalisi esplorativa completata.")
    return train_dataset, test_dataset

def main():
    """Entry point per EDA."""
    return run_eda()

if __name__ == "__main__":
    main()