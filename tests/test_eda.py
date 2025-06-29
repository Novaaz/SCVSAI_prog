import pytest
import torch
import pandas as pd
import sys
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from eda import *

class TestEDA:
    
    def test_create_data_dir(self):
        """Testa la creazione della directory data"""
        # Rimuovi directory se esiste
        import shutil
        if os.path.exists('test_data'):
            shutil.rmtree('test_data')
            
        # Crea directory
        create_data_dir()
        
        # Verifica che esista
        assert os.path.exists('data')
        
    def test_load_data_basic(self):
        """Testa il caricamento base dei dati"""
        train_dataset, test_dataset = load_data()
        
        # Verifica che non siano vuoti
        assert len(train_dataset) > 0
        assert len(test_dataset) > 0
        
        # Verifica primo elemento
        image, label = train_dataset[0]
        assert image.shape[0] == 1  # grayscale
        assert 0 <= label <= 9
        
    def test_analyze_data_small(self):
        """Testa l'analisi su dataset piccolo"""
        train_dataset, test_dataset = load_data()
        
        # Usa subset molto piccolo
        train_subset = torch.utils.data.Subset(train_dataset, range(50))
        test_subset = torch.utils.data.Subset(test_dataset, range(20))
        
        train_df, pixel_stats = analyze_data(train_subset, test_subset)
        
        # Verifica DataFrame
        assert isinstance(train_df, pd.DataFrame)
        assert len(train_df) == 10  # 10 classi
        assert all(col in train_df.columns for col in ['class_name', 'train_count', 'test_count'])
        
        # Verifica pixel stats
        assert isinstance(pixel_stats, dict)
        assert 'pixels' in pixel_stats
        assert 'image_means' in pixel_stats
        assert len(pixel_stats['image_means']) > 0

    def test_visualize_data(self):
        """Testa la creazione delle visualizzazioni"""
        # Usa dataset molto piccolo
        train_dataset, test_dataset = load_data()
        train_subset = torch.utils.data.Subset(train_dataset, range(100))
        
        train_df, pixel_stats = analyze_data(train_subset, test_dataset)
        
        # Mock plt.savefig per evitare di salvare file reali
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            with patch('matplotlib.pyplot.show'):  # Evita finestre popup
                visualize_data(train_subset, train_df, pixel_stats)
                
                # Verifica che i grafici siano stati "salvati"
                assert mock_savefig.call_count >= 3  # Almeno 3 grafici
    
    def test_analyze_data_detailed(self):
        """Testa analisi dati con controlli dettagliati"""
        train_dataset, test_dataset = load_data()
        
        # Subset ancora più piccolo per velocità
        train_subset = torch.utils.data.Subset(train_dataset, range(200))
        test_subset = torch.utils.data.Subset(test_dataset, range(100))
        
        train_df, pixel_stats = analyze_data(train_subset, test_subset)
        
        # Test dettagliati su train_df
        assert 'train_percentage' in train_df.columns
        assert train_df['train_percentage'].sum() <= 100.1  # Piccola tolleranza float
        assert all(train_df['train_count'] >= 0)
        assert all(train_df['test_count'] >= 0)
        
        # Test dettagliati su pixel_stats
        assert len(pixel_stats['pixels']) > 0
        assert len(pixel_stats['image_means']) > 0
        assert 0 <= pixel_stats['pixels'].mean() <= 1  # Valori normalizzati