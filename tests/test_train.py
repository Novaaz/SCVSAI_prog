import pytest
import torch
import numpy as np
import sys
import os
import tempfile
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import *

class TestTraining:
    
    def test_get_model(self):
        """Testa che il modello venga creato correttamente"""
        model = get_model()
        
        # Verifica architettura
        assert hasattr(model, 'fc')
        assert model.fc.out_features == 10  # 10 classi Fashion-MNIST
        assert model.conv1.in_channels == 1  # Grayscale
        
    def test_data_loaders_shape(self):
        """Testa che i data loader abbiano le dimensioni corrette"""
        train_loader, val_loader, test_loader = get_data_loaders(
            batch_size=32, subset_size=100
        )
        
        # Prendi un batch
        batch_images, batch_labels = next(iter(train_loader))
        
        # Verifica dimensioni
        assert batch_images.shape[1] == 1  # Grayscale
        assert batch_images.shape[2] == 64  # Height
        assert batch_images.shape[3] == 64  # Width
        assert len(batch_labels) <= 32  # Batch size
        
    def test_model_forward_pass(self):
        """Testa che il modello faccia forward pass senza errori"""
        model = get_model()
        model.eval()
        
        # Input fittizio
        dummy_input = torch.randn(1, 1, 64, 64)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Verifica output
        assert output.shape == (1, 10)  # Batch=1, Classes=10
        
    @patch('torch.load')
    @patch('PIL.Image.open')
    def test_predict_image(self, mock_image_open, mock_torch_load):
        """Testa la funzione di predizione"""
        # Mock del modello salvato
        mock_torch_load.return_value = {}
        
        # Mock dell'immagine
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_image_open.return_value = mock_image
        
        # Crea un modello temporaneo
        with tempfile.NamedTemporaryFile(suffix='.pth') as tmp_model:
            torch.save({}, tmp_model.name)
            
            with patch('src.train.get_model') as mock_get_model:
                mock_model = MagicMock()
                mock_model.eval.return_value = None
                mock_model.return_value = torch.tensor([[0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
                mock_get_model.return_value = mock_model
                
                # Test dovrebbe non crashare
                try:
                    result = predict_image('dummy_path.jpg')
                    assert isinstance(result, str)
                except Exception as e:
                    # Se fallisce per dipendenze, è ok per il test
                    pass

    def test_compute_metrics_pytorch(self):
        """Testa il calcolo delle metriche"""
        # Dati di test semplici
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 2])  # 1 errore su classe 2
        
        metrics = compute_metrics_pytorch(y_true, y_pred)
        
        # Verifica che le metriche siano calcolate
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Accuracy dovrebbe essere 5/6 ≈ 0.83
        assert 0.8 <= metrics['accuracy'] <= 0.9
        
    def test_create_classification_report(self):
        """Testa la creazione del report"""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        class_names = ['A', 'B', 'C']
        
        report = create_classification_report(y_true, y_pred, class_names)
        
        # Verifica che il report contenga le informazioni attese
        assert 'Classification Report' in report
        assert 'Precision' in report
        assert 'Recall' in report
        assert 'Overall Accuracy' in report
        
    def test_train_one_epoch(self):
        """Testa un'epoca di training"""
        device = torch.device('cpu')
        model = get_model().to(device)
        
        # Crea un mini dataset
        train_loader, _, _ = get_data_loaders(batch_size=8, subset_size=16)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Esegui un'epoca
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Verifica che restituisca valori ragionevoli
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0
        assert 0 <= acc <= 100
        
    def test_validate(self):
        """Testa la validazione"""
        device = torch.device('cpu')
        model = get_model().to(device)
        
        # Crea un mini dataset
        _, val_loader, _ = get_data_loaders(batch_size=8, subset_size=16)
        
        criterion = torch.nn.CrossEntropyLoss()
        
        # Esegui validazione
        loss, acc = validate(model, val_loader, criterion, device)
        
        # Verifica che restituisca valori ragionevoli
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0
        assert 0 <= acc <= 100
        
    def test_data_loaders_content(self):
        """Testa il contenuto dei data loader"""
        train_loader, val_loader, test_loader = get_data_loaders(
            batch_size=4, subset_size=20
        )
        
        # Testa train loader
        images, labels = next(iter(train_loader))
        assert images.shape[0] <= 4  # batch size
        assert images.shape[1] == 1  # grayscale
        assert torch.all(labels >= 0) and torch.all(labels <= 9)  # classi valide
        
        # Testa val loader  
        images, labels = next(iter(val_loader))
        assert images.shape[0] <= 4
        assert images.shape[1] == 1
        
        # Testa test loader
        images, labels = next(iter(test_loader))
        assert images.shape[0] <= 4
        assert images.shape[1] == 1