# test_app.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from io import BytesIO

from app import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests pour l'endpoint /health"""
    
    def test_health_success(self):
        """Test si le service est prêt"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        
    def test_health_structure(self):
        """Test la structure de la réponse health"""
        response = client.get("/health")
        data = response.json()
        
        if data["ready"]:
            assert "model" in data
        else:
            assert "error" in data


class TestPredictNextDay:
    """Tests pour l'endpoint /predict/next_day"""
    
    @patch('app.yf.download')
    @patch('app.model')
    @patch('app.cs_y')
    def test_predict_next_day_success(self, mock_cs_y, mock_model, mock_yf):
        """Test prédiction réussie avec données valides"""
        # Mock des données Yahoo Finance
        dates = pd.date_range(start='2020-01-01', end='2025-11-29', freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.uniform(1.2, 1.4, len(dates)),
            'High': np.random.uniform(1.25, 1.45, len(dates)),
            'Low': np.random.uniform(1.15, 1.35, len(dates)),
            'Close': np.random.uniform(1.2, 1.4, len(dates)),
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        mock_yf.return_value = mock_data
        
        # Mock du modèle
        mock_model.predict.return_value = np.array([[1.35]])
        mock_cs_y.inverse_transform.return_value = np.array([[1.35]])
        
        response = client.get("/predict/next_day?ticker=USDCAD=X&start=2020-01-01&end=2025-11-29")
        
        assert response.status_code == 200
        data = response.json()
        assert "next_date" in data
        assert "next_close_pred" in data
        assert isinstance(data["next_close_pred"], float)
        
    def test_predict_next_day_invalid_ticker(self):
        """Test avec un ticker invalide"""
        with patch('app.yf.download') as mock_yf:
            mock_yf.return_value = pd.DataFrame()  # DataFrame vide
            
            response = client.get("/predict/next_day?ticker=INVALID&start=2020-01-01")
            assert response.status_code == 400
            assert "Aucune donnée" in response.json()["detail"]
    
    def test_predict_next_day_custom_dates(self):
        """Test avec des dates personnalisées"""
        with patch('app.yf.download') as mock_yf, \
             patch('app.model') as mock_model, \
             patch('app.cs_y') as mock_cs_y:
            
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            mock_data = pd.DataFrame({
                'Open': np.random.uniform(1.2, 1.4, len(dates)),
                'High': np.random.uniform(1.25, 1.45, len(dates)),
                'Low': np.random.uniform(1.15, 1.35, len(dates)),
                'Close': np.random.uniform(1.2, 1.4, len(dates)),
                'Volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
            mock_yf.return_value = mock_data
            mock_model.predict.return_value = np.array([[1.30]])
            mock_cs_y.inverse_transform.return_value = np.array([[1.30]])
            
            response = client.get("/predict/next_day?ticker=EURUSD=X&start=2023-01-01&end=2023-12-31")
            assert response.status_code == 200


class TestPredictHistory:
    """Tests pour l'endpoint /predict/history"""
    
    @patch('app.yf.download')
    @patch('app.model')
    @patch('app.cs_y')
    def test_predict_history_success(self, mock_cs_y, mock_model, mock_yf):
        """Test récupération de l'historique des prédictions"""
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.uniform(1.2, 1.4, len(dates)),
            'High': np.random.uniform(1.25, 1.45, len(dates)),
            'Low': np.random.uniform(1.15, 1.35, len(dates)),
            'Close': np.random.uniform(1.2, 1.4, len(dates)),
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        mock_yf.return_value = mock_data
        
        # Mock prédictions
        n_predictions = len(dates) - 60
        mock_model.predict.return_value = np.random.uniform(1.2, 1.4, (n_predictions, 1))
        mock_cs_y.inverse_transform.return_value = np.random.uniform(1.2, 1.4, (n_predictions, 1))
        
        response = client.get("/predict/history?ticker=USDCAD=X&start=2024-01-01&end=2024-03-31")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Vérifier la structure d'un élément
        first_item = data[0]
        assert "date" in first_item
        assert "real_close" in first_item
        assert "pred_close" in first_item
        assert "abs_error" in first_item
        assert "pct_error" in first_item
        
    def test_predict_history_error_calculations(self):
        """Test que les erreurs sont bien calculées"""
        with patch('app.yf.download') as mock_yf, \
             patch('app.model') as mock_model, \
             patch('app.cs_y') as mock_cs_y:
            
            dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
            mock_data = pd.DataFrame({
                'Open': np.random.uniform(1.2, 1.4, len(dates)),
                'High': np.random.uniform(1.25, 1.45, len(dates)),
                'Low': np.random.uniform(1.15, 1.35, len(dates)),
                'Close': np.random.uniform(1.2, 1.4, len(dates)),
                'Volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
            mock_yf.return_value = mock_data
            
            n_predictions = len(dates) - 60
            mock_model.predict.return_value = np.random.uniform(1.2, 1.4, (n_predictions, 1))
            mock_cs_y.inverse_transform.return_value = np.random.uniform(1.2, 1.4, (n_predictions, 1))
            
            response = client.get("/predict/history")
            data = response.json()
            
            # Vérifier qu'il y a des erreurs calculées
            for item in data:
                assert item["abs_error"] >= 0
                assert item["pct_error"] >= 0


class TestPredictUploadCSV:
    """Tests pour l'endpoint /predict/upload_csv"""
    
    @patch('app.model')
    @patch('app.cs_y')
    def test_upload_csv_success(self, mock_cs_y, mock_model):
        """Test upload CSV réussi"""
        # Créer un CSV de test
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        df = pd.DataFrame({
            '<DATE>': dates,
            'open': np.random.uniform(1.2, 1.4, len(dates)),
            'high': np.random.uniform(1.25, 1.45, len(dates)),
            'low': np.random.uniform(1.15, 1.35, len(dates)),
            'close': np.random.uniform(1.2, 1.4, len(dates)),
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        csv_content = df.to_csv(sep='\t', index=False).encode('utf-8')
        
        mock_model.predict.return_value = np.array([[1.35]])
        mock_cs_y.inverse_transform.return_value = np.array([[1.35]])
        
        files = {'file': ('test.csv', BytesIO(csv_content), 'text/csv')}
        response = client.post("/predict/upload_csv", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "next_date" in data
        assert "next_close_pred" in data
        assert "info" in data
        assert "test.csv" in data["info"]
        
    def test_upload_csv_insufficient_columns(self):
        """Test avec CSV ayant trop peu de colonnes"""
        df = pd.DataFrame({
            '<DATE>': ['2024-01-01', '2024-01-02'],
            'open': [1.2, 1.3],
            'close': [1.25, 1.35]
        })
        
        csv_content = df.to_csv(sep='\t', index=False).encode('utf-8')
        files = {'file': ('test.csv', BytesIO(csv_content), 'text/csv')}
        
        response = client.post("/predict/upload_csv", files=files)
        assert response.status_code == 400
        assert "au moins 5 colonnes" in response.json()["detail"]
        
    def test_upload_csv_comma_separated(self):
        """Test avec CSV séparé par des virgules"""
        with patch('app.model') as mock_model, \
             patch('app.cs_y') as mock_cs_y:
            
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            df = pd.DataFrame({
                'date': dates,
                'open': np.random.uniform(1.2, 1.4, len(dates)),
                'high': np.random.uniform(1.25, 1.45, len(dates)),
                'low': np.random.uniform(1.15, 1.35, len(dates)),
                'close': np.random.uniform(1.2, 1.4, len(dates)),
                'volume': np.random.randint(1000, 10000, len(dates))
            })
            
            csv_content = df.to_csv(index=False).encode('utf-8')
            
            mock_model.predict.return_value = np.array([[1.35]])
            mock_cs_y.inverse_transform.return_value = np.array([[1.35]])
            
            files = {'file': ('test.csv', BytesIO(csv_content), 'text/csv')}
            response = client.post("/predict/upload_csv", files=files)
            
            assert response.status_code == 200


class TestErrorHandling:
    """Tests de gestion des erreurs"""
    
    def test_model_not_loaded(self):
        """Test quand le modèle n'est pas chargé"""
        with patch('app.model', None), \
             patch('app.load_error', "Model file not found"):
            
            response = client.get("/predict/next_day")
            assert response.status_code == 500
            assert "Artifacts non chargés" in response.json()["detail"]
    
    @patch('app.yf.download')
    def test_insufficient_data_for_window(self, mock_yf):
        """Test avec trop peu de données pour créer une fenêtre"""
        # Seulement 30 jours de données (moins que la fenêtre de 60)
        dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.uniform(1.2, 1.4, len(dates)),
            'High': np.random.uniform(1.25, 1.45, len(dates)),
            'Low': np.random.uniform(1.15, 1.35, len(dates)),
            'Close': np.random.uniform(1.2, 1.4, len(dates)),
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        mock_yf.return_value = mock_data
        
        response = client.get("/predict/next_day?start=2024-01-01&end=2024-01-30")
        assert response.status_code == 500


# Tests d'intégration
class TestIntegration:
    """Tests d'intégration de bout en bout"""
    
    @pytest.mark.skipif(
        not hasattr(app, 'model') or app.model is None,
        reason="Modèle non chargé"
    )
    def test_full_prediction_workflow(self):
        """Test du workflow complet de prédiction"""
        # 1. Vérifier la santé
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Faire une prédiction
        if health_response.json()["ready"]:
            pred_response = client.get("/predict/next_day?ticker=USDCAD=X")
            assert pred_response.status_code in [200, 500]  # Peut échouer si Yahoo est down


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])