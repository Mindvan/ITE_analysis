import unittest
import json
import os
from app import app

class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        """Подготовка тестового клиента"""
        self.app = app.test_client()
        self.app.testing = True
        
    def test_home_status_code(self):
        """Тест доступности главной страницы"""
        result = self.app.get('/')
        self.assertEqual(result.status_code, 200)
        
    def test_api_upload_empty_data(self):
        """Тест загрузки пустых данных"""
        result = self.app.post('/api/upload', 
                             data=json.dumps({}),
                             content_type='application/json')
        self.assertEqual(result.status_code, 400)
        
    def test_api_upload_valid_data(self):
        """Тест загрузки валидных данных"""
        test_data = {
            'experimentData': {
                'sessionId': 'test_session',
                'browser': {'name': 'test_browser'},
                'tasks': [],
                'gazeData': [],
                'emotionData': []
            }
        }
        result = self.app.post('/api/upload',
                             data=json.dumps(test_data),
                             content_type='application/json')
        self.assertEqual(result.status_code, 200)
        data = json.loads(result.data)
        self.assertIn('session_id', data)
        
    def test_api_analyze_empty_data(self):
        """Тест анализа без данных"""
        result = self.app.post('/api/analyze')
        self.assertEqual(result.status_code, 200)
        data = json.loads(result.data)
        self.assertIn('results', data)
        
    def test_api_analyze_with_session(self):
        """Тест анализа с указанием сессии"""
        # Сначала создаем тестовую сессию
        test_data = {
            'experimentData': {
                'sessionId': 'test_session',
                'browser': {'name': 'Chrome'},
                'tasks': [{'completed': True, 'duration': 1000}],
                'gazeData': [{'x': 100, 'y': 100}],
                'emotionData': [{'expressions': {'happy': 0.8}}]
            }
        }
        self.app.post('/api/upload',
                     data=json.dumps(test_data),
                     content_type='application/json')
        
        # Теперь анализируем эту сессию
        result = self.app.post('/api/analyze',
                             data=json.dumps({'session_id': 'test_session'}),
                             content_type='application/json')
        self.assertEqual(result.status_code, 200)
        data = json.loads(result.data)
        self.assertIn('results', data)
        
if __name__ == '__main__':
    unittest.main() 