from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import uuid
from datetime import datetime
from analysis import ITEAnalyzer
from werkzeug.utils import secure_filename
import base64
import csv

app = Flask(__name__)
CORS(app)  # Разрешаем CORS для фронтенда

# Конфигур.
UPLOAD_FOLDER = 'data/experiments'
RESULTS_FOLDER = 'data/results'
ALLOWED_EXTENSIONS = {'json'}

# Создаем папки если их нет
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Инициализируем анализатор
analyzer = ITEAnalyzer()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.getcwd(), filename)

@app.route('/')
def index():
    return jsonify({
        "message": "UX Experiment Backend API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/upload",
            "analyze": "/api/analyze",
            "results": "/api/results/<session_id>"
        }
    })

@app.route('/api/upload', methods=['POST'])
def upload_experiment_data():
    """Принимает JSON данные эксперимента от фронтенда"""
    try:
        # Прверяем наличие данных
        if 'experimentData' not in request.json:
            return jsonify({'error': 'No experiment data provided'}), 400
        
        data = request.json['experimentData']
        session_id = data.get('sessionId', str(uuid.uuid4()))
        
        # Сохраняем данные
        filename = f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'filename': filename,
            'message': f'Experiment data saved successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_experiment():
    """Запускает ITE анализ для указанной сессии, списка сессий или для всех сессий, если ничего не передано"""
    try:
        if request.is_json:
            data = request.json
            if 'experimentDataList' in data:
                # Анализируем список экспериментов
                results = analyzer.analyze_experiment_list(data['experimentDataList'])
            elif 'experimentData' in data:
                # Анализируем один эксперимент
                results = analyzer.analyze_experiment(data['experimentData'])
            elif 'session_id' in data and data['session_id']:
                session_id = data['session_id']
                experiment_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                                  if f.startswith(session_id)]
                if not experiment_files:
                    return jsonify({'error': 'Experiment data not found'}), 404
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], experiment_files[0])
                with open(filepath, 'r', encoding='utf-8') as f:
                    exp_data = json.load(f)
                results = analyzer.analyze_experiment(exp_data)
            else:
                # Анализируем все сессии
                results = analyzer.analyze_experiment()
        else:
            # Анализируем все сессии
            results = analyzer.analyze_experiment()

        # --- Сохранение результатов --- #
        # Определяем имя файла для результатов (используем текущее время)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_filename_base = f"results_{timestamp_str}"
        results_json_filename = f"{results_filename_base}.json"
        results_filepath_json = os.path.join(app.config['RESULTS_FOLDER'], results_json_filename)

        # Сохраняем JSON результаты
        with open(results_filepath_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Сохраняем графики и таблицы для каждого варианта X
        for key in ['full_features', 'no_gaze', 'no_emotion']:
            block = results.get(key)
            if not block:
                continue
            # Сохраняем boxplot
            for viz_name, plot_base64 in block.get('visualizations', {}).items():
                try:
                    if ',' in plot_base64:
                        plot_base64 = plot_base64.split(',')[1]
                    plot_binary_data = base64.b64decode(plot_base64)
                    results_png_filename = f"{results_filename_base}_{key}_{viz_name}.png"
                    results_filepath_png = os.path.join(app.config['RESULTS_FOLDER'], results_png_filename)
                    with open(results_filepath_png, 'wb') as f:
                        f.write(plot_binary_data)
                except Exception as e:
                    print(f"Ошибка при сохранении PNG графика ({key}): {e}")
            # Сохраняем таблицу средних эффектов и std (только на английском)
            try:
                mean_csv = f"{results_filename_base}_{key}_ite_mean.csv"
                std_csv = f"{results_filename_base}_{key}_ite_std.csv"
                with open(os.path.join(app.config['RESULTS_FOLDER'], mean_csv), 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Model', 'Mean'])
                    for k, v in block.get('ite_mean', {}).items():
                        # k уже на английском, т.к. формируется в analysis.py
                        writer.writerow([k, v])
                with open(os.path.join(app.config['RESULTS_FOLDER'], std_csv), 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Model', 'Std'])
                    for k, v in block.get('ite_std', {}).items():
                        writer.writerow([k, v])
            except Exception as e:
                print(f"Ошибка при сохранении CSV ({key}): {e}")

        # --- Возвращаем ответ фронтенду ---
        return jsonify({
            'status': 'success',
            'results': results,
            'results_file': results_json_filename # Возвращаем имя JSON файла
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<session_id>')
def get_results(session_id):
    """Получает результаты анализа для сессии"""
    try:
        # Ищем файл с результатами
        results_files = [f for f in os.listdir(app.config['RESULTS_FOLDER']) 
                        if f.startswith(f'results_{session_id}')]
        
        if not results_files:
            return jsonify({'error': 'Results not found'}), 404
        
        # берм самый свежий файл
        latest_file = sorted(results_files)[-1]
        filepath = os.path.join(app.config['RESULTS_FOLDER'], latest_file)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/experiments')
def list_experiments():
    """Список всех экспериментов"""
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        experiments = []
        
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    experiments.append({
                        'filename': file,
                        'session_id': data.get('sessionId'),
                        'timestamp': data.get('timestamp'),
                        'browser': data.get('browser', {}).get('name'),
                        'tasks_count': len(data.get('tasks', []))
                    })
        
        return jsonify({
            'status': 'success',
            'experiments': experiments
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 